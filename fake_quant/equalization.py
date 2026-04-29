"""
Per-channel equalization for down_proj activation quantization.

Implements the "bounded" equalization method from the Hailo SDK's APU equalization:
computes per-channel factors ∈ (0, 1] that stretch small channels and compress outlier
channels so all channels use the full quantization range.

Application is mathematically transparent:
    x_eq = x / factors          (online, per-channel)
    W_new[:, c] = W[:, c] * factors[c]   (weight compensation, one-time)
    W_new @ x_eq = W @ x       (unchanged output)
"""

import functools
import logging

import torch
import torch.nn as nn
from tqdm import tqdm

import model_utils
import quant_utils
import utils


class EqActivation(nn.Module):
    """Wrap an activation function with a per-output-channel rescale.

    Used for `--ugd_eq` when the inner activation is non-PWL (plain `nn.SiLU`,
    `nn.GELU`, etc.) — i.e. there is no HW per-channel `s_out` feature to
    bake the gate-side equalization factor into.  Instead, we apply
    ``y = act_fn(x) / eq_factors_g[c]`` explicitly in fp.  Down_proj's column
    rescale (`W_down[:, c] *= g[c] · u[c]`) compensates, so the math is
    transparent up to the `silu'(x) ≈ 1` approximation.
    """

    def __init__(self, act_fn: nn.Module, eq_factors_g: torch.Tensor):
        super().__init__()
        self.act_fn = act_fn
        # store as fp32 buffer; cast on use to match runtime dtype
        self.register_buffer('eq_factors_g',
                             eq_factors_g.detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act_fn(x)
        f = self.eq_factors_g.to(device=y.device, dtype=y.dtype)
        return y / f

    def extra_repr(self) -> str:
        f = self.eq_factors_g
        return (f"eq_g=[{f.min().item():.3f},{f.max().item():.3f}], "
                f"C={f.numel()}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_eq_factors_bounded(cmin, cmax, eps=1e-4):
    """Compute per-channel equalization factors using Hailo APU bounded method.

    Each factor is in (0, 1].  Dividing activations by these factors stretches
    smaller channels toward the global range, making static quantization scales
    more efficient.

    Args:
        cmin: per-channel minimums, shape [C]
        cmax: per-channel maximums, shape [C]
        eps:  threshold for dead channels / global range

    Returns:
        factors: tensor [C], values in (0, 1]
    """
    global_min = cmin.min().item()
    global_max = cmax.max().item()

    # Edge case: entire layer is dead (all zeros)
    if global_max <= eps and global_min >= -eps:
        return torch.ones_like(cmax)

    # Positive side: how much of global_max does each channel cover?
    f_max = torch.zeros_like(cmax)
    if global_max > eps:
        f_max = torch.clamp(cmax, min=0) / global_max

    # Negative side: how much of global_min does each channel cover?
    f_min = torch.zeros_like(cmin)
    if global_min < -eps:
        f_min = torch.clamp(cmin, max=0) / global_min

    # The allowed factor is the larger of the two bounds
    factors = torch.maximum(f_max, f_min)

    # Dead channels (factor ≈ 0) → leave unchanged
    factors = torch.where(factors <= eps, torch.ones_like(factors), factors)

    return factors


@torch.no_grad()
def collect_eq_stats(model, dataloader, nsamples, dev):
    """Forward pre-pass to collect per-channel min/max of down_proj inputs.

    Runs calibration data through the model layer-by-layer, collecting
    per-channel activation statistics for down_proj layers only.
    Uses the same Catcher pattern as calibrate_act_scales().

    Args:
        model: the (rotated) model with ActQuantWrapper layers
        dataloader: calibration data loader
        nsamples: number of samples to use
        dev: device to run on

    Returns:
        dict mapping full layer name → {'min': tensor [C], 'max': tensor [C]}
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    dtype = next(iter(model.parameters())).dtype
    seqlen = model.seqlen

    # --- Capture first-layer inputs using Catcher ---
    layers[0] = layers[0].to(dev)
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size),
                        dtype=dtype, device=dev)
    cache = {'i': 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            cache['position_embeddings'] = kwargs.get(
                'position_embeddings', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache.get('attention_mask', None)
    position_ids = cache.get('position_ids', None)
    position_embeddings = cache.get('position_embeddings', None)

    eq_stats = {}

    for i in tqdm(range(len(layers)), desc="Collecting eq stats"):
        layer = layers[i].to(dev)

        collectors = {}

        def collect_hook(module, inp, out, name):
            """Forward hook to collect min/max of pre-quantization input."""
            x = (module._cal_input if hasattr(module, '_cal_input')
                 and module._cal_input is not None
                 else (inp[0] if isinstance(inp, tuple) else inp))
            flat = x.reshape(-1, x.shape[-1]).float()
            cmin = flat.min(dim=0)[0]
            cmax = flat.max(dim=0)[0]
            if name not in collectors:
                collectors[name] = {'min': cmin, 'max': cmax}
            else:
                collectors[name]['min'] = torch.minimum(
                    collectors[name]['min'], cmin)
                collectors[name]['max'] = torch.maximum(
                    collectors[name]['max'], cmax)

        # Register hooks only on down_proj ActQuantWrapper layers
        hooks = []
        qlayers = quant_utils.find_qlayers(
            layer, layers=[quant_utils.ActQuantWrapper])
        for name, qlayer in qlayers.items():
            qlayer._calibrating = True
        for name, qlayer in qlayers.items():
            if 'down_proj' in name:
                full_name = f'model.layers.{i}.{name}'
                hooks.append(
                    qlayer.register_forward_hook(
                        functools.partial(collect_hook, name=full_name)))

        # Forward pass
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings)[0]

        # Cleanup
        for h in hooks:
            h.remove()
        for name, qlayer in qlayers.items():
            qlayer._calibrating = False
            qlayer._cal_input = None
            qlayer._cal_output = None

        eq_stats.update(collectors)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return eq_stats


def compute_eq_factors(eq_stats):
    """Compute equalization factors from collected statistics.

    Args:
        eq_stats: dict from collect_eq_stats()

    Returns:
        dict mapping layer name → factors tensor [C]
    """
    eq_factors = {}
    for name, stats in eq_stats.items():
        factors = compute_eq_factors_bounded(stats['min'], stats['max'])
        eq_factors[name] = factors.cpu()
        logger.info(f"  {name}: factor range [{factors.min():.4f}, "
                    f"{factors.max():.4f}], "
                    f"mean={factors.mean():.4f}")
    return eq_factors


def apply_eq_to_weights(model, eq_factors):
    """Modify down_proj weights to compensate for equalization.

    For each layer with equalization factors, multiplies each input channel
    of down_proj.weight by the corresponding factor:
        W_new[:, c] = W[:, c] * factors[c]

    This compensates for the online division x_eq = x / factors.

    Args:
        model: the model
        eq_factors: dict mapping layer name → factors tensor [C]
    """
    model_type = model_utils.model_type_extractor(model)
    layers = model_utils.get_transformer_layers(model, model_type=model_type)

    for i, layer_module in enumerate(layers):
        key = f'model.layers.{i}.mlp.down_proj.module'
        if key not in eq_factors:
            # Try without .module suffix
            key = f'model.layers.{i}.mlp.down_proj'
        if key not in eq_factors:
            continue

        factors = eq_factors[key]
        W = layer_module.mlp.down_proj
        # Handle ActQuantWrapper: access inner module's weight
        if isinstance(W, quant_utils.ActQuantWrapper):
            W = W.module
        dev = W.weight.device
        dtype = W.weight.dtype
        # W.weight shape: [out_features, in_features]
        # Multiply each input channel (column) by factors[c]
        W.weight.data = (W.weight.data.float()
                         * factors.to(dev).float().unsqueeze(0)
                         ).to(dtype)
        logger.info(f"  Applied eq factors to layer {i} down_proj weights")


def setup_eq_online(model, eq_factors):
    """Set eq_factors on ActQuantWrapper for online division in forward().

    Args:
        model: the model with ActQuantWrapper layers
        eq_factors: dict mapping layer name → factors tensor [C]
    """
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if 'down_proj' not in name:
            continue
        # Find matching key
        for key in eq_factors:
            if name in key or key in name:
                qlayers[name].eq_factors = eq_factors[key]
                logger.info(f"  Set online eq for {name}")
                break


# ---------------------------------------------------------------------------
# Branch-equalization (--ud_eq / --ugd_eq)
#
# Two new modes that don't use online division at down_proj input.  Instead,
# per-channel scales live on the up branch (W_up rows) and optionally the
# gate branch (PWL per-channel s_out via PWLActivation.eq_factors_g).
# Down_proj weight columns absorb the combined factor.
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_eq_stats_branches(model, dataloader, nsamples, dev, with_gate):
    """Forward pre-pass collecting per-channel min/max for the up branch
    (always) and the gate (silu output) branch (if `with_gate`).

    Returns a dict
        layer_idx -> {
            'up':   {'min': [C], 'max': [C]},
            'gate': {'min': [C], 'max': [C]},   # only if with_gate
        }
    where `layer_idx` is the integer index into model.model.layers — the gate
    and up MLP projections live as `mlp.gate_proj` / `mlp.up_proj`, and the
    PWL replacement of silu lives as `mlp.act_fn`.
    """
    import pwl_utils  # local to avoid circular import at module load time

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    dtype = next(iter(model.parameters())).dtype
    seqlen = model.seqlen

    # --- Capture first-layer inputs using Catcher (same pattern as collect_eq_stats) ---
    layers[0] = layers[0].to(dev)
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size),
                       dtype=dtype, device=dev)
    cache = {'i': 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            cache['position_embeddings'] = kwargs.get('position_embeddings', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache.get('attention_mask', None)
    position_ids = cache.get('position_ids', None)
    position_embeddings = cache.get('position_embeddings', None)

    branch_stats = {}

    for i in tqdm(range(len(layers)), desc="Collecting branch eq stats"):
        layer = layers[i].to(dev)

        per_layer = {}

        def update(name, x):
            flat = x.reshape(-1, x.shape[-1]).float()
            cmin = flat.min(dim=0)[0]
            cmax = flat.max(dim=0)[0]
            if name not in per_layer:
                per_layer[name] = {'min': cmin, 'max': cmax}
            else:
                per_layer[name]['min'] = torch.minimum(per_layer[name]['min'], cmin)
                per_layer[name]['max'] = torch.maximum(per_layer[name]['max'], cmax)

        # Hook up_proj output (post-Linear) — works whether or not it's
        # wrapped by ActQuantWrapper (output of the wrapper is the same tensor
        # in fake_quant fp16 pass-through configs).
        up_module = layer.mlp.up_proj
        h_up = up_module.register_forward_hook(
            lambda m, inp, out: update('up', out))
        hooks = [h_up]

        if with_gate:
            # The gate-side stats are taken at the OUTPUT of the activation
            # (silu / PWL replacement).  This is what the multiply with up
            # actually consumes — and it's where the equalization scale
            # would be carried by per-channel s_out on the new HW.
            act_module = getattr(layer.mlp, 'act_fn', None)
            assert act_module is not None, (
                "branch_eq with_gate=True expects layer.mlp.act_fn to exist")
            h_g = act_module.register_forward_hook(
                lambda m, inp, out: update('gate', out))
            hooks.append(h_g)

        # Forward pass over calibration samples
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings)[0]

        for h in hooks:
            h.remove()

        branch_stats[i] = per_layer
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return branch_stats


def compute_eq_factors_branches(branch_stats):
    """Apply compute_eq_factors_bounded per branch per layer.

    Returns a dict
        layer_idx -> {'u': tensor[C], 'g': tensor[C]}      (with_gate=True)
        layer_idx -> {'u': tensor[C]}                      (with_gate=False)
    """
    factors = {}
    for i, branches in branch_stats.items():
        entry = {}
        for branch_name, stats in branches.items():
            f = compute_eq_factors_bounded(stats['min'], stats['max']).cpu()
            key = 'u' if branch_name == 'up' else 'g'
            entry[key] = f
            logger.info(f"  layer {i} {branch_name}: factor range "
                        f"[{f.min():.4f}, {f.max():.4f}], mean={f.mean():.4f}")
        factors[i] = entry
    return factors


def apply_branch_eq_weights(model, factors):
    """Apply branch-eq factors to up_proj and down_proj weights only.

    For each layer i with factors[i] = {'u': ..., 'g': ... (optional)}:
      - W_up[c, :]   /= u[c]               (rows of up_proj.weight)
      - W_down[:, c] *= g[c] * u[c]        (cols of down_proj.weight,
                                              g defaults to 1 if absent)

    Does NOT touch the PWL activation — call apply_branch_eq_pwl for that.
    Used in the calibrator pre-pass (before GPTQ runs and before PWL is
    replaced, so eq_factors_g can't be set yet).
    """
    model_type = model_utils.model_type_extractor(model)
    layers = model_utils.get_transformer_layers(model, model_type=model_type)

    for i, layer_module in enumerate(layers):
        if i not in factors:
            continue
        f = factors[i]
        u = f['u']
        g = f.get('g', None)

        # up_proj rows
        up_mod = layer_module.mlp.up_proj
        if isinstance(up_mod, quant_utils.ActQuantWrapper):
            up_mod = up_mod.module
        dev = up_mod.weight.device
        dtype = up_mod.weight.dtype
        u_dev = u.to(dev).float()
        # W_up shape: [intermediate, hidden]; row c is the c-th output channel.
        up_mod.weight.data = (up_mod.weight.data.float()
                              / u_dev.unsqueeze(1)).to(dtype)

        # Combined factor for down_proj column rescale
        if g is not None:
            g_dev = g.to(dev).float()
            combined = g_dev * u_dev
        else:
            combined = u_dev

        # down_proj cols
        down_mod = layer_module.mlp.down_proj
        if isinstance(down_mod, quant_utils.ActQuantWrapper):
            down_mod = down_mod.module
        d_dev = down_mod.weight.device
        d_dtype = down_mod.weight.dtype
        down_mod.weight.data = (down_mod.weight.data.float()
                                * combined.to(d_dev).float().unsqueeze(0)
                                ).to(d_dtype)


def apply_branch_eq_gate(model, factors):
    """Apply per-channel gate-side equalization factor `g[c]` to each layer.

    Two paths depending on whether the inner activation is a PWLActivation:
      * PWL: bake g[c] into per-channel m_q[c]/n_q[c] via `set_eq_factors_g`.
        This is the HW-accurate path that simulates Pluto/Helium's per-channel
        n_q feature.
      * Non-PWL (plain nn.SiLU/nn.GELU/etc.): wrap with `EqActivation` so the
        forward applies `y / g[c]` in fp.  This is the fake-quant simulation
        path — the math is exact (modulo `silu'(x) ≈ 1`), no quantization
        noise is introduced because the silu output isn't quantized in the
        no-PWL config.

    Idempotency: if act_fn is already an `EqActivation`, take the inner
    activation out before re-wrapping, so factors don't compound.

    Called at runtime in main_for_test.py *after* the optional PWL
    replacement.  Only relevant for `--ugd_eq` (factors[i] contains 'g').
    """
    import pwl_utils  # local: avoid circular import at module load
    model_type = model_utils.model_type_extractor(model)
    layers = model_utils.get_transformer_layers(model, model_type=model_type)

    for i, layer_module in enumerate(layers):
        if i not in factors:
            continue
        g = factors[i].get('g', None)
        if g is None:
            continue
        act_mod = getattr(layer_module.mlp, 'act_fn', None)
        if act_mod is None:
            raise RuntimeError(f"layer {i}: mlp has no act_fn")

        if isinstance(act_mod, pwl_utils.PWLActivation):
            # HW-accurate PWL path: per-channel m_q/n_q
            dev = act_mod.thresholds.device
            act_mod.set_eq_factors_g(g.to(dev).float())
            logger.info(f"  Set PWL eq_factors_g for layer {i}")
        else:
            # Non-PWL path: wrap with EqActivation
            inner = act_mod
            if isinstance(inner, EqActivation):
                # Re-application: replace, don't nest
                inner = inner.act_fn
            layer_module.mlp.act_fn = EqActivation(inner, g.float())
            logger.info(f"  Wrapped {type(inner).__name__} with EqActivation "
                        f"for layer {i}")


# Backwards-compatible alias — old call site name.
apply_branch_eq_pwl = apply_branch_eq_gate


def apply_branch_eq(model, factors):
    """Convenience: full apply (weights + gate-side).  Used at runtime in
    main_for_test where the model is fully assembled (PWL replacement, if any,
    has already happened)."""
    apply_branch_eq_weights(model, factors)
    apply_branch_eq_gate(model, factors)
