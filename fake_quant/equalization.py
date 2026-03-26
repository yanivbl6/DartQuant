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
from tqdm import tqdm

import model_utils
import quant_utils
import utils

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
