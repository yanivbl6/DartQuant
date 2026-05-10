"""AdaQuant: Gradient-descent based layer-wise weight quantization.

Minimizes MSE = |XW - Xq Wq|^2 per layer using soft quantization with
temperature annealing.  Drop-in alternative to GPTQ (gptq_utils.py).
"""

import math
import re
import time
import logging

import torch
import torch.nn as nn
import tqdm

import utils
import quant_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------------
# AdaQuantParams – serializable parameter holder
# ---------------------------------------------------------------------------

class AdaQuantParams:
    """All hyperparameters for one AdaQuant run.

    Serialisable to / from a compact inline string so it can be passed as a
    single CLI argument (``--adaquant "lr.0.001_ep.20_optWSX_adam_cos"``).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 20,
        opt_W: bool = True,
        opt_w_scale: bool = True,
        opt_x_scale: bool = True,
        optimizer: str = 'adam',
        cosine_lr: bool = False,
        batch_size: int = 256,
        weight_decay: float = 0.0,
        t_start: float = 1.0,
        t_end: float = 0.01,
        per_block: bool = False,
        nsamples: int = None,
        tps: int = 16,
    ):
        self.lr = lr
        self.epochs = epochs
        self.opt_W = opt_W
        self.opt_w_scale = opt_w_scale
        self.opt_x_scale = opt_x_scale
        self.optimizer = optimizer        # 'adam' | 'sgd'
        self.cosine_lr = cosine_lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.t_start = t_start
        self.t_end = t_end
        self.per_block = per_block
        self.nsamples = nsamples          # None → use global --nsamples
        self.tps = tps                    # tokens per sample per epoch

    # -- serialisation -------------------------------------------------------

    def to_string(self) -> str:
        """Compact inline representation (underscore-separated tokens)."""
        parts = []
        parts.append(f'lr.{self.lr}')
        parts.append(f'ep.{self.epochs}')
        flags = ''
        if self.opt_W:
            flags += 'W'
        if self.opt_w_scale:
            flags += 'S'
        if self.opt_x_scale:
            flags += 'X'
        parts.append(f'opt{flags}')
        parts.append(self.optimizer)
        if self.cosine_lr:
            parts.append('cos')
        parts.append(f'bs.{self.batch_size}')
        if self.weight_decay != 0.0:
            parts.append(f'wd.{self.weight_decay}')
        parts.append(f't.{self.t_start}-{self.t_end}')
        if self.per_block:
            parts.append('blk')
        if self.nsamples is not None:
            parts.append(f'ns.{self.nsamples}')
        if self.tps != 16:
            parts.append(f'tps.{self.tps}')
        return '_'.join(parts)

    @classmethod
    def from_string(cls, s: str) -> 'AdaQuantParams':
        """Parse an inline params string back into an ``AdaQuantParams``."""
        if s is None or s == 'default':
            return cls()

        kwargs = {}
        tokens = s.split('_')
        i = 0
        while i < len(tokens):
            tok = tokens[i]

            if tok.startswith('lr.'):
                kwargs['lr'] = float(tok[3:])
            elif tok.startswith('ep.'):
                kwargs['epochs'] = int(tok[3:])
            elif tok.startswith('opt'):
                flags = tok[3:]
                kwargs['opt_W'] = 'W' in flags
                kwargs['opt_w_scale'] = 'S' in flags
                kwargs['opt_x_scale'] = 'X' in flags
            elif tok in ('adam', 'sgd'):
                kwargs['optimizer'] = tok
            elif tok == 'cos':
                kwargs['cosine_lr'] = True
            elif tok.startswith('bs.'):
                kwargs['batch_size'] = int(tok[3:])
            elif tok.startswith('wd.'):
                kwargs['weight_decay'] = float(tok[3:])
            elif tok.startswith('t.'):
                # t.{start}-{end}  – but start/end may contain dots (floats)
                # e.g. t.1.0-0.01
                rest = tok[2:]
                # Find the dash that separates start from end.
                # The dash is the one *not* at position 0 and preceded by a digit.
                m = re.match(r'^([0-9.]+)-([0-9.]+)$', rest)
                if m:
                    kwargs['t_start'] = float(m.group(1))
                    kwargs['t_end'] = float(m.group(2))
            elif tok == 'blk':
                kwargs['per_block'] = True
            elif tok.startswith('ns.'):
                kwargs['nsamples'] = int(tok[3:])
            elif tok.startswith('tps.'):
                kwargs['tps'] = int(tok[4:])
            else:
                logging.warning("AdaQuantParams.from_string: unknown token '%s'", tok)
            i += 1
        return cls(**kwargs)

    def __repr__(self):
        return f'AdaQuantParams({self.to_string()})'

    def __eq__(self, other):
        if not isinstance(other, AdaQuantParams):
            return False
        return self.to_string() == other.to_string()


# ---------------------------------------------------------------------------
# Soft quantization with temperature annealing
# ---------------------------------------------------------------------------

def soft_quantize(x, scale, zero, maxq, temperature, sym=True):
    """Differentiable soft-rounding that anneals to hard quantization.

    As *temperature* → 0 the sigmoid becomes a step function and this
    converges to the standard round-clip-dequant path.
    """
    if sym:
        minq = -(maxq + 1)
        x_scaled = x / scale
        x_clipped = torch.clamp(x_scaled, minq, maxq)
    else:
        x_scaled = x / scale + zero
        x_clipped = torch.clamp(x_scaled, 0, maxq)

    x_floor = x_clipped.floor()
    frac = x_clipped - x_floor
    soft_round = torch.sigmoid((frac - 0.5) / temperature)
    x_q = x_floor + soft_round
    if sym:
        x_q = torch.clamp(x_q, minq, maxq)
        return scale * x_q
    else:
        x_q = torch.clamp(x_q, 0, maxq)
        return scale * (x_q - zero)


# ---------------------------------------------------------------------------
# AdaQuant – per-layer optimiser
# ---------------------------------------------------------------------------

class AdaQuant:
    """Gradient-descent weight quantisation for a single ``nn.Linear``."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.weight.shape[0]     # out_features
        self.columns = layer.weight.shape[1]  # in_features
        self.inputs = []   # collected on CPU
        self.nsamples = 0

    def add_batch(self, inp, out):
        """Collect an input batch (stored on CPU to save GPU memory).

        Keeps the sequence dimension so we can subsample tokens per epoch.
        Stored as list of [seqlen, hidden] tensors.
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # inp: [batch, seqlen, hidden] – store each sample separately
        for s in range(inp.shape[0]):
            self.inputs.append(inp[s].to('cpu'))  # [seqlen, hidden]
        self.nsamples += inp.shape[0]

    def optimize(self, params: AdaQuantParams, w_quantizer, a_quantizer=None,
                 groupsize=-1):
        """Run AdaQuant gradient descent.

        Returns ``(loss, x_scale_dict_or_None)``.
        *x_scale_dict_or_None* is a dict ``{'scale': tensor, 'zero': tensor}``
        of the optimised activation scale when ``params.opt_x_scale`` is True,
        else ``None``.
        """
        dev = self.dev
        W_orig = self.layer.weight.data.clone().float().to(dev)

        sym_w = w_quantizer.sym
        maxq_w = w_quantizer.maxq.item() if isinstance(w_quantizer.maxq, torch.Tensor) else w_quantizer.maxq

        # --- Inputs stored as list of [seqlen, hidden] tensors ---
        # Stack into [nsamples, seqlen, hidden] on CPU
        all_inp_3d = torch.stack(self.inputs, dim=0)  # [nsamples, seqlen, K]
        n_samples = all_inp_3d.shape[0]
        seqlen = all_inp_3d.shape[1]
        tps = min(params.tps, seqlen)  # tokens per sample per epoch

        # --- Initialise learnable parameters ---
        learnable = []

        # Weight
        W_opt = W_orig.clone()
        if params.opt_W:
            W_opt = nn.Parameter(W_opt)
            learnable.append(W_opt)

        # Weight scale – initialise from quantizer
        if not w_quantizer.ready():
            if groupsize != -1:
                w_quantizer.find_params(W_orig[:, :groupsize])
            else:
                w_quantizer.find_params(W_orig)

        # For grouped quantization, compute initial per-group scales
        if groupsize != -1 and groupsize < self.columns:
            n_groups = (self.columns + groupsize - 1) // groupsize
            w_scales = []
            w_zeros = []
            with torch.no_grad():
                for g in range(n_groups):
                    g_start = g * groupsize
                    g_end = min(g_start + groupsize, self.columns)
                    wq_copy = quant_utils.WeightQuantizer()
                    wq_copy.configure(w_quantizer.bits, perchannel=True,
                                      sym=sym_w, mse=w_quantizer.mse,
                                      gscaler=w_quantizer.gscaler)
                    wq_copy.find_params(W_orig[:, g_start:g_end])
                    w_scales.append(wq_copy.scale.clone().to(dev))
                    w_zeros.append(wq_copy.zero.clone().to(dev))
            # Stack: each is [out_features, 1] -> [out_features, n_groups]
            w_scale_param = torch.cat(w_scales, dim=1).float()
            w_zero_param = torch.cat(w_zeros, dim=1).float()
        else:
            w_scale_param = w_quantizer.scale.clone().float().to(dev)
            w_zero_param = w_quantizer.zero.clone().float().to(dev)

        if params.opt_w_scale:
            w_scale_param = nn.Parameter(w_scale_param)
            learnable.append(w_scale_param)

        # Activation scale
        x_scale_param = None
        x_zero_param = None
        x_maxq = None
        x_sym = True
        if params.opt_x_scale and a_quantizer is not None and a_quantizer.bits < 16:
            x_sym = getattr(a_quantizer, 'sym', True)
            x_maxq = a_quantizer.maxq.item() if isinstance(a_quantizer.maxq, torch.Tensor) else a_quantizer.maxq
            # Initialise from current a_quantizer or compute from data
            if a_quantizer.static and a_quantizer.scale is not None and torch.any(a_quantizer.scale != 0):
                x_scale_init = a_quantizer.scale.clone().float().to(dev)
                x_zero_init = a_quantizer.zero.clone().float().to(dev) if a_quantizer.zero is not None else torch.zeros_like(x_scale_init)
            else:
                # Compute from collected inputs (flatten 3D → 2D for stats)
                with torch.no_grad():
                    x_flat = all_inp_3d.reshape(-1, all_inp_3d.shape[-1]).float().to(dev)
                    if x_sym:
                        x_absmax = x_flat.abs().max(dim=0)[0].clamp(min=1e-5)
                        x_scale_init = (x_absmax / x_maxq).unsqueeze(0)
                        x_zero_init = torch.zeros_like(x_scale_init)
                    else:
                        x_min = x_flat.min(dim=0)[0]
                        x_max = x_flat.max(dim=0)[0]
                        x_scale_init = ((x_max - x_min).clamp(min=1e-5) / x_maxq).unsqueeze(0)
                        x_zero_init = torch.round(-x_min / x_scale_init.squeeze(0)).unsqueeze(0)
                    del x_flat
            x_scale_param = nn.Parameter(x_scale_init)
            x_zero_param = x_zero_init  # zero not optimised (keep fixed)
            learnable.append(x_scale_param)

        if not learnable:
            logging.warning("AdaQuant: nothing to optimise – returning original weights")
            return 0.0, None

        # --- Optimizer ---
        if params.optimizer == 'sgd':
            opt = torch.optim.SGD(learnable, lr=params.lr, weight_decay=params.weight_decay)
        else:
            opt = torch.optim.Adam(learnable, lr=params.lr, weight_decay=params.weight_decay)

        scheduler = None
        if params.cosine_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params.epochs)

        # --- Training loop ---
        bs = params.batch_size
        final_loss = 0.0
        n_groups_w = 1
        if groupsize != -1 and groupsize < self.columns:
            n_groups_w = (self.columns + groupsize - 1) // groupsize

        # Each epoch: pick `tps` random token positions per sample → N_epoch rows
        N_epoch = n_samples * tps
        total_batches = (N_epoch + bs - 1) // bs
        epoch_pbar = tqdm.tqdm(range(params.epochs), desc="    epochs", leave=False)
        for epoch in epoch_pbar:
            # Linear temperature decay
            if params.epochs > 1:
                temperature = params.t_start + (params.t_end - params.t_start) * epoch / (params.epochs - 1)
            else:
                temperature = params.t_end

            epoch_loss = 0.0
            n_batches = 0

            # Subsample: random token positions per sample, then flatten
            tok_idx = torch.randint(0, seqlen, (n_samples, tps))  # [nsamples, tps]
            sample_idx = torch.arange(n_samples).unsqueeze(1).expand_as(tok_idx)
            epoch_inp = all_inp_3d[sample_idx, tok_idx].reshape(N_epoch, -1)  # [N_epoch, K] on CPU

            # Compute reference outputs for this epoch's subsample
            with torch.no_grad():
                ref_parts = []
                for s in range(0, N_epoch, 512):
                    x_c = epoch_inp[s:s + 512].float().to(dev)
                    ref_parts.append((x_c @ W_orig.t()).cpu())
                Y_ref = torch.cat(ref_parts, dim=0)
                del ref_parts

            # Shuffle rows
            perm = torch.randperm(N_epoch)
            batch_pbar = tqdm.tqdm(range(0, N_epoch, bs), desc="      batches",
                                   leave=False, total=total_batches)
            for b_start in batch_pbar:
                b_end = min(b_start + bs, N_epoch)
                idx = perm[b_start:b_end]

                x_batch = epoch_inp[idx].float().to(dev)
                y_ref = Y_ref[idx].float().to(dev)

                # Soft-quantize weights
                if n_groups_w > 1:
                    # Vectorised per-group: reshape to [out, n_groups, groupsize]
                    W_view = W_opt[:, :n_groups_w * groupsize].reshape(self.rows, n_groups_w, groupsize)
                    sc = w_scale_param if w_scale_param.dim() == 2 else w_scale_param.expand(self.rows, n_groups_w)
                    zr = w_zero_param if w_zero_param.dim() == 2 else w_zero_param.expand(self.rows, n_groups_w)
                    # [out, n_groups, 1] for broadcasting over groupsize
                    W_q = soft_quantize(W_view, sc.unsqueeze(2), zr.unsqueeze(2),
                                        maxq_w, temperature, sym=sym_w)
                    W_q = W_q.reshape(self.rows, -1)
                    # Handle remainder columns if any
                    if n_groups_w * groupsize < self.columns:
                        rem = W_opt[:, n_groups_w * groupsize:]
                        g_s = w_scale_param[:, -1:] if w_scale_param.dim() == 2 else w_scale_param
                        g_z = w_zero_param[:, -1:] if w_zero_param.dim() == 2 else w_zero_param
                        W_q = torch.cat([W_q, soft_quantize(rem, g_s, g_z,
                                         maxq_w, temperature, sym=sym_w)], dim=1)
                else:
                    W_q = soft_quantize(W_opt, w_scale_param, w_zero_param,
                                        maxq_w, temperature, sym=sym_w)

                # Soft-quantize activations (optional)
                if x_scale_param is not None:
                    x_batch = soft_quantize(x_batch, x_scale_param, x_zero_param,
                                            x_maxq, temperature, sym=x_sym)

                y_q = x_batch @ W_q.t()
                loss = nn.functional.mse_loss(y_q, y_ref)

                opt.zero_grad()
                loss.backward()
                opt.step()

                # Clamp scale to stay positive
                with torch.no_grad():
                    if params.opt_w_scale:
                        w_scale_param.data.clamp_(min=1e-8)
                    if x_scale_param is not None:
                        x_scale_param.data.clamp_(min=1e-8)

                epoch_loss += loss.item()
                n_batches += 1
                batch_pbar.set_postfix(loss=f"{loss.item():.4g}")

            batch_pbar.close()
            del epoch_inp, Y_ref
            if scheduler is not None:
                scheduler.step()

            final_loss = epoch_loss / max(n_batches, 1)
            epoch_pbar.set_postfix(loss=f"{final_loss:.4g}", T=f"{temperature:.3f}")

        # --- Final hard quantize and store ---
        with torch.no_grad():
            W_final = W_opt.data if isinstance(W_opt, nn.Parameter) else W_opt

            # Store per-group scale/zero for int_gemm compatibility
            _collect_params = (groupsize != -1 and groupsize < self.columns)
            _gptq_scales = []
            _gptq_zeros = []

            if n_groups_w > 1:
                W_hard_parts = []
                for g in range(n_groups_w):
                    g_start = g * groupsize
                    g_end = min(g_start + groupsize, self.columns)
                    g_scale = w_scale_param[:, g:g + 1] if w_scale_param.dim() == 2 else w_scale_param
                    g_zero = w_zero_param[:, g:g + 1] if w_zero_param.dim() == 2 else w_zero_param
                    g_s = g_scale.data if isinstance(g_scale, nn.Parameter) else g_scale
                    g_z = g_zero.data if isinstance(g_zero, nn.Parameter) else g_zero
                    if sym_w:
                        q = quant_utils.sym_quant_dequant(
                            W_final[:, g_start:g_end], g_s, maxq_w)
                    else:
                        q = quant_utils.asym_quant_dequant(
                            W_final[:, g_start:g_end], g_s, g_z, maxq_w)
                    W_hard_parts.append(q)
                    if _collect_params:
                        _gptq_scales.append(g_s.clone())
                        _gptq_zeros.append(g_z.clone())
                W_hard = torch.cat(W_hard_parts, dim=1)
            else:
                s = w_scale_param.data if isinstance(w_scale_param, nn.Parameter) else w_scale_param
                z = w_zero_param.data if isinstance(w_zero_param, nn.Parameter) else w_zero_param
                if sym_w:
                    W_hard = quant_utils.sym_quant_dequant(W_final, s, maxq_w)
                else:
                    W_hard = quant_utils.asym_quant_dequant(W_final, s, z, maxq_w)
                if not sym_w:
                    self.layer._gptq_w_scale = s.clone()
                    self.layer._gptq_w_zero = z.clone()

            self.layer.weight.data = W_hard.to(self.layer.weight.dtype)

            if _collect_params and _gptq_scales:
                self.layer._gptq_w_scale = torch.cat(_gptq_scales, dim=1)
                self.layer._gptq_w_zero = torch.cat(_gptq_zeros, dim=1)

        # --- Build optimised x_scale output ---
        x_scale_out = None
        if x_scale_param is not None:
            x_scale_out = {
                'scale': x_scale_param.data.cpu(),
                'zero': x_zero_param.cpu() if x_zero_param is not None else torch.zeros(1),
            }

        return final_loss, x_scale_out

    def free(self):
        self.inputs = []
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


# ---------------------------------------------------------------------------
# adaquant_fwrd – layer-by-layer orchestration (mirrors gptq_fwrd)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _capture_inputs(model, dataloader, dev, nsamples):
    """Run first layer to capture all calibration inputs (same as gptq_fwrd)."""
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
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

    model.config.use_cache = use_cache
    return inps, cache


def adaquant_fwrd(model, dataloader, dev, args, adaquant_params=None):
    """Layer-by-layer AdaQuant quantisation (mirrors ``gptq_fwrd``).

    Returns ``(quantizers, x_scales)`` where *x_scales* is a dict mapping
    quantizer names to ``{'scale': tensor, 'zero': tensor}`` (empty if
    ``adaquant_params.opt_x_scale`` is False).
    """
    if adaquant_params is None:
        adaquant_params = AdaQuantParams()

    nsamples = len(dataloader)
    p = adaquant_params
    print("=" * 60, flush=True)
    print("AdaQuant Configuration:", flush=True)
    _tps = min(p.tps, model.seqlen)
    _N_epoch = nsamples * _tps
    print(f"  samples     : {nsamples}", flush=True)
    print(f"  seqlen      : {model.seqlen}", flush=True)
    print(f"  tps         : {_tps} (tokens per sample per epoch)", flush=True)
    print(f"  rows/epoch  : {_N_epoch} (samples × tps)", flush=True)
    print(f"  epochs      : {p.epochs}", flush=True)
    print(f"  batch_size  : {p.batch_size} (rows per gradient step)", flush=True)
    print(f"  batches/ep  : {(_N_epoch + p.batch_size - 1) // p.batch_size}", flush=True)
    print(f"  lr          : {p.lr}", flush=True)
    print(f"  optimizer   : {p.optimizer}", flush=True)
    print(f"  cosine_lr   : {p.cosine_lr}", flush=True)
    print(f"  weight_decay: {p.weight_decay}", flush=True)
    print(f"  temperature : {p.t_start} → {p.t_end}", flush=True)
    print(f"  optimize    : W={p.opt_W}, w_scale={p.opt_w_scale}, x_scale={p.opt_x_scale}", flush=True)
    print(f"  per_block   : {p.per_block}", flush=True)
    print("=" * 60, flush=True)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    inps, cache = _capture_inputs(model, dataloader, dev, nsamples)
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache.get('position_embeddings', None)

    quantizers = {}
    x_scales = {}  # {layer_name.quantizer: {scale, zero}}

    sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module'],
    ]

    pbar = tqdm.tqdm(range(len(layers)), desc="(AdaQuant) Layers")
    for i in pbar:
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        layer_losses = []

        groups_to_process = sequential
        if adaquant_params.per_block:
            # Flatten all linears into one group
            all_names = [n for grp in sequential for n in grp]
            groups_to_process = [all_names]

        for names in groups_to_process:
            # Resolve wrapped / bare names (same logic as gptq_fwrd)
            subset = {}
            for n in names:
                if n in full:
                    subset[n] = full[n]
                elif n.endswith('.module') and n[:-len('.module')] in full:
                    subset[n[:-len('.module')]] = full[n[:-len('.module')]]

            aq = {}
            w_bits_map = getattr(args, 'w_bits_map', None)
            for name in subset:
                layer_weight_bits = args.w_bits
                layer_weight_sym = not args.w_asym
                if 'lm_head' in name:
                    continue
                if w_bits_map:
                    bare_name = name.replace('.module', '')
                    full_name = f'model.layers.{i}.{bare_name}'
                    layer_weight_bits = w_bits_map.get(full_name, layer_weight_bits)
                if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in name:
                    layer_weight_bits = args.w_bits_down_proj

                aq[name] = AdaQuant(subset[name])
                aq[name].quantizer = quant_utils.WeightQuantizer()
                aq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym,
                    mse=args.w_clip,
                    gscaler=getattr(args, 'gscaler_parsed', None),
                )

            # --- Collect inputs via hooks ---
            def add_batch(name):
                def tmp(_, inp, out):
                    aq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            with torch.no_grad():
                for j in range(nsamples):
                    outs[j] = layer(inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings)[0]

            for h in handles:
                h.remove()

            # --- Optimise each layer in the group ---
            _wgm = getattr(args, 'weight_group_mode', 'all')
            for name in subset:
                short_name = name.replace('self_attn.', '').replace('mlp.', '').replace('.module', '')
                pbar.set_postfix(sub=short_name)

                bare_name = name.replace('.module', '')
                _keep = (_wgm == 'all'
                         or any(f'{tok}_proj' in bare_name
                                for tok in _wgm.split('_')))
                layer_w_groupsize = args.w_groupsize if _keep else -1

                # Get the activation quantizer if this linear is wrapped
                a_quantizer = None
                if adaquant_params.opt_x_scale:
                    qlayers = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
                    wrapper_name = name if name in qlayers else name.replace('.module', '')
                    if wrapper_name in qlayers:
                        a_quantizer = qlayers[wrapper_name].quantizer

                _ns = aq[name].nsamples
                _tps = min(adaquant_params.tps, model.seqlen)
                _N_ep = _ns * _tps
                _bs = adaquant_params.batch_size
                _nbatch = (_N_ep + _bs - 1) // _bs
                print(f"  [{short_name}] {_ns} samples × {_tps} tps = {_N_ep} rows/epoch, bs={_bs}, {_nbatch} batches/epoch × {adaquant_params.epochs} epochs", flush=True)
                loss, x_scale_out = aq[name].optimize(
                    adaquant_params, aq[name].quantizer,
                    a_quantizer=a_quantizer,
                    groupsize=layer_w_groupsize,
                )
                layer_losses.append(loss)
                quantizers['model.layers.%d.%s' % (i, name)] = aq[name].quantizer

                if x_scale_out is not None:
                    key = f'model.layers.{i}.{name}.quantizer'
                    x_scales[key] = x_scale_out

                aq[name].free()

            # --- Enable int_gemm on just-quantised group (same as gptq_fwrd) ---
            if getattr(args, 'int_gemm', False):
                _a_bits = getattr(args, 'a_bits', 16)
                qlayers_ig = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
                for qname, ql in qlayers_ig.items():
                    if qname + '.module' not in names:
                        continue
                    if ql.quantizer.bits >= 16 and _a_bits < 16:
                        ql.quantizer.configure(bits=_a_bits, groupsize=-1,
                                               sym=True, clip_ratio=1.0)
                    if ql.quantizer.bits < 16 and getattr(ql.quantizer, 'groupsize', -1) <= 0:
                        _ig_wb = args.w_bits
                        bare = qname.replace('.module', '')
                        if w_bits_map:
                            _ig_wb = w_bits_map.get(f'model.layers.{i}.{bare}', _ig_wb)
                        if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in qname:
                            _ig_wb = args.w_bits_down_proj
                        _keep = (_wgm == 'all'
                                 or 'down_proj' in bare
                                 or (_wgm == 'down_o' and 'o_proj' in bare))
                        _ig_w_gs = args.w_groupsize if _keep else -1
                        ql.prepare_int_gemm(
                            w_bits=_ig_wb,
                            w_sym=not args.w_asym,
                            w_group_size=_ig_w_gs,
                            acc_bits=args.acc_bits,
                            acc_block_k=args.acc_block_k,
                            use_triton=getattr(args, 'int_gemm_use_triton', True),
                            acc_wrap=getattr(args, 'acc_wrap', False),
                            acc_dtype=getattr(args, 'acc_dtype', 'float'),
                            gscaler_parsed=getattr(args, 'gscaler_parsed', None),
                            lsb_mac_shift=getattr(args, 'lsb_mac_shift', 0),
                        )

        # Safety-net: enable int_gemm on remaining layers
        if getattr(args, 'int_gemm', False):
            _a_bits = getattr(args, 'a_bits', 16)
            _wgm_safety = getattr(args, 'weight_group_mode', 'all')
            qlayers_ig = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
            for qname, ql in qlayers_ig.items():
                if ql.use_int_gemm:
                    continue
                if ql.quantizer.bits >= 16 and _a_bits < 16:
                    ql.quantizer.configure(bits=_a_bits, groupsize=-1,
                                           sym=True, clip_ratio=1.0)
                if ql.quantizer.bits < 16 and getattr(ql.quantizer, 'groupsize', -1) <= 0:
                    _ig_wb = args.w_bits
                    bare = qname.replace('.module', '')
                    if w_bits_map:
                        _ig_wb = w_bits_map.get(f'model.layers.{i}.{bare}', _ig_wb)
                    if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in qname:
                        _ig_wb = args.w_bits_down_proj
                    _keep = (_wgm_safety == 'all'
                             or any(f'{tok}_proj' in bare
                                    for tok in _wgm_safety.split('_')))
                    _ig_w_gs = args.w_groupsize if _keep else -1
                    ql.prepare_int_gemm(
                        w_bits=_ig_wb,
                        w_sym=not args.w_asym,
                        w_group_size=_ig_w_gs,
                        acc_bits=args.acc_bits,
                        acc_block_k=args.acc_block_k,
                        use_triton=getattr(args, 'int_gemm_use_triton', True),
                        acc_wrap=getattr(args, 'acc_wrap', False),
                        acc_dtype=getattr(args, 'acc_dtype', 'float'),
                        gscaler_parsed=getattr(args, 'gscaler_parsed', None),
                        lsb_mac_shift=getattr(args, 'lsb_mac_shift', 0),
                    )

        avg_loss = sum(layer_losses) / len(layer_losses) if layer_losses else 0.0
        pbar.set_postfix(loss=f"{avg_loss:.4g}")

        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        del aq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----AdaQuant Quantization Done-----')
    return quantizers, x_scales
