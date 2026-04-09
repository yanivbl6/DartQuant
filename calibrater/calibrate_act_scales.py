"""
Calibrate static activation scales for all ActQuantWrapper and QKRotationWrapper
quantizers. Runs the full pipeline: rotations -> GPTQ weight quantization ->
layer-by-layer activation statistics collection.

The GPTQ checkpoint is saved so the experiment script can load it directly.

Simplified usage (mirrors experiment script conventions):
    python calibrate_act_scales.py --mode dart -m 1b --sym --r1 --r2
    python calibrate_act_scales.py --mode quarot -m 1b --sym
    python calibrate_act_scales.py --mode baseline -m 3b

Explicit paths (override auto-deduction):
    python calibrate_act_scales.py --mode dart -m 1b --sym \
        --r1_path /path/to/r1 --r2_path /path/to/r2 \
        --save_path /path/to/scales.pt \
        --gptq_checkpoint_path data/gptq_checkpoints/my_ckpt
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import gc
import functools
import math
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import experiment_config as cfg

sys.path.append('../fake_quant')

import model_utils
import data_utils
import rotation_utils
import quant_utils
import gptq_utils
import hadamard_utils
import utils


@torch.no_grad()
def calibrate_act_scales(model, dataloader, args):
    """Run calibration data through the model and collect per-channel activation scales."""

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dev = 'cuda'

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = args.nsamples
    seqlen = model.seqlen

    # --- Capture first-layer inputs using Catcher ---
    layers[0] = layers[0].to(dev)
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0}

    class Catcher(nn.Module):
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
    attention_mask = cache['attention_mask']
    position_ids = cache.get('position_ids', None)
    position_embeddings = cache.get('position_embeddings', None)

    # --- Collect per-channel min/max for each quantizer ---
    act_scales = {}

    for i in tqdm(range(len(layers)), desc="Calibrating layers"):
        layer = layers[i].to(dev)

        # Collectors: {quantizer_name: {"min": tensor, "max": tensor}}
        collectors = {}

        def collect_hook(module, inp, out, name):
            """Forward hook to collect min/max of pre-quantization input."""
            x = module._cal_input if hasattr(module, '_cal_input') and module._cal_input is not None else (inp[0] if isinstance(inp, tuple) else inp)
            flat = x.reshape(-1, x.shape[-1]).float()
            cmin = flat.min(dim=0)[0]
            cmax = flat.max(dim=0)[0]
            if name not in collectors:
                collectors[name] = {'min': cmin, 'max': cmax}
            else:
                collectors[name]['min'] = torch.minimum(collectors[name]['min'], cmin)
                collectors[name]['max'] = torch.maximum(collectors[name]['max'], cmax)

        def collect_output_hook(module, inp, out, name):
            """Forward hook to collect min/max of post-matmul output (for out_quantizer)."""
            x = module._cal_output if hasattr(module, '_cal_output') and module._cal_output is not None else (out if not isinstance(out, tuple) else out[0])
            flat = x.reshape(-1, x.shape[-1]).float()
            cmin = flat.min(dim=0)[0]
            cmax = flat.max(dim=0)[0]
            if name not in collectors:
                collectors[name] = {'min': cmin, 'max': cmax}
            else:
                collectors[name]['min'] = torch.minimum(collectors[name]['min'], cmin)
                collectors[name]['max'] = torch.maximum(collectors[name]['max'], cmax)

        # Register hooks on ActQuantWrapper modules
        hooks = []
        qlayers = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
        # Enable calibration mode so forward() stores intermediates
        # (works with both int_gemm and fake-quant paths)
        for name, qlayer in qlayers.items():
            qlayer._calibrating = True
        for name, qlayer in qlayers.items():
            full_name = f'model.layers.{i}.{name}'
            if qlayer.quantizer.bits < 16 or qlayer.quantizer.realint:
                hooks.append(
                    qlayer.register_forward_hook(
                        functools.partial(collect_hook, name=f'{full_name}.quantizer')))
            if qlayer.out_quantizer.bits < 16 or qlayer.out_quantizer.realint:
                hooks.append(
                    qlayer.register_forward_hook(
                        functools.partial(collect_output_hook, name=f'{full_name}.out_quantizer')))
            if qlayer.pre_quantizer.bits < 16 or qlayer.pre_quantizer.realint:
                def collect_pre_hook(module, inp, name):
                    """Forward pre-hook to collect min/max of wrapper input (before pre_quantizer)."""
                    x = inp[0] if isinstance(inp, tuple) else inp
                    flat = x.reshape(-1, x.shape[-1]).float()
                    cmin = flat.min(dim=0)[0]
                    cmax = flat.max(dim=0)[0]
                    if name not in collectors:
                        collectors[name] = {'min': cmin, 'max': cmax}
                    else:
                        collectors[name]['min'] = torch.minimum(collectors[name]['min'], cmin)
                        collectors[name]['max'] = torch.maximum(collectors[name]['max'], cmax)
                hooks.append(
                    qlayer.register_forward_pre_hook(
                        functools.partial(collect_pre_hook, name=f'{full_name}.pre_quantizer')))

        # Register hooks on PWLActivation quantizers (if present)
        for name, module in layer.named_modules():
            try:
                import pwl_utils
                if isinstance(module, pwl_utils.PWLActivation):
                    full_name = f'model.layers.{i}.{name}'
                    if module.input_quantizer.bits < 16 or module.input_quantizer.realint:
                        hooks.append(
                            module.register_forward_hook(
                                functools.partial(collect_hook, name=f'{full_name}.input_quantizer')))
                    if module.output_quantizer.bits < 16 or module.output_quantizer.realint:
                         hooks.append(
                            module.register_forward_hook(
                                functools.partial(collect_output_hook, name=f'{full_name}.output_quantizer')))
            except ImportError:
                break  # pwl_utils not available, skip

        # Register hook on QKRotationWrapper for K-cache and Q quantization
        rope_fn = model_utils.get_rope_function_name(model)
        wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
        if hasattr(layer.self_attn, wrapper_attr):
            wrapper = getattr(layer.self_attn, wrapper_attr)
            _need_k = wrapper.k_quantizer.bits < 16 or wrapper.k_quantizer.realint
            _need_q = wrapper.q_quantizer.bits < 16 or wrapper.q_quantizer.realint
            if _need_k or _need_q:
                original_forward = wrapper.forward

                def make_qk_hook(layer_idx, orig_fwd, wrap, collect_k, collect_q):
                    def hooked_forward(*a, **kw):
                        q, k = wrap.func(*a, **kw)
                        d = q.dtype
                        if wrap.use_r3:
                            from fast_hadamard_transform import hadamard_transform
                            q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(d)
                            k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(d)

                        # Collect and quantize K
                        if collect_k:
                            (bsz, num_heads, seq_len, head_dim) = k.shape
                            if wrap.k_groupsize == -1:
                                flat_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
                            elif wrap.k_groupsize >= head_dim:
                                flat_k = k.reshape(-1, head_dim)
                            else:
                                flat_k = k.reshape(-1, wrap.k_groupsize)
                            cmin = flat_k.float().min(dim=0)[0]
                            cmax = flat_k.float().max(dim=0)[0]
                            cname = f'layer.{layer_idx}.k_quantizer'
                            if cname not in collectors:
                                collectors[cname] = {'min': cmin, 'max': cmax}
                            else:
                                collectors[cname]['min'] = torch.minimum(collectors[cname]['min'], cmin)
                                collectors[cname]['max'] = torch.maximum(collectors[cname]['max'], cmax)
                            wrap.k_quantizer.find_params(flat_k)
                            if wrap.k_groupsize == -1:
                                k = wrap.k_quantizer(flat_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
                            else:
                                k = wrap.k_quantizer(flat_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
                            wrap.k_quantizer.free()

                        # Collect and quantize Q
                        if collect_q:
                            (bsz_q, num_heads_q, seq_len_q, head_dim_q) = q.shape
                            flat_q = q.transpose(1, 2).reshape(-1, num_heads_q * head_dim_q)
                            cmin_q = flat_q.float().min(dim=0)[0]
                            cmax_q = flat_q.float().max(dim=0)[0]
                            cname_q = f'layer.{layer_idx}.q_quantizer'
                            if cname_q not in collectors:
                                collectors[cname_q] = {'min': cmin_q, 'max': cmax_q}
                            else:
                                collectors[cname_q]['min'] = torch.minimum(collectors[cname_q]['min'], cmin_q)
                                collectors[cname_q]['max'] = torch.maximum(collectors[cname_q]['max'], cmax_q)
                            wrap.q_quantizer.find_params(flat_q)
                            q = wrap.q_quantizer(flat_q).reshape(
                                (bsz_q, seq_len_q, num_heads_q, head_dim_q)).transpose(1, 2).to(d)
                            wrap.q_quantizer.free()

                        return q, k
                    return hooked_forward

                wrapper.forward = make_qk_hook(i, original_forward, wrapper, _need_k, _need_q)

        # Collect residual quantizer statistics via post-forward hook
        _attn_rq = getattr(layer, '_attn_res_quantizer', None)
        _mlp_rq = getattr(layer, '_mlp_res_quantizer', None)
        if _attn_rq is not None or _mlp_rq is not None:
            def make_res_hook(layer_idx, lay):
                def collect_res(module, inp, out):
                    for tag in ('_attn_res_pre', '_mlp_res_pre'):
                        x = getattr(lay, tag, None)
                        if x is None:
                            continue
                        flat = x.reshape(-1, x.shape[-1]).float()
                        cmin = flat.min(dim=0)[0]
                        cmax = flat.max(dim=0)[0]
                        cname = f'layer.{layer_idx}.{tag.replace("_pre", "_quantizer")}'
                        if cname not in collectors:
                            collectors[cname] = {'min': cmin, 'max': cmax}
                        else:
                            collectors[cname]['min'] = torch.minimum(collectors[cname]['min'], cmin)
                            collectors[cname]['max'] = torch.maximum(collectors[cname]['max'], cmax)
                        setattr(lay, tag, None)  # free memory
                return collect_res
            hooks.append(layer.register_forward_hook(make_res_hook(i, layer)))

        # Run all samples through this layer
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings)[0]
            if j == 0 and outs[j].isnan().any():
                print(f"  WARNING: NaN in layer {i} output after sample 0 "
                      f"(nan={outs[j].isnan().sum()}/{outs[j].numel()})", flush=True)
                for sname, ql in qlayers.items():
                    co = ql._cal_output
                    if co is not None and co.isnan().any():
                        print(f"    {sname}: output NaN", flush=True)
                break  # stop after first NaN sample

        # Remove hooks and clean up calibration state
        for h in hooks:
            h.remove()
        for name, qlayer in qlayers.items():
            qlayer._calibrating = False
            qlayer._cal_input = None
            qlayer._cal_output = None

        # Compute scale/zero from collected min/max
        for cname, stats in collectors.items():
            cmin = stats['min']
            cmax = stats['max']

            qtz = _find_quantizer(layer, cname, i, model, rope_fn)
            if qtz is None:
                continue

            bits = qtz.bits
            sym = qtz.sym
            clip_ratio = qtz.clip_ratio
            _, maxq = quant_utils.get_minq_maxq(bits, sym)

            cmin = cmin * clip_ratio
            cmax = cmax * clip_ratio

            if sym:
                abs_max = torch.maximum(torch.abs(cmin), cmax)
                tmp = abs_max == 0
                scale = abs_max / maxq
                scale[tmp] = 1
                zero = torch.zeros_like(scale)
            else:
                tmp = (cmin == 0) & (cmax == 0)
                cmin[tmp] = -1
                cmax[tmp] = +1
                scale = (cmax - cmin) / maxq
                zero = torch.round(-cmin / scale)

            act_scales[cname] = {
                'scale': scale.cpu(),
                'zero': zero.cpu(),
            }

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return act_scales


def _find_quantizer(layer, cname, layer_idx, model, rope_fn):
    """Look up the ActQuantizer object matching a collector name."""
    qlayers = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])

    prefix = f'model.layers.{layer_idx}.'
    if cname.startswith(prefix) and cname.endswith('.quantizer'):
        subname = cname[len(prefix):-len('.quantizer')]
        if subname in qlayers:
            return qlayers[subname].quantizer

    if cname.startswith(prefix) and cname.endswith('.out_quantizer'):
        subname = cname[len(prefix):-len('.out_quantizer')]
        if subname in qlayers:
            return qlayers[subname].out_quantizer

    if cname.startswith(prefix) and cname.endswith('.pre_quantizer'):
        subname = cname[len(prefix):-len('.pre_quantizer')]
        if subname in qlayers:
            return qlayers[subname].pre_quantizer

    if cname == f'layer.{layer_idx}.k_quantizer':
        wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
        if hasattr(layer.self_attn, wrapper_attr):
            return getattr(layer.self_attn, wrapper_attr).k_quantizer

    if cname == f'layer.{layer_idx}.q_quantizer':
        wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
        if hasattr(layer.self_attn, wrapper_attr):
            return getattr(layer.self_attn, wrapper_attr).q_quantizer

    if cname == f'layer.{layer_idx}._attn_res_quantizer':
        rq = getattr(layer, '_attn_res_quantizer', None)
        if rq is not None:
            return rq

    if cname == f'layer.{layer_idx}._mlp_res_quantizer':
        rq = getattr(layer, '_mlp_res_quantizer', None)
        if rq is not None:
            return rq

    # PWLActivation quantizers
    try:
        import pwl_utils
        if cname.startswith(prefix) and cname.endswith('.input_quantizer'):
            subname = cname[len(prefix):-len('.input_quantizer')]
            for name, module in layer.named_modules():
                if name == subname and isinstance(module, pwl_utils.PWLActivation):
                    return module.input_quantizer
        if cname.startswith(prefix) and cname.endswith('.output_quantizer'):
            subname = cname[len(prefix):-len('.output_quantizer')]
            for name, module in layer.named_modules():
                if name == subname and isinstance(module, pwl_utils.PWLActivation):
                    return module.output_quantizer
    except ImportError:
        pass

    return None


MODEL_MAP = cfg.MODEL_MAP
R1_PATHS = cfg.R1_PATHS
R2_PATHS = cfg.R2_PATHS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate static activation scales',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calibrate_act_scales.py --mode dart -m 1b --sym --r1 --r2
  python calibrate_act_scales.py --mode quarot -m 3b --sym
  python calibrate_act_scales.py --mode baseline -m 7b
  python calibrate_act_scales.py --mode dart -m 1b --r1_path /my/r1 --r2_path /my/r2
""")

    # Model
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path, HF name, or shorthand: 1b, 3b, 7b')
    parser.add_argument('--hf_token', type=str, default=None)

    # Mode
    parser.add_argument('--mode', type=str, default='dart',
                        choices=['baseline', 'quarot', 'dart'])

    # R1/R2 rotation paths
    parser.add_argument('--r1', action='store_true',
                        help='Auto-lookup R1 path for this model (dart mode)')
    parser.add_argument('--r2', action='store_true',
                        help='Auto-lookup R2 path for this model (dart mode)')
    parser.add_argument('--r1_path', type=str, default=None,
                        help='Explicit R1 path (overrides --r1)')
    parser.add_argument('--r2_path', type=str, default=None,
                        help='Explicit R2 path (overrides --r2)')

    # Quantization (matching experiment script defaults)
    parser.add_argument('-w', '--w_bits', type=int, default=4)
    parser.add_argument('-a', '--a_bits', type=int, default=8)
    parser.add_argument('-k', '--k_bits', type=int, default=4)
    parser.add_argument('-v', '--v_bits', type=int, default=None,
                        help='V-cache bit-width (default: same as -k)')
    parser.add_argument('-G', '--groupsize', type=int, default=128,
                        help='Group size for W, K, V (default: 128)')
    parser.add_argument('--sym', action='store_true',
                        help='Symmetric K/V quantization (default: asymmetric)')
    parser.add_argument('--w_asym', action='store_true',
                        help='Asymmetric weight quantization (default: symmetric)')

    # Rarely changed tuning knobs
    parser.add_argument('--a_clip_ratio', type=float, default=0.9)
    parser.add_argument('--k_clip_ratio', type=float, default=1.0)
    parser.add_argument('--v_clip_ratio', type=float, default=1.0)
    parser.add_argument('--percdamp', type=float, default=0.1)
    parser.add_argument('--a_groupsize', type=int, default=-1)
    parser.add_argument('--a_residual', action='store_true')
    parser.add_argument('--kv_ex', type=int, default=0,
                        help='When non-zero, disable R3 and quantize K-cache to N bits')
    parser.add_argument('--proj_ex', type=int, default=0,
                        help='When non-zero, disable R4 and quantize down_proj input to N bits. Shorthand for --no_r4 --down_bits X')
    parser.add_argument('--no_r4', action='store_true',
                        help='Disable R4 rotation on down_proj (without changing bits)')
    parser.add_argument('--down_bits', type=int, default=None,
                        help='Override down_proj input activation bits (without disabling R4)')
    parser.add_argument('--eq', action='store_true',
                        help='Enable per-channel equalization on down_proj inputs')
    parser.add_argument('--fp32_had', action='store_true')
    parser.add_argument('--rotate_mode', type=str, default='hadamard',
                        choices=['hadamard', 'random'])

    # Calibration
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'])

    # PWL Activation Approximation
    parser.add_argument('--pwl_act', action='store_true',
                        help='Use PWL activation approximation during calibration')
    parser.add_argument('--pwl_n_segments', type=int, default=9)
    parser.add_argument('--pwl_input_bits', type=int, default=16)
    parser.add_argument('--pwl_output_bits', type=int, default=16)
    parser.add_argument('--pwl_mantissa_bits', type=int, default=10)
    parser.add_argument('--pwl_exp_bits', type=int, default=4)
    parser.add_argument('--pwl_offset_bits', type=int, default=13)
    parser.add_argument('--pwl_no_hw_sim', action='store_true')

    # Integer GEMM / Capped Accumulator
    parser.add_argument('--int_gemm', action='store_true',
                        help='Run calibration with integer GEMM active')
    parser.add_argument('--acc_bits', type=int, default=32,
                        help='Accumulator bit-width (default: 32 = no capping)')
    parser.add_argument('--acc_block_k', type=int, default=32,
                        help='K-block size for accumulator capping (default: 32)')
    parser.add_argument('--acc_wrap', action='store_true',
                        help='Use wrap-around instead of saturation on accumulator overflow')
    parser.add_argument('--acc_dtype', type=str, default='float',
                        help='Tier-2 accumulator dtype (e.g. fp16, int24). Default: float')

    # Softmax Output Quantization
    parser.add_argument('--smq', type=int, default=0,
                        help='Softmax output quantization bits (0=disabled)')

    # GGUF imitation (per-layer bit-width matching)
    parser.add_argument('--imitate_gguf', type=str, default=None,
                        help='Match per-layer weight bit-widths from a GGUF file. '
                             'Pass a quant scheme name (e.g. Q4_K_S, Q4_K_M, Q4_K_L) '
                             'to lookup <ModelName>-<scheme>.gguf, or an explicit .gguf path.')

    # GGUF pre-quantized weights
    parser.add_argument('--gguf', type=str, default=None,
                        help='Use GGUF pre-quantized weights. Pass a quant type '
                             '(e.g. Q4_K_S, Q4_K_M, Q4_K_L) to lookup '
                             '<ModelName>-<scheme>.gguf from quantized_models/, '
                             'or an explicit path to a .gguf file.')
    parser.add_argument('--quant_warnings', action='store_true',
                        help='Warn when quantization params mismatch GGUF tensor specs')

    # Weight sparsity stats
    parser.add_argument('--weights_stats', type=str, default=None,
                        help='Path to output file for weight sparsity stats.')

    # Group scale quantization
    parser.add_argument('--gscaler', type=str, default=None,
                        help='Group scale format: M5S3, M6E4b2, M6S4l2, etc. '
                             '(default: None = FP32 scales)')

    # AdaQuant (alternative to GPTQ)
    parser.add_argument('--adaquant', type=str, nargs='?', const='default', default=None,
                        help='Use AdaQuant instead of GPTQ. No value = defaults. '
                             'Inline params string to customise '
                             '(e.g., "lr.0.001_ep.20_optWSX_adam_cos")')

    # GPU waiting
    parser.add_argument('--wait', action='store_true',
                        help='Wait for a clear GPU (polls nvidia-smi, overrides CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--max_used_mb', type=int, default=200,
                        help='Max used memory (MiB) for a GPU to be "clear" (default: 200)')

    # Output paths (auto-deduced if not specified)
    parser.add_argument('--save_path', type=str, default=None,
                        help='Where to save scales .pt (auto-deduced if omitted)')
    parser.add_argument('--gptq_checkpoint_path', type=str, default=None,
                        help='GPTQ checkpoint dir (auto-deduced if omitted)')
    parser.add_argument('--gptq', action='store_true',
                        help='Delete cached GPTQ checkpoint and re-quantize')

    # Simulation version tag (no-op here, used for experiment tagging)
    parser.add_argument('--sim_version', type=int, default=0,
                        help='Simulation version tag for A/B comparisons (0=omitted from tag)')

    # FP32 model precision
    parser.add_argument('--fp32', action='store_true',
                        help='Run model in float32 instead of float16 (isolate precision effects)')

    # Real integer quantization (bypass the 16-bit passthrough)
    parser.add_argument('--realint', action='store_true',
                        help='Force real integer quantize/dequantize even at 16 bits')

    # Hardware-aligned activation scales
    parser.add_argument('--hw_align', action='store_true', default=False,
                        help='Tag calibration for hardware-aligned per-group scales')

    # Output quantization
    parser.add_argument('--quant_out', type=str, default='none',
                        choices=['none', 'up', 'mlp', 'spec', 'speco', 'all', 'r4', 'res', 'mm', 'ex'],
                        help='Output quantization: none, up, mlp, spec, speco, all, r4, res (residuals), mm (Q in attn), ex (all+res+mm)')

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Wait for a clear GPU before any CUDA initialization ---
    if args.wait:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
        from gpu_wait import wait_for_gpu
        gpu_id = wait_for_gpu(max_used_mb=args.max_used_mb)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Selected GPU {gpu_id}")

    cfg.resolve_v_bits(args)
    import transformers
    transformers.set_seed(args.seed)

    # --- Resolve model shorthand ---
    if args.model in MODEL_MAP:
        args.model = MODEL_MAP[args.model]
    model_name = os.path.basename(args.model.rstrip('/'))

    # --- Resolve --r1/--r2 auto-lookup ---
    if args.r1 and args.r1_path is None:
        if model_name not in R1_PATHS:
            print(f"Error: no default R1 path for {model_name}. Use --r1_path explicitly.")
            sys.exit(1)
        args.r1_path = R1_PATHS[model_name]
    if args.r2 and args.r2_path is None:
        if model_name not in R2_PATHS:
            print(f"Error: no default R2 path for {model_name}. Use --r2_path explicitly.")
            sys.exit(1)
        args.r2_path = R2_PATHS[model_name]

    # --- Derive symmetry flags (matching experiment script) ---
    # Activations are always asymmetric; --sym controls K/V/W
    args.a_asym = True
    args.k_asym = not args.sym
    args.v_asym = not args.sym
    args.w_asym = args.w_asym and not args.sym

    # --- Derive groupsizes from --groupsize ---
    args.w_groupsize = args.groupsize
    args.k_groupsize = args.groupsize
    args.v_groupsize = args.groupsize

    # --- Always-on flags for quarot/dart ---
    args.o_per_head = (args.mode in ('quarot', 'dart'))
    args.w_clip = True

    # --- Build quant tag (centralized in experiment_config) ---
    quant_tag = cfg.build_quant_tag(args, for_cal_cache=True)
    gptq_cache_tag = cfg.build_quant_tag(args, for_gptq_cache=True, for_cal_cache=True)

    # Resolve GGUF path (needed for weight loading, separate from tag)
    gguf_path = None
    if args.gguf is not None:
        gguf_path = cfg.resolve_gguf_path(args.gguf, args.model)

    save_prefix = args.mode

    # --- Auto-deduce output paths ---
    if args.save_path is None:
        args.save_path = f"../data/act_scales/{model_name}/{save_prefix}_{quant_tag}.pt"
    if args.gptq_checkpoint_path is None:
        args.gptq_checkpoint_path = f"../data/gptq_checkpoints/{save_prefix}_{model_name}_{gptq_cache_tag}"

    # --- Print resolved config ---
    print(f"Mode:       {args.mode}")
    print(f"Model:      {args.model}")
    print(f"Quant tag:  {quant_tag}")
    print(f"R1 path:    {args.r1_path}")
    print(f"R2 path:    {args.r2_path}")
    print(f"GPTQ ckpt:  {args.gptq_checkpoint_path}")
    print(f"Save path:  {args.save_path}")
    if gguf_path:
        print(f"GGUF:       {gguf_path}")
    print()

    model = model_utils.get_model(args.model, args.hf_token)
    if getattr(args, 'fp32', False):
        model = model.float()
    model.eval()
    model.model_name = model_name

    # --- Load GGUF pre-quantized weights (before rotations) ---
    if gguf_path:
        import gguf_utils
        gguf_utils.load_gguf_weights(
            model, gguf_path,
            quant_warnings=args.quant_warnings,
            w_bits=args.w_bits,
            w_groupsize=args.w_groupsize,
            w_sym=not args.w_asym,
        )

    # --- Resolve imitate_gguf: build per-layer bit-width map ---
    if getattr(args, 'imitate_gguf', None):
        import gguf_utils
        gguf_imitate_path = cfg.resolve_gguf_path(args.imitate_gguf, args.model)
        args.w_bits_map, _imit_label = gguf_utils.get_gguf_bits_map(gguf_imitate_path)
        print(f"imitate_gguf: loaded {len(args.w_bits_map)} layer bit-widths from {gguf_imitate_path}")

    # --- Expand --proj_ex into --no_r4 + --down_bits ---
    if args.proj_ex != 0:
        args.no_r4 = True
        if getattr(args, 'down_bits', None) is None:
            args.down_bits = args.proj_ex

    # --- Set up rotations to match experiment pipeline ---
    if args.mode in ('quarot', 'dart'):
        rotation_utils.fuse_layer_norms(model)

        class RotArgs:
            pass
        rot_args = RotArgs()
        rot_args.rotate_mode = args.rotate_mode
        rot_args.use_r1 = True
        rot_args.r1_path = args.r1_path
        if args.r1_path and '.pt' not in args.r1_path and '.bin' not in args.r1_path:
            rot_args.r1_path += '/' + args.r1_path.split('/')[-1] + '.pt'
        rot_args.use_r2 = 'offline'
        rot_args.r2_path = args.r2_path
        if args.r2_path and '.pt' not in args.r2_path and '.bin' not in args.r2_path:
            rot_args.r2_path += '/' + args.r2_path.split('/')[-1] + '.pt'
        rot_args.use_r4 = not getattr(args, 'no_r4', False)
        rot_args.use_r3 = (args.kv_ex == 0)
        rot_args.o_per_head = args.o_per_head
        rot_args.smooth = None

        rotation_utils.rotate_model(model, rot_args)

    if args.kv_ex != 0:
        args.k_bits = args.kv_ex

    if args.mode == 'baseline':
        pass

    utils.cleanup_memory(verbos=True)

    # --- Add activation quantization wrappers ---
    quant_utils.add_actquant(model)
    qlayers = quant_utils.find_qlayers(model)

    # Configure online Hadamard for R4 (down_proj) and online R2 (o_proj)
    if args.mode in ('quarot', 'dart'):
        for name in qlayers:
            if 'down_proj' in name and not getattr(args, 'no_r4', False):
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name and rot_args.use_r2 == 'online':
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had

    # --- Configure activation quantizer bit-widths ---
    # Must happen before GPTQ so int_gemm per-group setup sees correct groupsize
    # (e.g., o_per_head sets groupsize > 0 on o_proj, which must be respected).
    down_proj_groupsize = -1
    if args.a_groupsize > 0:
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)

    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    for name in qlayers:
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize
        layer_a_sym = not args.a_asym
        layer_a_clip = args.a_clip_ratio
        residual = args.a_residual

        if 'v_proj' in name and (args.v_bits < 16 or getattr(args, 'realint', False)):
            qlayers[name].out_quantizer.configure(
                bits=args.v_bits, groupsize=args.v_groupsize,
                sym=not args.v_asym, clip_ratio=args.v_clip_ratio)

        if 'lm_head' in name:
            layer_input_bits = 16

        if args.o_per_head and 'o_proj' in name:
            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            layer_groupsize = model_dim // num_heads

        if 'down_proj' in name:
            if getattr(args, 'down_bits', None) is not None:
                layer_input_bits = args.down_bits
            layer_groupsize = down_proj_groupsize

        qlayers[name].quantizer.configure(
            bits=layer_input_bits, groupsize=layer_groupsize,
            sym=layer_a_sym, clip_ratio=layer_a_clip, residual=residual)

        if getattr(args, 'realint', False):
            qlayers[name].quantizer.realint = True
            # Only set realint on out_quantizer if it was already configured (bits < 16)
            if qlayers[name].out_quantizer.maxq != 0:
                qlayers[name].out_quantizer.realint = True

    # --- Equalization pre-pass (before GPTQ) ---
    eq_factors = None
    if getattr(args, 'eq', False):
        import equalization as eq_module
        print("Running equalization pre-pass to collect down_proj stats...")
        eq_dataloader = data_utils.get_loaders(
            args.calib_dataset, nsamples=args.nsamples,
            seed=args.seed, model=args.model,
            seqlen=model.seqlen, eval_mode=False)
        eq_stats = eq_module.collect_eq_stats(
            model, eq_dataloader, args.nsamples, dev='cuda')
        eq_factors = eq_module.compute_eq_factors(eq_stats)
        print(f"Computed equalization factors for {len(eq_factors)} layers")
        eq_module.apply_eq_to_weights(model, eq_factors)
        eq_module.setup_eq_online(model, eq_factors)
        print("Equalization applied to weights and online scaling")

    # --- Weight quantization (GPTQ or AdaQuant) ---
    # Must happen before activation calibration so we measure activations
    # flowing through quantized weights (matching actual inference).
    _has_w_bits_map = bool(getattr(args, 'w_bits_map', None))
    _use_adaquant = getattr(args, 'adaquant', None) is not None
    if args.w_bits < 16 or _has_w_bits_map:
        _w_suffix = 'w0' if _has_w_bits_map else f'w{args.w_bits}'

        if _use_adaquant:
            # --- AdaQuant path ---
            import sys, os as _os
            sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'fake_quant'))
            import adaquant_utils

            _aq_base = args.gptq_checkpoint_path.replace('gptq_checkpoints', 'adaquant_checkpoints')
            _aq_ckpt = _os.path.join(_aq_base, f'{model_name}_{_w_suffix}')

            if args.gptq and os.path.isdir(_aq_ckpt):
                import shutil
                print(f"--gptq: removing cached AdaQuant checkpoint: {_aq_ckpt}")
                shutil.rmtree(_aq_ckpt)

            if os.path.isdir(_aq_ckpt) and any(
                    f.endswith('.pth') for f in os.listdir(_aq_ckpt)):
                print(f"Loading existing AdaQuant checkpoint from: {_aq_ckpt}")
                utils.load_model_in_parts(model, _aq_ckpt)
                if args.w_asym and getattr(args, 'int_gemm', False):
                    from int_acc_gemm import load_gptq_w_params
                    load_gptq_w_params(model, _aq_ckpt)
            else:
                _w_desc = f"w_bits_map({len(args.w_bits_map)} layers)" if _has_w_bits_map else f"w{args.w_bits}"
                print(f"Running AdaQuant ({_w_desc}, groupsize={args.w_groupsize}) ...")
                aq_params = adaquant_utils.AdaQuantParams.from_string(args.adaquant)
                aq_nsamples = aq_params.nsamples if aq_params.nsamples is not None else args.nsamples
                trainloader = data_utils.get_loaders(
                    args.calib_dataset, nsamples=aq_nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False)

                class AqArgs:
                    pass
                aq_args = AqArgs()
                aq_args.nsamples = aq_nsamples
                aq_args.w_bits = args.w_bits
                aq_args.w_asym = args.w_asym
                aq_args.w_groupsize = args.w_groupsize
                aq_args.w_clip = args.w_clip
                aq_args.w_bits_down_proj = None
                aq_args.w_bits_map = getattr(args, 'w_bits_map', None)
                aq_args.int_gemm = getattr(args, 'int_gemm', False)
                aq_args.a_bits = getattr(args, 'a_bits', 16)
                aq_args.acc_bits = getattr(args, 'acc_bits', 32)
                aq_args.acc_block_k = getattr(args, 'acc_block_k', 32)
                aq_args.acc_wrap = getattr(args, 'acc_wrap', False)
                aq_args.int_gemm_use_triton = True
                aq_args.gscaler_parsed = quant_utils.parse_gscaler(
                    getattr(args, 'gscaler', None))

                _quantizers, _x_scales = adaquant_utils.adaquant_fwrd(
                    model, trainloader, 'cuda', aq_args, aq_params)

                if os.path.isdir(_aq_ckpt) and any(
                        f.endswith('.pth') for f in os.listdir(_aq_ckpt)):
                    print(f"AdaQuant checkpoint already exists (written by another run) – skipping save: {_aq_ckpt}")
                else:
                    os.makedirs(_aq_ckpt, exist_ok=True)
                    print(f"Saving AdaQuant checkpoint to: {_aq_ckpt}")
                    utils.save_model_in_parts(model, _aq_ckpt,
                                              prefix=f'{model_name}_part')
                    if _x_scales:
                        torch.save(_x_scales, os.path.join(_aq_ckpt, '_adaquant_x_scales.pt'))
                    if args.w_asym and getattr(args, 'int_gemm', False):
                        from int_acc_gemm import save_gptq_w_params
                        save_gptq_w_params(model, _aq_ckpt)

        else:
            # --- GPTQ path ---
            _gptq_ckpt = os.path.join(args.gptq_checkpoint_path, f'{model_name}_{_w_suffix}')

            if args.gptq and os.path.isdir(_gptq_ckpt):
                import shutil
                print(f"--gptq: removing cached GPTQ checkpoint: {_gptq_ckpt}")
                shutil.rmtree(_gptq_ckpt)

            if os.path.isdir(_gptq_ckpt) and any(
                    f.endswith('.pth') for f in os.listdir(_gptq_ckpt)):
                print(f"Loading existing GPTQ checkpoint from: {_gptq_ckpt}")
                utils.load_model_in_parts(model, _gptq_ckpt)
                if args.w_asym and getattr(args, 'int_gemm', False):
                    from int_acc_gemm import load_gptq_w_params
                    load_gptq_w_params(model, _gptq_ckpt)
            else:
                _w_desc = f"w_bits_map({len(args.w_bits_map)} layers)" if _has_w_bits_map else f"w{args.w_bits}"
                print(f"Running GPTQ ({_w_desc}, groupsize={args.w_groupsize}) ...")
                trainloader = data_utils.get_loaders(
                    args.calib_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False)

                class GptqArgs:
                    pass
                gptq_args = GptqArgs()
                gptq_args.nsamples = args.nsamples
                gptq_args.w_bits = args.w_bits
                gptq_args.w_asym = args.w_asym
                gptq_args.w_groupsize = args.w_groupsize
                gptq_args.w_clip = args.w_clip
                gptq_args.w_bits_down_proj = None
                gptq_args.w_bits_map = getattr(args, 'w_bits_map', None)
                gptq_args.percdamp = args.percdamp
                gptq_args.act_order = False
                gptq_args.w_static_groups = False
                # Forward int_gemm args so GPTQ enables capped accumulator between groups
                gptq_args.int_gemm = getattr(args, 'int_gemm', False)
                gptq_args.a_bits = getattr(args, 'a_bits', 16)
                gptq_args.acc_bits = getattr(args, 'acc_bits', 32)
                gptq_args.acc_block_k = getattr(args, 'acc_block_k', 32)
                gptq_args.acc_wrap = getattr(args, 'acc_wrap', False)
                gptq_args.int_gemm_use_triton = True
                # Forward group-scale quantization config
                gptq_args.gscaler_parsed = quant_utils.parse_gscaler(
                    getattr(args, 'gscaler', None))

                gptq_utils.gptq_fwrd(model, trainloader, 'cuda', gptq_args)

                # Guard against parallel runs with the same GPTQ cache tag:
                # if another process saved while we were running, skip saving.
                if os.path.isdir(_gptq_ckpt) and any(
                        f.endswith('.pth') for f in os.listdir(_gptq_ckpt)):
                    print(f"GPTQ checkpoint already exists (written by another run) – skipping save: {_gptq_ckpt}")
                else:
                    os.makedirs(_gptq_ckpt, exist_ok=True)
                    print(f"Saving GPTQ checkpoint to: {_gptq_ckpt}")
                    utils.save_model_in_parts(model, _gptq_ckpt,
                                              prefix=f'{model_name}_part')
                    if args.w_asym and getattr(args, 'int_gemm', False):
                        from int_acc_gemm import save_gptq_w_params
                        save_gptq_w_params(model, _gptq_ckpt)

        utils.cleanup_memory(verbos=True)

    # --- Weight sparsity stats ---
    if getattr(args, 'weights_stats', None) and args.w_bits < 16:
        import json as _json
        qlayers_ws = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        stats_rows = []
        for name, qlayer in qlayers_ws.items():
            w = qlayer.module.weight.data
            N, K = w.shape
            total = w.numel()
            n_zeros = (w == 0).sum().item()
            pct_zero = n_zeros / total if total > 0 else 0.0
            ops = 2 * N * K
            stats_rows.append({
                'layer': name,
                'shape': [N, K],
                'total': total,
                'zeros': n_zeros,
                'pct_zero': pct_zero,
                'ops': ops,
            })

        total_ops = sum(r['ops'] for r in stats_rows)
        effective_sparsity = (
            sum(r['ops'] * r['pct_zero'] for r in stats_rows) / total_ops
            if total_ops > 0 else 0.0
        )

        ws_path = args.weights_stats
        os.makedirs(os.path.dirname(os.path.abspath(ws_path)), exist_ok=True)
        with open(ws_path, 'w') as f:
            f.write(f"{'Layer':<60} {'Shape':>14} {'Total':>10} {'Zeros':>10} {'%Zero':>8} {'OPs':>14}\n")
            f.write('-' * 120 + '\n')
            for r in stats_rows:
                shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
                f.write(f"{r['layer']:<60} {shape_str:>14} {r['total']:>10} {r['zeros']:>10} {r['pct_zero']:>8.4f} {r['ops']:>14}\n")
            f.write('-' * 120 + '\n')
            f.write(f"Effective sparsity (ops-weighted): {effective_sparsity:.6f}\n")
            f.write(f"Total OPs: {total_ops}\n")
            f.write('\n--- JSON ---\n')
            _json.dump({
                'layers': stats_rows,
                'effective_sparsity': effective_sparsity,
                'total_ops': total_ops,
            }, f, indent=2)
            f.write('\n')

        print(f"Weight stats written to {ws_path} (effective sparsity: {effective_sparsity:.6f})")

    # Add K-cache quantization wrappers
    if args.k_bits < 16 or getattr(args, 'realint', False):
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        # QKRotationWrapper clamps k_groupsize to a valid divisor of head_dim
        k_quant_config = {
            'k_bits': args.k_bits, 'k_groupsize': args.k_groupsize,
            'k_sym': not args.k_asym, 'k_clip_ratio': args.k_clip_ratio,
            'use_r3': (args.mode in ('quarot', 'dart')) and (args.kv_ex == 0),
        }
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn, rope_function_name,
                config=model.config, **k_quant_config)
        if getattr(args, 'realint', False):
            for layer in layers:
                wrapper_attr = f'{rope_function_name}_qk_rotation_wrapper'
                if hasattr(layer.self_attn, wrapper_attr):
                    getattr(layer.self_attn, wrapper_attr).k_quantizer.realint = True

    # --- Replace activations with PWL (before calibration so scales reflect PWL) ---
    if args.pwl_act:
        import pwl_utils
        act_name = getattr(model.config, 'hidden_act', 'silu')
        hw_config = None if args.pwl_no_hw_sim else pwl_utils.HWConfig(
            mantissa_bits=args.pwl_mantissa_bits,
            exp_bits=args.pwl_exp_bits,
            offset_bits=args.pwl_offset_bits)
        replaced = pwl_utils.replace_activation_with_pwl(
            model, act_name=act_name,
            n_segments=args.pwl_n_segments,
            hw_config=hw_config,
            input_bits=args.pwl_input_bits,
            output_bits=args.pwl_output_bits)
        print(f"Replaced {len(replaced)} activations with PWL ({act_name}, {args.pwl_n_segments} segments)")
        if getattr(args, 'realint', False):
            for pwl_mod in pwl_utils.find_pwl_activations(model).values():
                pwl_mod.input_quantizer.realint = True
                pwl_mod.output_quantizer.realint = True

    # --- Configure output quantization (--quant_out) ---
    # Must happen AFTER GPTQ/checkpoint loading, which overwrites buffers via load_state_dict.
    quant_out = getattr(args, 'quant_out', 'none')
    if quant_out != 'none':
        qlayers_qo = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name, qlayer in qlayers_qo.items():
            if quant_utils.should_quant_out(name, quant_out):
                if qlayer.out_quantizer.maxq == 0:
                    qlayer.out_quantizer.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
                    qlayer.out_quantizer.realint = True
            if quant_utils.should_quant_pre(name, quant_out):
                qlayer.pre_quantizer.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
                qlayer.pre_quantizer.realint = True
        quant_utils.link_adjacent_quantizers(model)

    # Setup residual quantizers (--quant_out res/ex)
    if quant_utils.needs_residual_quant(quant_out):
        quant_utils.setup_residual_quantizers(model)

    # Setup Q quantizer in attention (--quant_out mm/ex)
    if quant_utils.needs_mm_quant(quant_out):
        layers = model.model.layers
        rope_fn = model_utils.get_rope_function_name(model)
        for layer in layers:
            wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
            if hasattr(layer.self_attn, wrapper_attr):
                wrapper = getattr(layer.self_attn, wrapper_attr)
                wrapper.q_quantizer.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
                wrapper.q_quantizer.realint = True

    # --- Enable integer GEMM on ActQuantWrappers ---
    if args.int_gemm:
        print(f"Enabling integer GEMM for calibration: acc_bits={args.acc_bits}, acc_block_k={args.acc_block_k}", flush=True)
        ig_qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        print(f"  Found {len(ig_qlayers)} ActQuantWrapper layers", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"  CUDA synced, proceeding with prepare_int_gemm...", flush=True)
        _ig_w_bits_map = getattr(args, 'w_bits_map', None)
        n_ig = 0
        for name, qlayer in ig_qlayers.items():
            if 'lm_head' in name or qlayer.quantizer.bits > 16:
                continue
            if getattr(qlayer.quantizer, 'groupsize', -1) > 0:
                logging.info(f"  skipping int_gemm for {name} (act groupsize={qlayer.quantizer.groupsize})")
                continue
            # Resolve per-layer w_bits (from --imitate_gguf bit-width map)
            layer_w_bits = args.w_bits
            if _ig_w_bits_map:
                layer_w_bits = _ig_w_bits_map.get(name, layer_w_bits)
            if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in name:
                layer_w_bits = args.w_bits_down_proj
            dev = qlayer.module.weight.device
            print(f"  [{n_ig}] {name}: weight on {dev}, shape={list(qlayer.module.weight.shape)}, bits={qlayer.quantizer.bits}, w_bits={layer_w_bits}", flush=True)
            qlayer.prepare_int_gemm(
                w_bits=layer_w_bits, w_sym=not args.w_asym,
                w_group_size=args.w_groupsize,
                acc_bits=args.acc_bits, acc_block_k=args.acc_block_k,
                acc_wrap=args.acc_wrap,
                acc_dtype=getattr(args, 'acc_dtype', 'float'),
                gscaler_parsed=getattr(args, 'gscaler_parsed', None))
            n_ig += 1
            print(f"    done.", flush=True)
        print(f"Integer GEMM prepared for {n_ig} layers", flush=True)

    # --- Enable softmax output quantization ---
    if args.smq > 0:
        import smq_utils
        smq_utils.enable_smq(args.smq)
        print(f"Enabled softmax output quantization: {args.smq} bits")

    # --- Get calibration data ---
    dataloader = data_utils.get_loaders(
        args.calib_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False)

    # --- Run activation scale calibration ---
    act_scales = calibrate_act_scales(model, dataloader, args)

    # --- Include equalization factors in saved output ---
    if eq_factors is not None:
        act_scales['__eq_factors__'] = {k: v.cpu() for k, v in eq_factors.items()}
        print(f"Including equalization factors for {len(eq_factors)} layers")

    # --- Save ---
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(act_scales, args.save_path)
    print(f"\nSaved {len(act_scales)} activation scale entries to {args.save_path}")
    for k in sorted(act_scales.keys()):
        if k.startswith('__'):
            continue
        s = act_scales[k]['scale']
        print(f"  {k}: scale shape={list(s.shape)}, range=[{s.min():.6f}, {s.max():.6f}]")


if __name__ == '__main__':
    main()
