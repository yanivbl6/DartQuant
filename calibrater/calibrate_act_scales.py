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
        --gptq_checkpoint_path /tmp/my_ckpt
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
            """Forward hook to collect min/max of the input tensor."""
            x = inp[0] if isinstance(inp, tuple) else inp
            flat = x.reshape(-1, x.shape[-1]).float()
            cmin = flat.min(dim=0)[0]
            cmax = flat.max(dim=0)[0]
            if name not in collectors:
                collectors[name] = {'min': cmin, 'max': cmax}
            else:
                collectors[name]['min'] = torch.minimum(collectors[name]['min'], cmin)
                collectors[name]['max'] = torch.maximum(collectors[name]['max'], cmax)

        def collect_output_hook(module, inp, out, name):
            """Forward hook to collect min/max of the output tensor (for v_proj out_quantizer)."""
            x = out if not isinstance(out, tuple) else out[0]
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
        for name, qlayer in qlayers.items():
            full_name = f'model.layers.{i}.{name}'
            if qlayer.quantizer.bits < 16:
                hooks.append(
                    qlayer.module.register_forward_hook(
                        functools.partial(collect_hook, name=f'{full_name}.quantizer')))
            if qlayer.out_quantizer.bits < 16:
                hooks.append(
                    qlayer.module.register_forward_hook(
                        functools.partial(collect_output_hook, name=f'{full_name}.out_quantizer')))

        # Register hook on QKRotationWrapper for K-cache
        rope_fn = model_utils.get_rope_function_name(model)
        wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
        if hasattr(layer.self_attn, wrapper_attr):
            wrapper = getattr(layer.self_attn, wrapper_attr)
            if wrapper.k_quantizer.bits < 16:
                original_forward = wrapper.forward

                def make_k_hook(layer_idx, orig_fwd, wrap):
                    def hooked_forward(*a, **kw):
                        q, k = wrap.func(*a, **kw)
                        d = q.dtype
                        if wrap.use_r3:
                            from fast_hadamard_transform import hadamard_transform
                            q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(d)
                            k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(d)
                        (bsz, num_heads, seq_len, head_dim) = k.shape
                        if wrap.k_groupsize == -1:
                            flat_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
                        else:
                            flat_k = k.reshape(-1, head_dim)
                        cmin = flat_k.float().min(dim=0)[0]
                        cmax = flat_k.float().max(dim=0)[0]
                        cname = f'layer.{layer_idx}.k_quantizer'
                        if cname not in collectors:
                            collectors[cname] = {'min': cmin, 'max': cmax}
                        else:
                            collectors[cname]['min'] = torch.minimum(collectors[cname]['min'], cmin)
                            collectors[cname]['max'] = torch.maximum(collectors[cname]['max'], cmax)
                        # Still quantize K for correct layer output propagation
                        wrap.k_quantizer.find_params(flat_k)
                        if wrap.k_groupsize == -1:
                            k = wrap.k_quantizer(flat_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
                        else:
                            k = wrap.k_quantizer(flat_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
                        wrap.k_quantizer.free()
                        return q, k
                    return hooked_forward

                wrapper.forward = make_k_hook(i, original_forward, wrapper)

        # Run all samples through this layer
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings)[0]

        # Remove hooks
        for h in hooks:
            h.remove()

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
            maxq = maxq.float()

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

    if cname == f'layer.{layer_idx}.k_quantizer':
        wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
        if hasattr(layer.self_attn, wrapper_attr):
            return getattr(layer.self_attn, wrapper_attr).k_quantizer

    return None


MODEL_BASE = "/data/users/sashas/LLMC/Models/meta-llama"
MODEL_MAP = {
    '1b': f'{MODEL_BASE}/Llama-3.2-1B-Instruct',
    '3b': f'{MODEL_BASE}/Llama-3.2-3B-Instruct',
    '7b': f'{MODEL_BASE}/Llama-2-7b-hf',
}

# Default R1/R2 paths per model (same as experiment script dart mode)
R1_PATHS = {
    'Llama-2-7b-hf':          '../data/trained_rotation/wikitext2_128samples/r1/sgd.0.0015.0.9.10.64.0.1.1',
    'Llama-3.2-1B-Instruct':  '../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1',
    'Llama-3.2-3B-Instruct':  '../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1',
}
R2_PATHS = {
    'Llama-2-7b-hf':          '../data/trained_rotation/wikitext2_128samples/r2/sgd.0.001.0.9.10.64.2',
    'Llama-3.2-1B-Instruct':  '../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r2/sgd.0.001.0.9.10.64.2',
    'Llama-3.2-3B-Instruct':  '../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r2/sgd.0.001.0.9.10.64.2',
}


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
    parser.add_argument('--v_bits', type=int, default=4)
    parser.add_argument('-G', '--groupsize', type=int, default=128,
                        help='Group size for W, K, V (default: 128)')
    parser.add_argument('--sym', action='store_true',
                        help='Symmetric K/V quantization (default: asymmetric)')

    # Rarely changed tuning knobs
    parser.add_argument('--a_clip_ratio', type=float, default=0.9)
    parser.add_argument('--k_clip_ratio', type=float, default=1.0)
    parser.add_argument('--v_clip_ratio', type=float, default=1.0)
    parser.add_argument('--percdamp', type=float, default=0.1)
    parser.add_argument('--a_groupsize', type=int, default=-1)
    parser.add_argument('--a_residual', action='store_true')
    parser.add_argument('--fp32_had', action='store_true')
    parser.add_argument('--rotate_mode', type=str, default='hadamard',
                        choices=['hadamard', 'random'])

    # Calibration
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'])

    # Output paths (auto-deduced if not specified)
    parser.add_argument('--save_path', type=str, default=None,
                        help='Where to save scales .pt (auto-deduced if omitted)')
    parser.add_argument('--gptq_checkpoint_path', type=str, default=None,
                        help='GPTQ checkpoint dir (auto-deduced if omitted)')

    return parser.parse_args()


def main():
    args = parse_args()
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
    args.w_asym = False  # W is always symmetric

    sym_tag = "wSym_kSym_vSym" if args.sym else "kAsym_vAsym"

    # --- Derive groupsizes from --groupsize ---
    args.w_groupsize = args.groupsize
    args.k_groupsize = args.groupsize
    args.v_groupsize = args.groupsize

    # --- Always-on flags for quarot/dart ---
    args.o_per_head = (args.mode in ('quarot', 'dart'))
    args.w_clip = True

    # --- Build quant tag (matching experiment script convention) ---
    quant_tag = f"w{args.w_bits}a{args.a_bits}k{args.k_bits}v{args.v_bits}_g{args.groupsize}_aAsym_{sym_tag}"
    save_prefix = args.mode

    # --- Auto-deduce output paths ---
    if args.save_path is None:
        args.save_path = f"../data/act_scales/{model_name}/{save_prefix}_{quant_tag}.pt"
    if args.gptq_checkpoint_path is None:
        args.gptq_checkpoint_path = f"/tmp/{save_prefix}_{model_name}_{quant_tag}"

    # --- Print resolved config ---
    print(f"Mode:       {args.mode}")
    print(f"Model:      {args.model}")
    print(f"Quant tag:  {quant_tag}")
    print(f"R1 path:    {args.r1_path}")
    print(f"R2 path:    {args.r2_path}")
    print(f"GPTQ ckpt:  {args.gptq_checkpoint_path}")
    print(f"Save path:  {args.save_path}")
    print()

    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    model.model_name = model_name

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
        rot_args.use_r4 = True
        rot_args.use_r3 = True
        rot_args.o_per_head = args.o_per_head
        rot_args.smooth = None

        rotation_utils.rotate_model(model, rot_args)
    elif args.mode == 'baseline':
        pass

    utils.cleanup_memory(verbos=True)

    # --- Add activation quantization wrappers ---
    quant_utils.add_actquant(model)
    qlayers = quant_utils.find_qlayers(model)

    # Configure online Hadamard for R4 (down_proj) and online R2 (o_proj)
    if args.mode in ('quarot', 'dart'):
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name and args.o_per_head:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had

    # --- GPTQ weight quantization ---
    # Must happen before activation calibration so we measure activations
    # flowing through quantized weights (matching actual inference).
    if args.w_bits < 16:
        _gptq_ckpt = os.path.join(args.gptq_checkpoint_path, f'{model_name}_w{args.w_bits}')

        if os.path.isdir(_gptq_ckpt) and any(
                f.endswith('.pth') for f in os.listdir(_gptq_ckpt)):
            print(f"Loading existing GPTQ checkpoint from: {_gptq_ckpt}")
            utils.load_model_in_parts(model, _gptq_ckpt)
        else:
            print(f"Running GPTQ (w{args.w_bits}, groupsize={args.w_groupsize}) ...")
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
            gptq_args.percdamp = args.percdamp
            gptq_args.act_order = False
            gptq_args.w_static_groups = False

            gptq_utils.gptq_fwrd(model, trainloader, 'cuda', gptq_args)

            os.makedirs(_gptq_ckpt, exist_ok=True)
            print(f"Saving GPTQ checkpoint to: {_gptq_ckpt}")
            utils.save_model_in_parts(model, _gptq_ckpt,
                                      prefix=f'{model_name}_part')

        utils.cleanup_memory(verbos=True)

    # --- Configure activation quantizer bit-widths ---
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

        if 'v_proj' in name and args.v_bits < 16:
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
            layer_groupsize = down_proj_groupsize

        qlayers[name].quantizer.configure(
            bits=layer_input_bits, groupsize=layer_groupsize,
            sym=layer_a_sym, clip_ratio=layer_a_clip, residual=residual)

    # Add K-cache quantization wrappers
    if args.k_bits < 16:
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        # Cap k_groupsize at head_dim (mirrors shell script logic for small-head models)
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        k_groupsize = args.k_groupsize
        if k_groupsize > 0 and k_groupsize > head_dim:
            k_groupsize = head_dim
        k_quant_config = {
            'k_bits': args.k_bits, 'k_groupsize': k_groupsize,
            'k_sym': not args.k_asym, 'k_clip_ratio': args.k_clip_ratio,
            'use_r3': (args.mode in ('quarot', 'dart')),
        }
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn, rope_function_name,
                config=model.config, **k_quant_config)

    # --- Get calibration data ---
    dataloader = data_utils.get_loaders(
        args.calib_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False)

    # --- Run activation scale calibration ---
    act_scales = calibrate_act_scales(model, dataloader, args)

    # --- Save ---
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(act_scales, args.save_path)
    print(f"\nSaved {len(act_scales)} activation scale entries to {args.save_path}")
    for k in sorted(act_scales.keys()):
        s = act_scales[k]['scale']
        print(f"  {k}: scale shape={list(s.shape)}, range=[{s.min():.6f}, {s.max():.6f}]")


if __name__ == '__main__':
    main()
