"""
Calibrate static activation scales for all ActQuantWrapper and QKRotationWrapper
quantizers. Runs the model layer-by-layer on calibration data, collecting per-channel
min/max statistics, then saves {name: {scale, zero}} to a .pt file.

Usage:
    python calibrate_act_scales.py \
        --model /path/to/model \
        --mode dart \
        --r1_path ../data/trained_rotation/.../r1/... \
        --r2_path ../data/trained_rotation/.../r2/... \
        --a_bits 8 --k_bits 4 --v_bits 4 \
        --a_groupsize -1 --k_groupsize 128 --v_groupsize 128 \
        --a_clip_ratio 0.9 --a_asym \
        --save_path ../data/act_scales/model_name/scales.pt
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import gc
import functools
import math
from tqdm import tqdm

sys.path.append('../fake_quant')

import model_utils
import data_utils
import rotation_utils
import quant_utils
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
    # We accumulate running min/max across all samples.
    act_scales = {}

    for i in tqdm(range(len(layers)), desc="Calibrating layers"):
        layer = layers[i].to(dev)

        # Collectors: {quantizer_name: {"min": tensor, "max": tensor}}
        collectors = {}

        def collect_hook(module, inp, out, name):
            """Forward hook to collect min/max of the input tensor."""
            x = inp[0] if isinstance(inp, tuple) else inp
            # Reduce over all dims except the last (channel dim) -> per-channel stats
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
                # For K-cache, we need to collect after RoPE + R3 Hadamard.
                # The wrapper itself applies find_params dynamically; instead
                # we hook the wrapper's forward to capture the k tensor.
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
                        # Still need to quantize K for correct layer output
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

        # Remove hooks, restore original forward
        for h in hooks:
            h.remove()
        if hasattr(layer.self_attn, wrapper_attr):
            wrapper = getattr(layer.self_attn, wrapper_attr)
            if hasattr(wrapper, '_orig_forward'):
                wrapper.forward = wrapper._orig_forward

        # Compute scale/zero from collected min/max
        for cname, stats in collectors.items():
            cmin = stats['min']
            cmax = stats['max']

            # Determine quantizer config to compute scales correctly
            # Find the matching quantizer to get bits/sym/clip_ratio
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

    # ActQuantWrapper quantizers: "model.layers.{i}.{subname}.quantizer"
    prefix = f'model.layers.{layer_idx}.'
    if cname.startswith(prefix) and cname.endswith('.quantizer'):
        subname = cname[len(prefix):-len('.quantizer')]
        if subname in qlayers:
            return qlayers[subname].quantizer

    if cname.startswith(prefix) and cname.endswith('.out_quantizer'):
        subname = cname[len(prefix):-len('.out_quantizer')]
        if subname in qlayers:
            return qlayers[subname].out_quantizer

    # K-cache quantizer: "layer.{i}.k_quantizer"
    if cname == f'layer.{layer_idx}.k_quantizer':
        wrapper_attr = f'{rope_fn}_qk_rotation_wrapper'
        if hasattr(layer.self_attn, wrapper_attr):
            return getattr(layer.self_attn, wrapper_attr).k_quantizer

    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate static activation scales')
    parser.add_argument('--model', type=str, required=True, help='Model path or HF name')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--calib_dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'])

    # Mode: determines rotation config
    parser.add_argument('--mode', type=str, default='dart',
                        choices=['baseline', 'quarot', 'dart'],
                        help='Which rotation mode to use (determines R1-R4 config)')
    parser.add_argument('--r1_path', type=str, default=None)
    parser.add_argument('--r2_path', type=str, default=None)
    parser.add_argument('--rotate_mode', type=str, default='hadamard',
                        choices=['hadamard', 'random'])

    # Quantization config (must match experiment config)
    parser.add_argument('--a_bits', type=int, default=8)
    parser.add_argument('--a_groupsize', type=int, default=-1)
    parser.add_argument('--a_asym', action='store_true', default=False)
    parser.add_argument('--a_clip_ratio', type=float, default=0.9)
    parser.add_argument('--a_residual', action='store_true', default=False)
    parser.add_argument('--k_bits', type=int, default=4)
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action='store_true', default=False)
    parser.add_argument('--k_clip_ratio', type=float, default=1.0)
    parser.add_argument('--v_bits', type=int, default=4)
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action='store_true', default=False)
    parser.add_argument('--v_clip_ratio', type=float, default=1.0)
    parser.add_argument('--o_per_head', action='store_true', default=False)
    parser.add_argument('--fp32_had', action='store_true', default=False)

    parser.add_argument('--save_path', type=str, required=True,
                        help='Where to save the calibrated scales .pt file')
    return parser.parse_args()


def main():
    args = parse_args()
    import transformers
    transformers.set_seed(args.seed)

    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()

    # --- Set up rotations to match experiment pipeline ---
    if args.mode in ('quarot', 'dart'):
        rotation_utils.fuse_layer_norms(model)

        # Build a namespace that rotation_utils.rotate_model expects
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

        rotation_utils.rotate_model(model, rot_args)
    elif args.mode == 'baseline':
        # No rotations, no norm fusion
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

    # Configure quantizer bit-widths
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
        k_quant_config = {
            'k_bits': args.k_bits, 'k_groupsize': args.k_groupsize,
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

    # --- Run calibration ---
    act_scales = calibrate_act_scales(model, dataloader, args)

    # --- Save ---
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(act_scales, args.save_path)
    print(f"Saved {len(act_scales)} activation scale entries to {args.save_path}")
    for k in sorted(act_scales.keys()):
        s = act_scales[k]['scale']
        print(f"  {k}: scale shape={list(s.shape)}, range=[{s.min():.6f}, {s.max():.6f}]")


if __name__ == '__main__':
    main()
