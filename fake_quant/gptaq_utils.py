"""GPTAQ — closed-form FP-target variant of GPTQ.

For each Linear, the per-Linear loss is shifted from GPTQ's
``||W·X_Q - W'·X_Q||²`` to ``||W·X_FP - W'·X_Q||²``: match the FP-trajectory
output of the layer using the Q-trajectory input. The closed-form minimiser
is ``W' = W + W (C - H) H^-1`` where ``H = X_Q X_Q^T`` (already collected by
plain GPTQ) and ``C = X_FP X_Q^T`` is a NEW per-Linear cross-correlation
statistic.

This module mirrors :func:`gptq_utils.gptq_fwrd` but adds a parallel FP
forward pass per block (against a snapshot of the un-quantized block
weights) so the FP-trajectory inputs to every inner Linear are available.
At each calibration sample, FP and Q forwards run in lockstep and the
hooks accumulate ``H`` and ``C`` together via
:meth:`gptq_utils.GPTQ.add_cross_batch`.

After the per-subset Hessian is collected, every Linear is pre-shifted by
``W → W + δ`` (see :meth:`gptq_utils.GPTQ.pre_shift_fp_target`), then the
existing GPTQ Cholesky path quantises the shifted weight. All other
machinery — scalewise rounding, ``_gptq_w_scale`` collection, the
:func:`gptq_utils.GPTQ.fasterquant` Cholesky safeguard — applies unchanged.

Activated via ``--gptaq`` (requires ``--fp16_calib``); see
``main_for_test.py`` for the inference-side dispatch and
``calibrater/calibrate_act_scales.py`` for the calibration-side hook.
"""

import copy
import logging

import torch
import torch.nn as nn
import tqdm

import gptq_utils
import quant_utils
import utils


@torch.no_grad()
def gptaq_fwrd(model, dataloader, dev, args):
    """Sibling of :func:`gptq_utils.gptq_fwrd` for the GPTAQ closed-form path.

    Same outer structure (per-block iteration, four sub-layer subsets) but:

    1. At each block, snapshot FP weights (``fp_layer = deepcopy(layer)``)
       BEFORE any subset is quantised. Move to ``dev`` for FP-side forwards.
    2. For every subset, run **paired** FP + Q forwards per calibration
       sample. The Q hook reads the matching FP input from a temp dict and
       calls :meth:`gptq_utils.GPTQ.add_cross_batch` to accumulate both
       ``H`` and ``C`` in lockstep.
    3. Per Linear in the subset, call ``pre_shift_fp_target()`` to apply
       the closed-form shift ``W → W + W (C - H) H^-1``, then run the
       existing :meth:`gptq_utils.GPTQ.fasterquant` Cholesky path on the
       shifted weight.
    4. After all subsets are quantised, FP-forward ``fp_layer`` with
       ``fp_inps[i]`` to get ``fp_inps[i+1]`` for the next block. Q-forward
       the (now-quantised) ``layer`` to get ``q_inps[i+1]`` (existing).

    Memory: one extra ``deepcopy`` of the current block on ``dev``
    (~125 MB for Llama-3.2-1B, ~750 MB for Llama-2-7B). Constant overhead
    — discarded between blocks.
    """
    logging.info('-----GPTAQ Quantization-----')
    print(dev)

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
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev,
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

    # GPTAQ adds a parallel FP-trajectory inps buffer. The two trajectories
    # share the seed batch (block 0 input is the embedding output, identical
    # for both) and diverge as later blocks are quantised.
    inps_fp = inps.clone()
    outs = torch.zeros_like(inps)
    outs_fp = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache.get('position_embeddings', None)

    quantizers = {}
    sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module'],
    ]
    pbar = tqdm.tqdm(range(len(layers)), desc="(GPTAQ Quant.) Layers")
    for i in pbar:
        layer = layers[i].to(dev)
        # Snapshot FP weights of this block. Done BEFORE any subset in
        # this block is quantised so fp_layer holds the un-quantised
        # version of every inner Linear for the duration of block-i work.
        fp_layer = copy.deepcopy(layer).to(dev)

        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        full_fp = quant_utils.find_qlayers(fp_layer, layers=[torch.nn.Linear])
        layer_losses = []

        for names in sequential:
            subset = {}
            subset_fp = {}
            for n in names:
                if n in full:
                    subset[n] = full[n]
                    subset_fp[n] = full_fp[n]
                elif n.endswith('.module') and n[:-len('.module')] in full:
                    bare = n[:-len('.module')]
                    subset[bare] = full[bare]
                    subset_fp[bare] = full_fp[bare]

            gptq = {}
            w_bits_map = getattr(args, 'w_bits_map', None)
            scalewise = getattr(args, 'scalewise', False)
            hwscale_parsed = getattr(args, 'hwscale_parsed', None)
            fp16_act_scales = getattr(args, 'fp16_act_scales', None)
            for name in subset:
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if 'lm_head' in name:
                    continue
                bare_name = name.replace('.module', '')
                # --weight_group_mode: per-Linear weight groupsize override.
                # 'all' (default): every Linear at args.w_groupsize.
                # 'down':   keep down_proj per-group; rest per-channel.
                # 'down_o': keep down_proj AND o_proj per-group; rest per-channel.
                _wgm = getattr(args, 'weight_group_mode', 'all')
                _keep = (_wgm == 'all'
                         or any(f'{tok}_proj' in bare_name
                                for tok in _wgm.split('_')))
                layer_w_groupsize = args.w_groupsize if _keep else -1
                layer_scalewise = scalewise and (layer_w_groupsize > 0)
                if w_bits_map:
                    full_name = f'model.layers.{i}.{bare_name}'
                    layer_weight_bits = w_bits_map.get(full_name, layer_weight_bits)
                if args.w_bits_down_proj is not None and 'down_proj' in name:
                    layer_weight_bits = args.w_bits_down_proj
                fp4_mode = getattr(args, 'fp4', 'none')
                layer_use_fp4 = (fp4_mode == 'all') or (
                    fp4_mode == 'down' and 'down_proj' in name
                )
                if layer_use_fp4 and args.w_asym:
                    raise ValueError(
                        "--fp4 is symmetric-only; remove --w_asym for FP4 layers"
                    )
                # Scalewise FP16 act scale lookup — same as gptq_fwrd.
                layer_act_scale = None
                if layer_scalewise and fp16_act_scales is not None:
                    qkey = f'model.layers.{i}.{bare_name}.quantizer'
                    entry = fp16_act_scales.get(qkey)
                    if entry is not None:
                        a = entry.get('scale')
                        if (getattr(args, 'hw_accurate', False)
                                and 'down_proj' not in bare_name):
                            a = a.flatten().amax().expand(a.flatten().numel())
                        layer_act_scale = a
                gptq[name] = gptq_utils.GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym,
                    mse=args.w_clip,
                    gscaler=getattr(args, 'gscaler_parsed', None),
                    nvfp4=layer_use_fp4,
                    scalewise=layer_scalewise,
                    hwscale_spec=hwscale_parsed,
                    layer_act_scale=layer_act_scale,
                )
                if gptq[name].quantizer.scalewise:
                    gptq[name].quantizer.init_scalewise(
                        subset[name].weight.data, layer_w_groupsize)

            # Paired FP/Q activation capture. The FP hook stashes per-sample
            # X_FP per Linear in fp_cache; the Q hook reads the matching
            # entry and calls add_cross_batch (updates both H and C).
            fp_cache = {}

            def make_fp_hook(name):
                def fn(_, inp, _out):
                    fp_cache[name] = inp[0].data
                return fn

            def make_q_hook(name):
                def fn(_, inp, _out):
                    x_q = inp[0].data
                    x_fp = fp_cache.get(name)
                    if x_fp is None:
                        # FP forward should have populated this — fall back
                        # to vanilla GPTQ stat to avoid silent corruption.
                        gptq[name].add_batch(x_q, _out.data)
                    else:
                        gptq[name].add_cross_batch(x_fp, x_q)
                return fn

            fp_handles = [subset_fp[n].register_forward_hook(make_fp_hook(n))
                          for n in subset]
            q_handles = [subset[n].register_forward_hook(make_q_hook(n))
                         for n in subset]
            try:
                for j in range(args.nsamples):
                    fp_cache.clear()
                    # FP forward fills fp_cache for this sample.
                    _ = fp_layer(inps_fp[j].unsqueeze(0),
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 position_embeddings=position_embeddings)[0]
                    # Q forward reads fp_cache and accumulates H + C.
                    outs[j] = layer(inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings)[0]
            finally:
                for h in fp_handles:
                    h.remove()
                for h in q_handles:
                    h.remove()
                fp_cache.clear()

            _wgm = getattr(args, 'weight_group_mode', 'all')
            for name in subset:
                # GPTAQ: pre-shift W ← W + W (C - H) H^-1, then run GPTQ
                # on the shifted weight via the standard fasterquant path.
                gptq[name].pre_shift_fp_target(percdamp=args.percdamp)
                bare_name = name.replace('.module', '')
                _keep = (_wgm == 'all'
                         or any(f'{tok}_proj' in bare_name
                                for tok in _wgm.split('_')))
                layer_w_groupsize = args.w_groupsize if _keep else -1
                # If init_scalewise was called above on the un-shifted W,
                # re-init now so the per-layer hwscale global reflects the
                # shifted weight magnitudes.
                if gptq[name].quantizer.scalewise:
                    gptq[name].quantizer.init_scalewise(
                        subset[name].weight.data, layer_w_groupsize)

                loss = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize,
                    actorder=args.act_order, static_groups=args.w_static_groups,
                )
                layer_losses.append(loss)
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

            # Enable int_gemm on just-quantised group so the next subset's
            # Hessians reflect capped accumulator output (mirrors gptq_fwrd).
            if getattr(args, 'int_gemm', False):
                _enable_int_gemm_on_subset(layer, names, args, i, w_bits_map)

        # Safety-net pass: enable int_gemm on any remaining wrapped Linear.
        if getattr(args, 'int_gemm', False):
            _enable_int_gemm_on_subset(layer, None, args, i, w_bits_map)

        avg_loss = sum(layer_losses) / len(layer_losses) if layer_losses else 0.0
        pbar.set_postfix(loss=f"{avg_loss:.4g}")

        # Block tail: produce next-block inputs for both trajectories.
        # FP trajectory uses the FP snapshot (un-quantised); Q trajectory
        # uses ``layer`` (now-quantised in place by fasterquant).
        for j in range(args.nsamples):
            outs_fp[j] = fp_layer(
                inps_fp[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings)[0]
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        del fp_layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        inps_fp, outs_fp = outs_fp, inps_fp

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTAQ Quantization Done-----')
    return quantizers


def _enable_int_gemm_on_subset(layer, names, args, layer_idx, w_bits_map):
    """Enable capped int_gemm on Linears in *names* (or all wrapped Linears
    when ``names is None``). Same logic as gptq_fwrd's int_gemm follow-up;
    extracted so both gptq_fwrd and gptaq_fwrd can call it without
    duplicating the per-layer-bits resolution.
    """
    qlayers_ig = quant_utils.find_qlayers(
        layer, layers=[quant_utils.ActQuantWrapper])
    for qname, ql in qlayers_ig.items():
        if names is not None and (qname + '.module') not in names:
            continue
        if names is None and ql.use_int_gemm:
            continue
        if ql.quantizer.bits > 16:
            continue
        if ql.quantizer.bits <= 16 and getattr(ql.quantizer, 'groupsize', -1) <= 0:
            _ig_wb = args.w_bits
            bare = qname.replace('.module', '')
            if w_bits_map:
                _ig_wb = w_bits_map.get(f'model.layers.{layer_idx}.{bare}', _ig_wb)
            if (getattr(args, 'w_bits_down_proj', None) is not None
                    and 'down_proj' in qname):
                _ig_wb = args.w_bits_down_proj
            fp4_mode = getattr(args, 'fp4', 'none')
            layer_use_fp4 = (fp4_mode == 'all') or (
                fp4_mode == 'down' and 'down_proj' in qname
            )
            _wgm = getattr(args, 'weight_group_mode', 'all')
            _keep = (_wgm == 'all'
                     or any(f'{tok}_proj' in bare
                            for tok in _wgm.split('_')))
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
                nvfp4=layer_use_fp4,
            )
