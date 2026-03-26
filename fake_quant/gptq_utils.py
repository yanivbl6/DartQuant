import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Collect per-group scale/zero for asymmetric weights so that
        # prepare_int_weights can reuse GPTQ's exact parameters instead of
        # re-deriving them from fake-quantized values (which fails when not
        # all quantization levels are used in a group).
        # Skip when actorder=True + static_groups=False: columns are permuted
        # so per-group params in permuted order don't map to final column groups.
        _collect_gptq_params = (
            groupsize != -1
            and not self.quantizer.sym
            and not (actorder and not static_groups)
        )
        _gptq_scales = []  # will be [n_groups] list of [N,1] tensors
        _gptq_zeros = []

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)
                if _collect_gptq_params:
                    _gptq_scales.append(quantizer.scale.clone())
                    _gptq_zeros.append(quantizer.zero.clone())

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                            if _collect_gptq_params:
                                _gptq_scales.append(self.quantizer.scale.clone())
                                _gptq_zeros.append(self.quantizer.zero.clone())
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        mean_loss = Losses.mean().item()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

        # Store per-group GPTQ scale/zero on the layer as plain attributes
        # (NOT register_buffer — keeps the main checkpoint format unchanged).
        # prepare_int_weights checks for these and reuses them instead of
        # re-deriving from fake-quantized values.
        if _collect_gptq_params and _gptq_scales:
            # Each entry is [N, 1]; stack to [N, n_groups]
            self.layer._gptq_w_scale = torch.cat(_gptq_scales, dim=1)  # [N, n_groups]
            self.layer._gptq_w_zero = torch.cat(_gptq_zeros, dim=1)    # [N, n_groups]

        # For per-channel asymmetric (groupsize == -1), also save.
        if groupsize == -1 and not self.quantizer.sym:
            self.layer._gptq_w_scale = self.quantizer.scale.clone()
            self.layer._gptq_w_zero = self.quantizer.zero.clone()

        return mean_loss

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    print(dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    # for transformers >= 4.44.2,model.model has rotary emb
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache.get('position_embeddings', None)

    quantizers = {}
    sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module']
    ]
    pbar = tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Layers")
    for i in pbar:
        # print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        layer_losses = []
        for names in sequential:
            # Some layers may not be wrapped (e.g. k/v_proj when -k 16 -v 16),
            # so try both the .module name and the bare name.
            subset = {}
            for n in names:
                if n in full:
                    subset[n] = full[n]
                elif n.endswith('.module') and n[:-len('.module')] in full:
                    subset[n[:-len('.module')]] = full[n[:-len('.module')]]

            gptq = {}
            w_bits_map = getattr(args, 'w_bits_map', None)
            for name in subset:
                # print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if w_bits_map:
                    bare_name = name.replace('.module', '')
                    full_name = f'model.layers.{i}.{bare_name}'
                    layer_weight_bits = w_bits_map.get(full_name, layer_weight_bits)
                if args.w_bits_down_proj is not None and 'down_proj' in name:
                    layer_weight_bits = args.w_bits_down_proj
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip,
                    gscaler=getattr(args, 'gscaler_parsed', None)
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids, position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                loss = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize,
                    actorder=args.act_order, static_groups=args.w_static_groups,
                )
                layer_losses.append(loss)
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

            # Enable int_gemm on just-quantized group so subsequent groups'
            # Hessians reflect the capped accumulator output.
            # Three tiers: bits <= 8 → int8 GEMM, 9-16 → int16 GEMM, >16 → skip (fake-quant)
            if getattr(args, 'int_gemm', False):
                qlayers_ig = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
                for qname, ql in qlayers_ig.items():
                    if qname + '.module' not in names:
                        continue
                    if ql.quantizer.bits > 16:
                        pass  # too wide for int GEMM — leave as fake-quant
                    elif ql.quantizer.bits <= 16 and getattr(ql.quantizer, 'groupsize', -1) <= 0:
                        # Resolve per-layer w_bits from bit-width map
                        _ig_wb = args.w_bits
                        if w_bits_map:
                            bare = qname.replace('.module', '')
                            _ig_wb = w_bits_map.get(f'model.layers.{i}.{bare}', _ig_wb)
                        if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in qname:
                            _ig_wb = args.w_bits_down_proj
                        ql.prepare_int_gemm(
                            w_bits=_ig_wb,
                            w_sym=not args.w_asym,
                            w_group_size=args.w_groupsize,
                            acc_bits=args.acc_bits,
                            acc_block_k=args.acc_block_k,
                            use_triton=getattr(args, 'int_gemm_use_triton', True),
                            acc_wrap=getattr(args, 'acc_wrap', False),
                            acc_dtype=getattr(args, 'acc_dtype', 'float'),
                            gscaler_parsed=getattr(args, 'gscaler_parsed', None),
                        )

        # Enable capped int GEMM on any remaining layers (safety net)
        if getattr(args, 'int_gemm', False):
            qlayers_ig = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
            for qname, ql in qlayers_ig.items():
                if ql.use_int_gemm:
                    continue  # already set up per-group
                if ql.quantizer.bits > 16:
                    pass  # too wide for int GEMM — leave as fake-quant
                elif ql.quantizer.bits <= 16 and getattr(ql.quantizer, 'groupsize', -1) <= 0:
                    # Resolve per-layer w_bits from bit-width map
                    _ig_wb = args.w_bits
                    if w_bits_map:
                        bare = qname.replace('.module', '')
                        _ig_wb = w_bits_map.get(f'model.layers.{i}.{bare}', _ig_wb)
                    if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in qname:
                        _ig_wb = args.w_bits_down_proj
                    ql.prepare_int_gemm(
                        w_bits=_ig_wb,
                        w_sym=not args.w_asym,
                        w_group_size=args.w_groupsize,
                        acc_bits=args.acc_bits,
                        acc_block_k=args.acc_block_k,
                        use_triton=getattr(args, 'int_gemm_use_triton', True),
                        acc_wrap=getattr(args, 'acc_wrap', False),
                        acc_dtype=getattr(args, 'acc_dtype', 'float'),
                        gscaler_parsed=getattr(args, 'gscaler_parsed', None),
                    )

        avg_loss = sum(layer_losses) / len(layer_losses) if layer_losses else 0.0
        pbar.set_postfix(loss=f"{avg_loss:.4g}")

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----')
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, dev, args, stochastic=False):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}
    static_groups = args.w_static_groups
    groupsize = args.w_groupsize

    desc = "(Stochastic Quant.) Layers" if stochastic else "(RtN Quant.) Layers"
    for i in tqdm.tqdm(range(len(layers)), desc=desc):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                          layers=[torch.nn.Linear])

        w_bits_map = getattr(args, 'w_bits_map', None)
        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if w_bits_map:
                bare_name = name.replace('.module', '')
                full_name = f'model.layers.{i}.{bare_name}'
                layer_weight_bits = w_bits_map.get(full_name, layer_weight_bits)
            if args.w_bits_down_proj is not None and 'down_proj' in name:
                layer_weight_bits = args.w_bits_down_proj

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not (args.w_asym), mse=args.w_clip,
                gscaler=getattr(args, 'gscaler_parsed', None)
            )
            W = subset[name].weight.data

            if groupsize != -1:
                if static_groups:
                    assert W.shape[1] % groupsize == 0, "Weight matrix columns must be divisible by groupsize for static groups"
                    groups = []
                    for j in range(0, W.shape[1], groupsize):
                        group_quantizer = quant_utils.WeightQuantizer()
                        group_quantizer.configure(
                            layer_weight_bits, perchannel=True,
                            sym=not (args.w_asym), mse=args.w_clip,
                            gscaler=getattr(args, 'gscaler_parsed', None)
                        )
                        group_quantizer.find_params(W[:, j:j + groupsize])
                        groups.append(group_quantizer)
                else:
                    groups = None

                for j in range(0, W.shape[1], groupsize):
                    if not static_groups:
                        quantizer.find_params(W[:, j:j + groupsize])
                        quantized_w = quantizer.quantize(W[:, j:j + groupsize],
                                                         stochastic=stochastic)
                    else:
                        quantized_w = groups[j // groupsize].quantize(
                            W[:, j:j + groupsize], stochastic=stochastic)
                    W[:, j:j + groupsize] = quantized_w

            else:
                quantizer.find_params(W)
                W = quantizer.quantize(W, stochastic=stochastic)

            subset[name].weight.data = W.to(next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers
