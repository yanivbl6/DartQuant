import math
import os
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
        # GPTAQ cross-correlation X_FP @ X_Q^T (allocated lazily by
        # add_cross_batch — None ⇒ plain GPTQ path).
        self.C = None
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

    def add_cross_batch(self, inp_fp, inp_q):
        """GPTAQ-mode batch update — accumulates both H (= X_Q X_Q^T running
        mean) and C (= X_FP X_Q^T running mean) in lockstep on a single
        nsamples bump. Use this INSTEAD of add_batch when running GPTAQ; the
        two modes must not be mixed for a given GPTQ instance.

        inp_fp/inp_q: same shape ([B, S, in_features] or [B*S, in_features]),
        same per-sample order — token j of inp_q must be the Q-trajectory
        counterpart of token j of inp_fp.
        """
        if self.C is None:
            self.C = torch.zeros((self.columns, self.columns), device=self.dev)
        if len(inp_q.shape) == 2:
            inp_q = inp_q.unsqueeze(0)
        if len(inp_fp.shape) == 2:
            inp_fp = inp_fp.unsqueeze(0)
        tmp = inp_q.shape[0]
        assert inp_fp.shape[0] == tmp, \
            f"FP/Q batch mismatch: {inp_fp.shape[0]} vs {tmp}"
        if len(inp_q.shape) == 3:
            inp_q = inp_q.reshape((-1, inp_q.shape[-1]))
        if len(inp_fp.shape) == 3:
            inp_fp = inp_fp.reshape((-1, inp_fp.shape[-1]))
        assert inp_fp.shape == inp_q.shape, \
            f"FP/Q token mismatch: {inp_fp.shape} vs {inp_q.shape}"
        inp_q = inp_q.t()
        inp_fp = inp_fp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.C *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        scale = math.sqrt(2 / self.nsamples)
        inp_q_s = scale * inp_q.float()
        inp_fp_s = scale * inp_fp.float()
        self.H += inp_q_s.matmul(inp_q_s.t())
        self.C += inp_fp_s.matmul(inp_q_s.t())

    def pre_shift_fp_target(self, percdamp=0.01):
        """GPTAQ closed-form pre-shift on self.layer.weight.

        Solves   W' = W + W (C - H) H^-1   so the per-Linear loss
        ||W·X_FP - W'·X_Q||² is minimised before the GPTQ Cholesky loop
        runs. See plan: enchanted-stargazing-token.md ("Algorithm").

        Mutates self.layer.weight in place. No-op when self.C is None
        (GPTAQ inactive) or when add_cross_batch was never called.

        Set DART_DISABLE_PRESHIFT=1 to ablate the pre-shift entirely
        (turns gptaq into vanilla GPTQ for diagnostic A/B comparison).
        """
        if self.C is None or self.nsamples == 0:
            return
        if os.environ.get('DART_DISABLE_PRESHIFT') == '1':
            logging.warning("GPTAQ pre-shift: ABLATED via DART_DISABLE_PRESHIFT=1")
            return
        # Two views of H: H_raw is the un-regularised statistic that goes
        # into (C - H); H_reg is the same matrix with NaN scrub + diagonal
        # floor + percdamp applied, used only for the solve. Mixing them
        # breaks the identity case (X_FP == X_Q ⇒ C == H_raw ⇒ no shift).
        H_raw = self.H.clone()
        C = self.C.clone()
        nan_axis = torch.isnan(torch.diagonal(H_raw))
        if nan_axis.any():
            n_nan = int(nan_axis.sum().item())
            logging.warning(
                "GPTAQ pre-shift: %d column(s) contaminated with NaN; isolating",
                n_nan)
            H_raw[nan_axis, :] = 0.0
            H_raw[:, nan_axis] = 0.0
            H_raw = torch.where(torch.isnan(H_raw),
                                torch.zeros_like(H_raw), H_raw)
            C[nan_axis, :] = 0.0
            C[:, nan_axis] = 0.0
            C = torch.where(torch.isnan(C), torch.zeros_like(C), C)

        # Guard 1: skip when (C - H) is at numerical-noise level relative to
        # H. When trajectories coincide (e.g. block 0, where X_FP == X_Q
        # exactly) the mathematical shift is zero, but cuBLAS computes A·A^T
        # and A·B^T via different reduction orders, leaving a tiny residual
        # that H^-1 then amplifies into a real weight perturbation.
        CH_norm = (C - H_raw).norm()
        H_norm = H_raw.norm().clamp(min=1e-12)
        rel_noise = (CH_norm / H_norm).item()
        if rel_noise < 1e-3:
            logging.warning(
                "GPTAQ pre-shift: skip — (C-H)/H rel norm %.2e below 1e-3",
                rel_noise)
            return
        # Diagnostic: log the rel_noise when it doesn't trigger the guard
        # so we can tune the threshold. (warning level so it survives
        # calibrate_act_scales' default Python logging level.)
        logging.warning("GPTAQ pre-shift diag: rel_noise=%.4e", rel_noise)

        diag = torch.arange(self.columns, device=self.dev)
        H_diag = torch.diag(H_raw)
        diag_pos = H_diag[H_diag > 0]
        diag_floor = (diag_pos.mean().item() * 1e-3) if diag_pos.numel() > 0 else 1e-8
        damp = percdamp * torch.mean(H_diag.clamp(min=diag_floor))
        H_reg = H_raw.clone()
        H_reg[diag, diag] = torch.maximum(
            H_reg[diag, diag] + damp,
            torch.tensor(diag_floor, device=self.dev))
        # Solve H_reg · Y^T = (C - H_raw)^T  ⇒  Y = (C - H_raw) · H_reg^-1.
        # NB: (C - H_raw) is what carries the FP/Q gap signal — when the
        # trajectories agree it is exactly zero and the shift vanishes.
        M_rhs = (C - H_raw).t()
        try:
            Y_t = torch.linalg.solve(H_reg, M_rhs)
        except torch._C._LinAlgError as exc:
            logging.warning(
                "GPTAQ pre-shift: linalg.solve failed (%s); skipping pre-shift",
                exc)
            return
        Y = Y_t.t()
        W = self.layer.weight.data.float()
        delta = W @ Y                            # [rows, columns]
        if torch.isnan(delta).any():
            logging.warning(
                "GPTAQ pre-shift: NaN in delta; skipping pre-shift")
            return

        # Guard 2: cap |delta|/|W| at MAX_REL. Ill-conditioned H (low-rank
        # along low-activation channels — common in down_proj with 8192-dim
        # input) makes the solve return a Y whose columns can have large
        # eigenvalues, and delta = W·Y overwhelms W. Scale the shift down
        # so the legitimate direction survives without blowing up magnitude.
        MAX_REL = 0.1
        delta_norm = delta.norm()
        W_norm = W.norm().clamp(min=1e-12)
        rel = (delta_norm / W_norm).item()
        # Diagnostic: log every call so we can tune the threshold.
        logging.warning("GPTAQ pre-shift diag: |delta|/|W|=%.4e", rel)
        if rel > MAX_REL:
            scale = MAX_REL / rel
            logging.warning(
                "GPTAQ pre-shift: |delta|/|W|=%.3f > %.2f; scaling by %.3f",
                rel, MAX_REL, scale)
            delta = delta * scale

        self.layer.weight.data = (W + delta).to(self.layer.weight.data.dtype)

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            # For scalewise, this initial whole-W call would snap once over the
            # entire K-range; the per-group calls below overwrite it. Reset
            # the K cursor first so the act-scale window is taken from K=0.
            if getattr(self.quantizer, 'scalewise', False):
                self.quantizer._k_start = 0
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Collect per-group scale/zero so prepare_int_weights can reuse GPTQ's
        # exact parameters instead of re-deriving them from fake-quantized
        # values (which fails when not all quantization levels are used in a
        # group, or when the level set is non-uniform like FP4).
        # Also required for scalewise: the GPTQ-time scale is intentionally
        # rounded to the M<m>S<s> grid (combined_q / a_group), so recovery
        # from fake-quantized weights would give a different scale and
        # break the runtime assumption that qlayer.w_scale * a_scale equals
        # the rounded combined_q.
        # Skip when actorder=True + static_groups=False: columns are permuted
        # so per-group params in permuted order don't map to final column groups.
        _collect_gptq_params = (
            groupsize != -1
            and (not self.quantizer.sym
                 or getattr(self.quantizer, 'nvfp4', False)
                 or getattr(self.quantizer, 'scalewise', False)
                 or getattr(self.quantizer, 'gscaler', None) is not None)
            and not (actorder and not static_groups)
        )
        _gptq_scales = []  # will be [n_groups] list of [N,1] tensors
        _gptq_zeros = []

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                if getattr(quantizer, 'scalewise', False):
                    quantizer._k_start = i
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

        # Cholesky path with numerical safeguards. Under scalewise the per-group
        # weight scale can land on the M<m>S<s> grid edge where a few groups
        # snap to underflow values, which propagates through the GPTQ chain
        # (later subsets see slightly different activations) and can produce
        # zero-variance columns in H or NaN entries. Detect and recover by
        # isolating the contaminated rows/cols (eye-style) and bumping
        # percdamp until Cholesky succeeds.
        diag = torch.arange(self.columns, device=self.dev)
        H_orig = H.clone()
        # NaN contamination: a NaN input column produces a NaN row+col in H
        # (and via outer-product propagation, NaN spreads symmetrically).
        # A column k is the contamination source when H[k,k] is NaN; the
        # rest of H[k, :] / H[:, k] inherit NaN but the bad axis is k.
        # Identify bad axes by NaN diagonal entries, then wipe just those
        # rows/cols so untouched columns retain their real data.
        nan_axis = torch.isnan(torch.diagonal(H_orig))
        if nan_axis.any():
            n_nan = int(nan_axis.sum().item())
            logging.warning(
                "GPTQ: %d column(s) contaminated with NaN; isolating before Cholesky",
                n_nan)
            H_orig[nan_axis, :] = 0.0
            H_orig[:, nan_axis] = 0.0
            # Replace residual NaNs (cells whose row OR col was bad) with 0.
            H_orig = torch.where(torch.isnan(H_orig),
                                  torch.zeros_like(H_orig), H_orig)
        H_diag = torch.diag(H_orig)
        # Floor any near-zero diagonal entries so damping has something to
        # ride on (otherwise dead columns stay at ~0 + percdamp*mean which
        # may itself be 0 if the whole layer is dead).
        diag_pos = H_diag[H_diag > 0]
        diag_floor = (diag_pos.mean().item() * 1e-3) if diag_pos.numel() > 0 else 1e-8
        Hinv = None
        last_err = None
        for retry_damp in (percdamp, percdamp * 5.0, percdamp * 50.0):
            H = H_orig.clone()
            damp = retry_damp * torch.mean(torch.diag(H).clamp(min=diag_floor))
            H[diag, diag] = torch.maximum(
                H[diag, diag] + damp,
                torch.tensor(diag_floor, device=self.dev))
            try:
                L = torch.linalg.cholesky(H)
                Hi = torch.cholesky_inverse(L)
                Hinv = torch.linalg.cholesky(Hi, upper=True)
                if retry_damp != percdamp:
                    logging.warning(
                        "GPTQ: Cholesky required percdamp=%.3g (default %.3g)",
                        retry_damp, percdamp)
                break
            except torch._C._LinAlgError as exc:
                last_err = exc
                continue
        if Hinv is None:
            raise last_err
        H = Hinv

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
                            if getattr(self.quantizer, 'scalewise', False):
                                self.quantizer._k_start = i1 + i
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

        # For per-channel (groupsize == -1), also save when:
        #   - asym: zero point is data-dependent, recovery can't infer it
        #   - nvfp4: non-uniform code set, recovery can't infer
        #   - sym + gscaler: find_params snapped the scale to the M<m>S<s>
        #     grid; runtime int_gemm recovery (abs_max/maxq from fake-quant W)
        #     returns the un-snapped value, so prepare_int_weights would use
        #     a scale GPTQ never optimized against.
        if groupsize == -1 and (not self.quantizer.sym
                                or getattr(self.quantizer, 'nvfp4', False)
                                or getattr(self.quantizer, 'gscaler', None) is not None):
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
            scalewise = getattr(args, 'scalewise', False)
            hwscale_parsed = getattr(args, 'hwscale_parsed', None)
            fp16_act_scales = getattr(args, 'fp16_act_scales', None)
            for name in subset:
                # print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
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
                # scalewise requires a positive groupsize (init_scalewise raises
                # otherwise). Disable per-Linear for the per-channel case.
                layer_scalewise = scalewise and (layer_w_groupsize > 0)
                if w_bits_map:
                    full_name = f'model.layers.{i}.{bare_name}'
                    layer_weight_bits = w_bits_map.get(full_name, layer_weight_bits)
                if args.w_bits_down_proj is not None and 'down_proj' in name:
                    layer_weight_bits = args.w_bits_down_proj
                fp4_mode = getattr(args, 'fp4', 'none')
                layer_use_fp4 = (fp4_mode == 'all') or (
                    fp4_mode == 'down' and 'down_proj' in name
                ) or (
                    fp4_mode == 'down_o'
                    and ('down_proj' in name or 'o_proj' in name)
                )
                if layer_use_fp4 and args.w_asym:
                    raise ValueError(
                        "--fp4 is symmetric-only; remove --w_asym for FP4 layers"
                    )
                # Look up the FP16 act scale for this layer (scalewise only)
                layer_act_scale = None
                if layer_scalewise and fp16_act_scales is not None:
                    qkey = f'model.layers.{i}.{bare_name}.quantizer'
                    entry = fp16_act_scales.get(qkey)
                    if entry is not None:
                        a = entry.get('scale')
                        # Match the runtime act-scale grouping so the snap
                        # is at the same granularity as the deployed scale.
                        # hw_accurate collapses non-down_proj to per-tensor;
                        # down_proj keeps per-column (reduced to per-K-group
                        # at runtime, which matches the per-weight-group
                        # max we take inside find_params).
                        if (getattr(args, 'hw_accurate', False)
                                and 'down_proj' not in bare_name):
                            a = a.flatten().amax().expand(a.flatten().numel())
                        layer_act_scale = a
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip,
                    gscaler=getattr(args, 'gscaler_parsed', None),
                    nvfp4=layer_use_fp4,
                    scalewise=layer_scalewise,
                    hwscale_spec=hwscale_parsed,
                    layer_act_scale=layer_act_scale,
                )
                if gptq[name].quantizer.scalewise:
                    gptq[name].quantizer.init_scalewise(
                        subset[name].weight.data, layer_w_groupsize)

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

            _wgm = getattr(args, 'weight_group_mode', 'all')
            for name in subset:
                bare_name = name.replace('.module', '')
                _keep = (_wgm == 'all'
                         or any(f'{tok}_proj' in bare_name
                                for tok in _wgm.split('_')))
                layer_w_groupsize = args.w_groupsize if _keep else -1
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
                        bare = qname.replace('.module', '')
                        if w_bits_map:
                            _ig_wb = w_bits_map.get(f'model.layers.{i}.{bare}', _ig_wb)
                        if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in qname:
                            _ig_wb = args.w_bits_down_proj
                        _ig_fp4_mode = getattr(args, 'fp4', 'none')
                        _ig_use_fp4 = (_ig_fp4_mode == 'all') or (
                            _ig_fp4_mode == 'down' and 'down_proj' in qname
                        ) or (
                            _ig_fp4_mode == 'down_o'
                            and ('down_proj' in qname or 'o_proj' in qname)
                        )
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
                            nvfp4=_ig_use_fp4,
                        )

        # Enable capped int GEMM on any remaining layers (safety net)
        if getattr(args, 'int_gemm', False):
            qlayers_ig = quant_utils.find_qlayers(layer, layers=[quant_utils.ActQuantWrapper])
            _wgm_safety = getattr(args, 'weight_group_mode', 'all')
            for qname, ql in qlayers_ig.items():
                if ql.use_int_gemm:
                    continue  # already set up per-group
                if ql.quantizer.bits > 16:
                    pass  # too wide for int GEMM — leave as fake-quant
                elif ql.quantizer.bits <= 16 and getattr(ql.quantizer, 'groupsize', -1) <= 0:
                    # Resolve per-layer w_bits from bit-width map
                    _ig_wb = args.w_bits
                    bare = qname.replace('.module', '')
                    if w_bits_map:
                        _ig_wb = w_bits_map.get(f'model.layers.{i}.{bare}', _ig_wb)
                    if getattr(args, 'w_bits_down_proj', None) is not None and 'down_proj' in qname:
                        _ig_wb = args.w_bits_down_proj
                    _ig_fp4_mode = getattr(args, 'fp4', 'none')
                    _ig_use_fp4 = (_ig_fp4_mode == 'all') or (
                        _ig_fp4_mode == 'down' and 'down_proj' in qname
                    ) or (
                        _ig_fp4_mode == 'down_o'
                        and ('down_proj' in qname or 'o_proj' in qname)
                    )
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
                        nvfp4=_ig_use_fp4,
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
            fp4_mode = getattr(args, 'fp4', 'none')
            layer_use_fp4 = (fp4_mode == 'all') or (
                fp4_mode == 'down' and 'down_proj' in name
            ) or (
                fp4_mode == 'down_o'
                and ('down_proj' in name or 'o_proj' in name)
            )
            if layer_use_fp4 and args.w_asym:
                raise ValueError(
                    "--fp4 is symmetric-only; remove --w_asym for FP4 layers"
                )

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not (args.w_asym), mse=args.w_clip,
                gscaler=getattr(args, 'gscaler_parsed', None),
                nvfp4=layer_use_fp4,
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
                            gscaler=getattr(args, 'gscaler_parsed', None),
                            nvfp4=layer_use_fp4,
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
