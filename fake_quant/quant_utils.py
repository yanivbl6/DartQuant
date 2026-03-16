import math
import re
import transformers
import torch
import utils
import hadamard_utils
import fast_hadamard_transform


# ── Mantissa+Shift group-scale quantization ──────────────────────────────────

_GSCALER_RE = re.compile(r'^M(\d+)[SE](\d+)(?:([bl])(\d+))?$')


def parse_gscaler(spec):
    """Parse a gscaler spec string into a config dict.

    Examples:
        'M5S3'    -> {mantissa_bits: 5, shift_bits: 3, bias: 0}
        'M6E4b2'  -> {mantissa_bits: 6, shift_bits: 4, bias: 2}
        'M6S4l2'  -> {mantissa_bits: 6, shift_bits: 4, bias: -2}

    Returns None if spec is None.
    Raises ValueError on invalid format.
    """
    if spec is None:
        return None
    m = _GSCALER_RE.match(spec)
    if not m:
        raise ValueError(
            f"Invalid --gscaler format: '{spec}'. "
            f"Expected e.g. M5S3, M6E4b2, M6S4l2.")
    mantissa_bits = int(m.group(1))
    shift_bits = int(m.group(2))
    bias_dir = m.group(3)  # 'b', 'l', or None
    bias_val = int(m.group(4)) if m.group(4) else 0
    if bias_dir == 'l':
        bias_val = -bias_val
    return {
        'mantissa_bits': mantissa_bits,
        'shift_bits': shift_bits,
        'bias': bias_val,
        'spec': spec,
    }


def snap_scale_to_gscaler(scale, gscaler):
    """Snap FP32 scale tensor to nearest mantissa+shift representable value.

    scale: tensor of positive FP32 values
    gscaler: dict from parse_gscaler()

    Returns: tensor of same shape with snapped values.
    Format: value = mantissa * 2^(-shift)
    """
    M = gscaler['mantissa_bits']
    S = gscaler['shift_bits']
    bias = gscaler['bias']
    max_mantissa = (1 << M) - 1  # 2^M - 1
    s_min = bias
    s_max = bias + (1 << S) - 1  # bias + 2^S - 1

    orig_shape = scale.shape
    s_flat = scale.flatten().float()

    best_val = torch.full_like(s_flat, float('inf'))
    best_err = torch.full_like(s_flat, float('inf'))

    for shift in range(s_min, s_max + 1):
        # mantissa = round(scale * 2^shift), clamped to [1, max_mantissa]
        two_pow_shift = 2.0 ** shift
        m = torch.clamp(torch.round(s_flat * two_pow_shift), 1, max_mantissa)
        candidate = m * (2.0 ** (-shift))
        err = (candidate - s_flat).abs()
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_val = torch.where(better, candidate, best_val)

    return best_val.reshape(orig_shape).to(scale.dtype)


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def _stochastic_round(x):
    """Round with probability proportional to fractional part (unbiased)."""
    floor = x.floor()
    return floor + (torch.rand_like(x) < (x - floor)).to(x.dtype)


def asym_quant(x, scale, zero, maxq, stochastic=False):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    _round = _stochastic_round if stochastic else torch.round
    q = torch.clamp(_round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq, stochastic=False):
    return asym_dequant(*asym_quant(x, scale, zero, maxq, stochastic=stochastic))


def sym_quant(x, scale, maxq, stochastic=False):
    scale = scale.to(x.device)
    _round = _stochastic_round if stochastic else torch.round
    q = torch.clamp(_round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq, stochastic=False):
    return sym_dequant(*sym_quant(x, scale, maxq, stochastic=stochastic))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.


def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16
        self.static = False
        # sd_check: compare static vs dynamic quantization
        self._sd_check = 0        # threshold (0 = disabled)
        self._sd_norm = float('inf')
        self._sd_name = ''
        self._in_sd_check = False  # re-entry guard
        self._sd_logged_first = False

    def free(self):
        if self.static:
            return  # keep pre-calibrated scales
        self.zero = None
        self.scale = None

    def _sd_report(self, result_static, result_dynamic):
        """Report sd_check relative error between static and dynamic results."""
        diff = result_static.float() - result_dynamic.float()
        denom = torch.linalg.vector_norm(result_dynamic.float(), ord=self._sd_norm)
        rel_err = (torch.linalg.vector_norm(diff, ord=self._sd_norm) / denom).item() if denom > 0 else 0.0

        norm_str = {1: 'L1', 2: 'L2', float('inf'): 'Linf'}[self._sd_norm]
        if not self._sd_logged_first:
            print(f"[sd_check] {self._sd_name}: {norm_str}_rel_err={rel_err:.4e}", flush=True)
            self._sd_logged_first = True
        if rel_err > self._sd_check:
            print(f"[sd_check] {self._sd_name}: {norm_str}_rel_err={rel_err:.4e} "
                  f"EXCEEDS threshold {self._sd_check:.4e}", flush=True)

    def _compare_sd(self, x, q_static):
        """Compare static fake-quant result with what dynamic would produce."""
        saved_scale = self.scale.clone()
        saved_zero = self.zero.clone() if self.zero is not None else None
        self.static = False
        self._in_sd_check = True
        q_dynamic = self.forward(x)
        self._in_sd_check = False
        self.static = True
        self.scale = saved_scale
        if saved_zero is not None:
            self.zero = saved_zero

        self._sd_report(q_static, q_dynamic)

    def forward(self, x):
        x_dtype = x.dtype

        if self.bits == 16:
            return x

        if not self.static:
            self.find_params(x)  # dynamic: recompute every forward
        # else: use pre-loaded self.scale / self.zero

        if self.residual:
            if self.sym:
                tmp_q = sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = sym_quant_dequant(x - tmp_q, self.scale, self.maxq).to(x_dtype)
                result = tmp_q + residual_q
            else:
                tmp_q = asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = asym_quant_dequant(x - tmp_q, self.scale, self.zero, self.maxq).to(x_dtype)
                result = tmp_q + residual_q
        elif self.sym:
            result = sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        else:
            result = asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

        if self._sd_check > 0 and self.static and not self._in_sd_check:
            self._compare_sd(x, result)

        return result

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def quantize_to_int(self, x):
        """Quantize activations to int8 and return (q_int8, scale_vec, zp_correction).

        Unlike ``forward`` (fake-quant) or ``quantize`` (returns float integers),
        this method returns *actual* int8 values and a 1-D scale vector,
        suitable for feeding into the integer GEMM kernel.

        Static mode with groupsize > 0: per-group scales along K, aligned with
        acc_block_k.  Each group of G columns shares one scale.  Returns
        ``(q_int8[M,K], group_scales[n_groups], None)``.

        Static mode with groupsize <= 0: per-tensor scale (max of all column
        scales).  Returns ``(q_int8[M,K], per_token_scale[M], zp_correction)``.

        Dynamic mode: computes per-token scales from x at runtime.
        Returns ``(q_int8[M,K], per_token_scale[M], zp_correction)``.

        For asymmetric quantization, q is shifted into signed int8 range and a
        zero-point correction factor is returned.  The caller must combine it
        with the precomputed ``w_zp_correction`` vector:

            output += zp_correction[:, None] * w_zp_correction[None, :]

        Requires bits <= 8.
        """
        assert self.bits <= 8, \
            f"quantize_to_int requires bits <= 8 (got {self.bits}); values would overflow int8"

        dev = x.device
        self.maxq = self.maxq.to(dev)
        K = x.shape[-1]
        M = x.reshape(-1, K).shape[0]

        if self.static and self.groupsize > 0:
            # Per-group static scales, pre-derived from per-column calibration.
            # self.scale is [n_groups] (set during setup in main_for_test.py).
            G = self.groupsize
            n_groups = K // G
            group_scale = self.scale.to(dev)  # [n_groups]
            group_zero = self.zero.to(dev)    # [n_groups]

            x_2d = x.reshape(M, K)
            x_grouped = x_2d.reshape(M, n_groups, G)

            if self.sym:
                q_grouped = torch.clamp(
                    torch.round(x_grouped / group_scale[None, :, None]),
                    -(self.maxq + 1), self.maxq)
                q_int8 = q_grouped.reshape(M, K).to(torch.int8)
            else:
                # Asymmetric: quantize to [0, maxq], shift to signed int8
                q_grouped = torch.clamp(
                    torch.round(x_grouped / group_scale[None, :, None])
                    + group_zero[None, :, None],
                    0, self.maxq)
                shift = (int(self.maxq.item()) + 1) // 2  # 128 for 8-bit
                q_int8 = (q_grouped - shift).reshape(M, K).to(torch.int8)

            # zp_correction is None: for per-group static asymmetric, the
            # correction is precomputed as static_zp_bias in ActQuantWrapper.
            return q_int8, group_scale, None

        if self.static:
            # Per-tensor static scale (groupsize <= 0 fallback).
            # Collapse per-column scales to a single scalar = max over columns.
            col_scale = self.scale.to(dev).flatten()

            if self.sym:
                per_tensor_scale = col_scale.max()
                q = torch.clamp(torch.round(x / per_tensor_scale),
                                -(self.maxq + 1), self.maxq)
                q_int8 = q.reshape(-1, q.shape[-1]).to(torch.int8)
                per_token_scale = torch.full((M,), per_tensor_scale.item(),
                                             device=dev, dtype=col_scale.dtype)
                zp_correction = None
            else:
                col_zero = self.zero.to(dev).flatten()
                # Recover representable range per column, then global extremes
                col_min = -col_zero * col_scale
                col_max = (self.maxq - col_zero) * col_scale
                global_min = col_min.min()
                global_max = col_max.max()
                per_tensor_scale = (global_max - global_min) / self.maxq
                if per_tensor_scale == 0:
                    per_tensor_scale = torch.ones_like(per_tensor_scale)
                global_zero = torch.round(-global_min / per_tensor_scale)

                q = torch.clamp(torch.round(x / per_tensor_scale) + global_zero,
                                0, self.maxq)
                shift = (int(self.maxq.item()) + 1) // 2
                q_int8 = (q - shift).reshape(-1, q.shape[-1]).to(torch.int8)
                per_token_scale = torch.full((M,), per_tensor_scale.item(),
                                             device=dev, dtype=col_scale.dtype)
                zp_corr_scalar = per_tensor_scale * (shift - global_zero)
                zp_correction = torch.full((M,), zp_corr_scalar.item(),
                                           device=dev, dtype=col_scale.dtype)

            return q_int8, per_token_scale, zp_correction

        # Dynamic: compute per-token scales from x at runtime.
        assert self.groupsize <= 0, \
            "quantize_to_int with groupsize > 0 requires static mode"
        self.find_params(x)

        # Extract per-token scalar scale (works for both sym and asym).
        # self.scale has shape [..., K] with repeated values along K.
        # IMPORTANT: flat_scale[:, 0] has stride K (non-contiguous).  The
        # Triton kernel indexes A_scale_ptr with stride 1, so we MUST return
        # a contiguous 1-D tensor — otherwise the kernel reads garbage.
        flat_scale = self.scale.reshape(-1, self.scale.shape[-1])
        per_token_scale = flat_scale[:, 0].contiguous()  # [M], stride-1

        if self.sym:
            q, _scale = sym_quant(x, self.scale, self.maxq)
            q_int8 = q.reshape(-1, q.shape[-1]).to(torch.int8)
            zp_correction = None
        else:
            q, _scale, zero = asym_quant(x, self.scale, self.zero, self.maxq)
            # q is in [0, maxq]. Shift into signed int8 range.
            shift = (int(self.maxq.item()) + 1) // 2   # 128 for 8-bit
            q_int8 = (q - shift).reshape(-1, q.shape[-1]).to(torch.int8)

            flat_zero = zero.reshape(-1, zero.shape[-1])
            per_token_zero = flat_zero[:, 0].contiguous()  # [M]
            # Per-token correction: scale * (shift - zero)
            zp_correction = per_token_scale * (shift - per_token_zero)

        self.free()

        return q_int8, per_token_scale, zp_correction

    def configure(self, bits,
                  groupsize=-1,
                  sym=False,
                  clip_ratio=1.0,
                  residual=False,
                  static=False):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'
        self.residual = residual
        self.static = static

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)    # QuaRot 源码,不要加会导致推理时间X10
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)


class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module: torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        # Integer GEMM with capped accumulator
        self.use_int_gemm = False
        self.acc_bits = 32
        self.acc_block_k = 32
        self.acc_wrap = False
        self.int_gemm_use_triton = True
        self.register_buffer('w_int', None)
        self._buffers['w_int'] = None
        self.register_buffer('w_scale', None)
        self._buffers['w_scale'] = None
        self.register_buffer('w_zp_correction', None)
        self._buffers['w_zp_correction'] = None
        self.register_buffer('w_zp', None)
        self._buffers['w_zp'] = None
        self.register_buffer('w_zp_cross', None)
        self._buffers['w_zp_cross'] = None
        self.w_group_size = -1
        self.register_buffer('static_zp_bias', None)
        self._buffers['static_zp_bias'] = None
        # Calibration intermediates (set when _calibrating=True)
        self._calibrating = False
        self._cal_input = None     # post-rotation, pre-quantization
        self._cal_output = None    # post-matmul, pre-output-quantization

    def prepare_int_gemm(self, w_bits, w_sym=True, w_group_size=-1,
                         acc_bits=32, acc_block_k=32, use_triton=True,
                         acc_wrap=False):
        """Pre-compute integer weight representation for capped-accumulator GEMM."""
        from int_acc_gemm import prepare_int_weights
        self.use_int_gemm = True
        self.acc_bits = acc_bits
        self.acc_block_k = acc_block_k
        self.acc_wrap = acc_wrap
        self.int_gemm_use_triton = use_triton
        self.w_group_size = w_group_size

        w_int, w_scale, w_zp = prepare_int_weights(self.module, w_bits, w_sym, w_group_size)
        dev = self.module.weight.device
        self.w_int = w_int.to(dev)
        self.w_scale = w_scale.to(dev)
        self.w_zp = w_zp.to(dev) if w_zp is not None else None

        # Precompute per-output-channel zero-point correction for asymmetric
        # activations:  w_zp_correction[j] = Σ_k  s_w(j,k) · q_w(j,k)
        N, K = w_int.shape
        if w_group_size > 0:
            n_groups = w_scale.shape[1]
            padded_K = n_groups * w_group_size
            if padded_K > K:
                w_int_padded = torch.nn.functional.pad(w_int.float(), (0, padded_K - K))
            else:
                w_int_padded = w_int.float()
            # [N, n_groups, group_size] -> sum over group -> [N, n_groups]
            group_sums = w_int_padded.reshape(N, n_groups, w_group_size).sum(dim=2)
            self.w_zp_correction = (w_scale * group_sums).sum(dim=1)
        else:
            self.w_zp_correction = w_scale * w_int.float().sum(dim=1)

        # Precompute cross-term for dynamic activation + weight asymmetry:
        # w_zp_cross[j] = Σ_g w_zp[j,g] * w_scale[j,g] * G
        # Used at runtime as: output += a_zp_correction[:, None] * w_zp_cross[None, :]
        if w_zp is not None:
            if w_group_size > 0 and w_zp.dim() == 2:
                self.w_zp_cross = (w_zp * w_scale * w_group_size).sum(dim=1).to(dev)  # [N]
            else:
                self.w_zp_cross = (w_zp * w_scale * K).to(dev)  # [N]
        else:
            self.w_zp_cross = None

    def compute_static_zp_bias(self):
        """Precompute [N] zero-point correction for static asymmetric per-group mode.

        Must be called AFTER per-group scale conversion AND prepare_int_gemm.
        For asymmetric, the int8 shift creates a per-group bias that depends
        only on calibration-derived (scale, zero) and the fixed weight integers.
        This collapses to a constant [N] vector added to the output.
        """
        q = self.quantizer
        if q.sym or not q.static or q.groupsize <= 0:
            return
        if self.w_int is None:
            return

        dev = self.w_int.device
        G = q.groupsize
        group_scale = q.scale.to(dev)     # [n_groups]
        group_zero = q.zero.to(dev)       # [n_groups]
        maxq = q.maxq.to(dev)
        shift = (int(maxq.item()) + 1) // 2  # 128 for 8-bit

        # a_zp_corr[g] = group_scale[g] * (shift - group_zero[g])
        a_zp_corr = group_scale * (shift - group_zero)  # [n_groups]

        N, K = self.w_int.shape
        n_act_groups = K // G

        # Sum of weight integers per (output_channel, act_group): [N, n_act_groups]
        w_int_grouped = self.w_int.float().reshape(N, n_act_groups, G)
        w_int_group_sum = w_int_grouped.sum(dim=2)

        # Map each activation group to its weight group scale
        if self.w_group_size > 0:
            act_group_starts = torch.arange(n_act_groups, device=dev) * G
            w_group_idx = act_group_starts // self.w_group_size  # [n_act_groups]
            w_scale_per_ag = self.w_scale[:, w_group_idx]  # [N, n_act_groups]
        else:
            w_scale_per_ag = self.w_scale.unsqueeze(1).expand(N, n_act_groups)

        # static_zp_bias[j] = Σ_g a_zp_corr[g] * w_scale[j,g] * w_int_group_sum[j,g]
        weighted = w_scale_per_ag * w_int_group_sum  # [N, n_act_groups]
        self.static_zp_bias = (a_zp_corr.unsqueeze(0) * weighted).sum(dim=1)  # [N]

        # Cross-term for combined static activation + weight asymmetry:
        # Σ_g a_zp_corr[act_g] * w_zp[j, w_g] * w_scale[j, w_g] * G
        if self.w_zp is not None:
            if self.w_group_size > 0 and self.w_zp.dim() == 2:
                # Map each activation group to its weight group
                w_zp_per_ag = self.w_zp[:, w_group_idx]    # [N, n_act_groups]
                cross = w_zp_per_ag * w_scale_per_ag * G   # [N, n_act_groups]
                self.static_zp_bias += (a_zp_corr.unsqueeze(0) * cross).sum(dim=1)
            else:
                # Per-channel: w_zp is [N], each act group contributes G elements
                w_zp_dev = self.w_zp.to(dev)
                w_scale_dev = self.w_scale.to(dev) if self.w_scale.dim() == 1 else self.w_scale[:, 0].to(dev)
                cross_per_ag = w_zp_dev.unsqueeze(1) * w_scale_dev.unsqueeze(1) * G  # [N, 1]
                self.static_zp_bias += (a_zp_corr.unsqueeze(0) * cross_per_ag).sum(dim=1)

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        if self.use_int_gemm:
            overflow = 'wrap' if self.acc_wrap else 'saturate'
            str_ += f'\nInt GEMM: acc_bits={self.acc_bits}, block_k={self.acc_block_k}, overflow={overflow}'

        return str_

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:

            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                                                               scale=1 / math.sqrt(init_shape[-1] // self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1,
                     init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self._calibrating:
            self._cal_input = x  # post-rotation, pre-quantization

        if self.use_int_gemm and self.quantizer.bits <= 8:
            # Integer GEMM path: quantize activations to int inside the kernel
            from int_acc_gemm import int_gemm_capped
            _ig_kwargs = dict(
                w_int=self.w_int, w_scale=self.w_scale,
                act_quantizer=self.quantizer, acc_bits=self.acc_bits,
                block_k=self.acc_block_k, w_group_size=self.w_group_size,
                bias=self.bias, use_triton=self.int_gemm_use_triton,
                acc_wrap=self.acc_wrap, w_zp_correction=self.w_zp_correction,
                w_zp=self.w_zp, w_zp_cross=self.w_zp_cross,
            )
            x_pre = x  # save pre-quantization input for sd_check
            x = int_gemm_capped(x_float=x, **_ig_kwargs).to(x_dtype)

            # Static asymmetric per-group zero-point correction (precomputed bias)
            if self.quantizer.static and self.static_zp_bias is not None:
                x = x + self.static_zp_bias.to(device=x.device, dtype=x.dtype)

            # sd_check: compare static int_gemm vs dynamic int_gemm
            q = self.quantizer
            if q._sd_check > 0 and q.static and not q._in_sd_check:
                saved = (q.static, q.groupsize,
                         q.scale.clone(),
                         q.zero.clone() if q.zero is not None else None)
                q.static = False
                q.groupsize = -1
                q._in_sd_check = True
                x_dyn = int_gemm_capped(x_float=x_pre, **_ig_kwargs).to(x_dtype)
                q._in_sd_check = False
                q.static, q.groupsize = saved[0], saved[1]
                q.scale = saved[2]
                if saved[3] is not None:
                    q.zero = saved[3]
                q._sd_report(x, x_dyn)
        else:
            # Original fake-quant path
            if self.quantizer.bits < 16:  # Quantize, if needed
                # self.quantizer.find_params(x)  # QuaRot 源码，现修改在 quantizer.forward 函数中
                x = self.quantizer(x).to(x_dtype)
                self.quantizer.free()

            x = self.module(x).to(x_dtype)

        if self._calibrating:
            self._cal_output = x  # post-matmul, pre-output-quantization

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            # self.out_quantizer.find_params(x) # QuaRot 源码，现修改在 quantizer.forward 函数中
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        groupsize=-1, static_groups=False,
        gscaler=None
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.gscaler = gscaler
        if sym:
            self.maxq = torch.tensor(2**(bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

        self.groupsize = groupsize
        self.static_groups = static_groups

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin.masked_fill(tmp, -1)
            xmax.masked_fill(tmp, +1)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1),
                                           zero1.unsqueeze(1), self.maxq)

                q = (q - x).abs().pow(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                best = torch.where(tmp, err, best)
                self.scale = torch.where(tmp, scale1, self.scale)
                self.zero = torch.where(tmp, zero1, self.zero)

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if self.gscaler is not None:
            self.scale = snap_scale_to_gscaler(self.scale, self.gscaler)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x, stochastic=False):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq,
                                         stochastic=stochastic).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero,
                                      self.maxq, stochastic=stochastic).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)


def find_qlayers(module, layers=[torch.nn.Linear,
                                 ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
