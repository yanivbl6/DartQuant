import math
import re
import transformers
import torch
import utils
import hadamard_utils
import fast_hadamard_transform


# ── Mantissa+Shift group-scale quantization ──────────────────────────────────

# Regex accepts either an integer bias (b/l<N>) or the literals 'min' / 'mid'
# for hwscale's per-layer auto-bias modes.
_GSCALER_RE = re.compile(r'^M(\d+)[SE](\d+)(?:([bl])?(\d+|min|mid|opt|best))?$')


def parse_gscaler(spec):
    """Parse a gscaler / hwscale spec string into a config dict.

    Examples:
        'M5S3'     -> {mantissa_bits: 5, shift_bits: 3, bias: 0}
        'M6E4b2'   -> {mantissa_bits: 6, shift_bits: 4, bias: 2}
        'M6S4l2'   -> {mantissa_bits: 6, shift_bits: 4, bias: -2}
        'M4S3bmin' -> {... bias_auto: 'min'}  (per-layer, anchor at min_nonzero)
        'M4S3bmid' -> {... bias_auto: 'mid'}  (per-layer, anchor at median_abs)
        'M6S3bopt' -> {... bias_auto: 'opt'}  (per-layer, MSE-optimal global
                                                via brute-force search)
        'M4S4best' -> {... bias_auto: 'best'} (per-layer, MSE-optimal global
                                                including a fractional mantissa
                                                — non-power-of-2 globals)

    Returns None if spec is None.
    Raises ValueError on invalid format.
    """
    if spec is None:
        return None
    m = _GSCALER_RE.match(spec)
    if not m:
        raise ValueError(
            f"Invalid --gscaler / --hwscale format: '{spec}'. "
            f"Expected e.g. M5S3, M6E4b2, M6S4l2, M4S3bmin, M4S3bmid, "
            f"M6S3bopt, M4S4best.")
    mantissa_bits = int(m.group(1))
    shift_bits = int(m.group(2))
    bias_dir = m.group(3)  # 'b', 'l', or None
    bias_raw = m.group(4)
    bias_auto = bias_raw if bias_raw in ('min', 'mid', 'opt', 'best') else None
    if bias_auto is not None:
        bias_val = None  # computed per-layer at merge time
    else:
        bias_val = int(bias_raw) if bias_raw else 0
        if bias_dir == 'l':
            bias_val = -bias_val
    return {
        'mantissa_bits': mantissa_bits,
        'shift_bits': shift_bits,
        'bias': bias_val,
        'bias_auto': bias_auto,
        'bias_dir': bias_dir,
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


def hwscale_global_from_spec(spec, raw_scale=None):
    """Derive the per-layer FP32 global scale from a parsed hwscale spec.

    MxSy (no bias suffix)          -> global = 1.0
    MxSybz (positive bias)          -> global = 2^-z  (stored scales are larger)
    MxSylz (negative bias via 'l')  -> global = 2^+z
    MxSybmin (auto, anchor at min)  -> z = -floor(log2(min_nonzero(raw_scale)/2))
                                       so 2*global < every nonzero scale
                                       (i.e. stored_min ≥ 2, inside MxSy).
    MxSybmid (auto, anchor at mid)  -> center the grid on log2(median(|raw|)).
                                       g = floor(log2(median_abs)) - 1 + 2^(S-1),
                                       z = -g.  Floor rounds toward "downward"
                                       side of the asymmetric grid to reduce
                                       clipping on that side.
    MxSybopt (auto, MSE-optimal)    -> brute-force search z in [z_mid-4, z_mid+4]
                                       picking the global with min squared
                                       snap error on raw_scale.  z_mid is the
                                       same starting point as the 'mid' branch.
    MxSybest (auto, MSE-optimal,    -> like 'opt' but searches non-power-of-2
              non-power-of-2)         globals: global = m_frac * 2^(-z), with
                                       m_frac in {1, 17/16, ..., 31/16}.
                                       Returns FP32 scalar (may not be 2^-k).
    """
    if spec is None:
        return 1.0
    mode = spec.get('bias_auto')
    if mode:
        assert raw_scale is not None, \
            "hwscale auto-bias requires raw_scale tensor for per-layer derivation"
        if mode == 'min':
            nonzero = raw_scale[raw_scale > 0]
            if nonzero.numel() == 0:
                z = 0
            else:
                min_nz = nonzero.min().item()
                # floor(log2(min_nz / 2)) is strictly below log2(min_nz), so
                # 2 * global = 2^(1-z) < min_nz for every nonzero entry (and
                # stored_min = min_nz / global ≥ 2, inside MxSy's grid).
                z = max(0, int(-math.floor(math.log2(min_nz / 2.0))))
        elif mode == 'mid':
            abs_nz = raw_scale.abs()
            abs_nz = abs_nz[abs_nz > 0]
            if abs_nz.numel() == 0:
                z = 0
            else:
                median_abs = abs_nz.median().item()
                # Floor log2(median_abs) to push the effective grid center
                # slightly below the true median: the MxSy grid extends more
                # upward than downward from its mantissa=2/shift=2^(S-1)
                # center, so flooring compensates by adding headroom below.
                median_log = int(math.floor(math.log2(median_abs)))
                S = spec['shift_bits']
                # Median shift index is 2^(S-1); degenerates to 0 when S=0
                # (only one shift, so the grid is a pure integer mantissa grid).
                half_shift = (1 << (S - 1)) if S >= 1 else 0
                g = median_log - 1 + half_shift
                z = -g
        elif mode == 'opt':
            # MSE-optimal: search z over [z_mid-R, z_mid+R], snap, pick min MSE.
            # Vectorized across the 9 candidate z values: build [n_z, N] tensor
            # and run snap once.
            abs_nz = raw_scale.abs()
            abs_nz = abs_nz[abs_nz > 0]
            if abs_nz.numel() == 0:
                z = 0
            else:
                # Subsample for speed: 200k matches full-tensor optimum in tests.
                _MAX = 200_000
                if abs_nz.numel() > _MAX:
                    gen = torch.Generator(device='cpu').manual_seed(0)
                    idx = torch.randperm(abs_nz.numel(), generator=gen)[:_MAX]
                    abs_nz_sub = abs_nz[idx.to(abs_nz.device)]
                else:
                    abs_nz_sub = abs_nz
                # Center of search: same as 'mid' branch.
                median_abs = abs_nz_sub.median().item()
                median_log = int(math.floor(math.log2(median_abs)))
                S = spec['shift_bits']
                half_shift = (1 << (S - 1)) if S >= 1 else 0
                z_mid = -(median_log - 1 + half_shift)
                # bias=0 spec for the inner snap (global is what we're searching).
                bias0_spec = {
                    'mantissa_bits': spec['mantissa_bits'],
                    'shift_bits': spec['shift_bits'],
                    'bias': 0,
                    'spec': spec['spec'],
                }
                R = 4
                z_candidates = torch.arange(
                    z_mid - R, z_mid + R + 1, device=abs_nz_sub.device, dtype=torch.float32)
                g_try = torch.pow(2.0, -z_candidates)              # [n_z]
                scaled = abs_nz_sub.unsqueeze(0) / g_try.unsqueeze(1)  # [n_z, N]
                snapped = snap_scale_to_gscaler(scaled, bias0_spec)   # [n_z, N]
                recovered = snapped * g_try.unsqueeze(1)              # [n_z, N]
                mse = ((recovered - abs_nz_sub.unsqueeze(0)) ** 2).sum(dim=1)  # [n_z]
                z = int(z_candidates[int(mse.argmin().item())].item())
        elif mode == 'best':
            # MSE-optimal across non-power-of-2 globals: search both z and a
            # 4-bit fractional mantissa. global = m_frac * 2^(-z), m_frac in
            # [1, 2). Vectorized across all 9*16=144 candidates in one snap.
            abs_nz = raw_scale.abs()
            abs_nz = abs_nz[abs_nz > 0]
            if abs_nz.numel() == 0:
                global_scalar = 1.0
            else:
                _MAX = 100_000
                if abs_nz.numel() > _MAX:
                    gen = torch.Generator(device='cpu').manual_seed(0)
                    idx = torch.randperm(abs_nz.numel(), generator=gen)[:_MAX]
                    abs_nz_sub = abs_nz[idx.to(abs_nz.device)]
                else:
                    abs_nz_sub = abs_nz
                median_abs = abs_nz_sub.median().item()
                median_log = int(math.floor(math.log2(median_abs)))
                S = spec['shift_bits']
                half_shift = (1 << (S - 1)) if S >= 1 else 0
                z_mid = -(median_log - 1 + half_shift)
                bias0_spec = {
                    'mantissa_bits': spec['mantissa_bits'],
                    'shift_bits': spec['shift_bits'],
                    'bias': 0,
                    'spec': spec['spec'],
                }
                R = 4
                F_BITS = 4  # 16 fractional-mantissa values, matches fp8
                F = 1 << F_BITS
                dev = abs_nz_sub.device
                z_grid = torch.arange(
                    z_mid - R, z_mid + R + 1, device=dev, dtype=torch.float32)
                # frac_grid in [1, 2), uniformly spaced
                frac_grid = 1.0 + torch.arange(F, device=dev, dtype=torch.float32) / F
                global_grid = (frac_grid.unsqueeze(0) * torch.pow(2.0, -z_grid).unsqueeze(1)
                              ).flatten()                         # [n_cand]
                scaled = abs_nz_sub.unsqueeze(0) / global_grid.unsqueeze(1)  # [n_cand, N]
                snapped = snap_scale_to_gscaler(scaled, bias0_spec)          # [n_cand, N]
                recovered = snapped * global_grid.unsqueeze(1)               # [n_cand, N]
                mse = ((recovered - abs_nz_sub.unsqueeze(0)) ** 2).sum(dim=1)  # [n_cand]
                global_scalar = float(global_grid[int(mse.argmin().item())].item())
            if spec.get('bias_dir') == 'l':
                global_scalar = 1.0 / global_scalar
            return global_scalar
        else:
            raise ValueError(f"Unknown hwscale auto-bias mode: '{mode}'")
        if spec.get('bias_dir') == 'l':
            z = -z
        return float(2.0 ** (-z))
    return float(2.0 ** (-spec['bias']))


def snap_to_hwscale(raw_scale, spec):
    """Snap merged per-group scale tensor to MxSy + separate global FP32 scale.

    Given the raw merged scale `raw_scale = a_gscale * w_gscale`, returns
    `(stored, global_scalar)` such that `stored * global_scalar ≈ raw_scale`,
    where `stored` is representable in MxSy (bias=0) form and `global_scalar`
    is an FP32 multiplier applied once at T2.

    For auto-bias specs (bmin/lmin), `global_scalar` is derived per-layer
    from the minimum nonzero entry in `raw_scale`.

    Parameters
    ----------
    raw_scale : positive float tensor
    spec      : dict from `parse_gscaler()`
    """
    global_scalar = hwscale_global_from_spec(spec, raw_scale=raw_scale)
    # Shift the raw values into the MxSy (bias=0) representable range, snap,
    # leaving the residual `global_scalar` to be applied at T2.
    bias0_spec = {
        'mantissa_bits': spec['mantissa_bits'],
        'shift_bits': spec['shift_bits'],
        'bias': 0,
        'spec': spec['spec'],
    }
    scaled = raw_scale / global_scalar
    stored = snap_scale_to_gscaler(scaled, bias0_spec)
    return stored, global_scalar


def get_minq_maxq(bits, sym):
    # Return float32 tensors to prevent fp16 promotion overflow.
    # int64 maxq (e.g. 262143 for 18-bit) would silently become inf
    # when PyTorch promotes it to fp16 during mixed-type arithmetic.
    if sym:
        maxq = torch.tensor(2**(bits - 1) - 1, dtype=torch.float32)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1, dtype=torch.float32)
        minq = torch.tensor(0, dtype=torch.float32)

    return minq, maxq


def _stochastic_round(x):
    """Round with probability proportional to fractional part (unbiased)."""
    floor = x.floor()
    return floor + (torch.rand_like(x) < (x - floor)).to(x.dtype)


def asym_quant(x, scale, zero, maxq, stochastic=False):
    assert maxq != float('inf'), "maxq is inf — likely fp16 overflow"
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


# FP4 (NVFP4) integer level set: 16 codes ordered ascending.
# NVFP4 levels are {0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}; multiplying
# by 2 gives the integer code set below, and the per-group scale becomes amax/12
# so dequant value (code * scale) matches the original NVFP4 dequant value.
_FP4_CODES = torch.tensor(
    [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0,
     1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    dtype=torch.float32,
)


def fp4_quant_dequant(x, scale):
    """Snap x/scale to the nearest FP4 integer code, return code * scale.

    Uses bucketize on the midpoints between adjacent codes — exact nearest-
    neighbour mapping with no off-by-one at the level boundaries.
    """
    codes = _FP4_CODES.to(device=x.device, dtype=x.dtype)
    scale = scale.to(x.device)
    x_norm = x / scale.clamp(min=1e-10)
    mids = (codes[:-1] + codes[1:]) / 2.0
    idx = torch.bucketize(x_norm.contiguous(), mids)
    return codes[idx] * scale


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
        self.realint = False
        self.groupsize = -1
        self.sym = False
        self.clip_ratio = 1.0
        self.residual = False
        self.stochastic = False
        self.static = False
        # Link to upstream out_quantizer for quantizer→quantizer chain detection
        self.shared_scale_source = None
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
            # Diagnostic: show scale/zero/input stats
            rs = result_static.float()
            rd = result_dynamic.float()
            print(f"  static  out: min={rs.min().item():.4e} max={rs.max().item():.4e} mean={rs.mean().item():.4e}", flush=True)
            print(f"  dynamic out: min={rd.min().item():.4e} max={rd.max().item():.4e} mean={rd.mean().item():.4e}", flush=True)
            print(f"  diff:        min={diff.min().item():.4e} max={diff.max().item():.4e} abs_max={diff.abs().max().item():.4e}", flush=True)
            if self.scale is not None:
                s = self.scale.float()
                print(f"  static scale: shape={list(s.shape)} min={s.min().item():.4e} max={s.max().item():.4e} mean={s.mean().item():.4e}", flush=True)
            if self.zero is not None:
                z = self.zero.float()
                print(f"  static zero:  shape={list(z.shape)} min={z.min().item():.4e} max={z.max().item():.4e} mean={z.mean().item():.4e}", flush=True)
            print(f"  bits={self.bits} maxq={self.maxq}", flush=True)

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

        if self.bits == 16 and not self.realint:
            return x

        if self.realint:
            assert self.maxq != float('inf'), \
                f"maxq overflowed (bits={self.bits}); likely fp16 promotion bug"

        if not self.static:
            self.find_params(x)  # dynamic: recompute every forward
        elif self.shared_scale_source is not None:
            # Quantizer→quantizer chain: use upstream scale to avoid double quantization loss
            self.scale = self.shared_scale_source.scale.to(x.device)
            if not self.sym and not self.shared_scale_source.sym:
                self.zero = self.shared_scale_source.zero.to(x.device)
        # else: use pre-loaded self.scale / self.zero

        if self.residual:
            if self.sym:
                tmp_q = sym_quant_dequant(x, self.scale, self.maxq, stochastic=self.stochastic).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = sym_quant_dequant(x - tmp_q, self.scale, self.maxq, stochastic=self.stochastic).to(x_dtype)
                result = tmp_q + residual_q
            else:
                tmp_q = asym_quant_dequant(x, self.scale, self.zero, self.maxq, stochastic=self.stochastic).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = asym_quant_dequant(x - tmp_q, self.scale, self.zero, self.maxq, stochastic=self.stochastic).to(x_dtype)
                result = tmp_q + residual_q
        elif self.sym:
            result = sym_quant_dequant(x, self.scale, self.maxq, stochastic=self.stochastic).to(x_dtype)
        else:
            result = asym_quant_dequant(x, self.scale, self.zero, self.maxq, stochastic=self.stochastic).to(x_dtype)

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
        """Quantize activations to integer and return (q_int, scale_vec, zp_correction).

        Unlike ``forward`` (fake-quant) or ``quantize`` (returns float integers),
        this method returns *actual* int8/int16 values and a 1-D scale vector,
        suitable for feeding into the integer GEMM kernel.

        For bits <= 8, returns int8.  For 9 <= bits <= 16, returns int16.

        Static mode with groupsize > 0: per-group scales along K, aligned with
        acc_block_k.  Each group of G columns shares one scale.  Returns
        ``(q_int[M,K], group_scales[n_groups], None)``.

        Static mode with groupsize <= 0: per-tensor scale (max of all column
        scales).  Returns ``(q_int[M,K], per_token_scale[M], zp_correction)``.

        Dynamic mode: computes per-token scales from x at runtime.
        Returns ``(q_int[M,K], per_token_scale[M], zp_correction)``.

        For asymmetric quantization, q is shifted into signed range and a
        zero-point correction factor is returned.  The caller must combine it
        with the precomputed ``w_zp_correction`` vector:

            output += zp_correction[:, None] * w_zp_correction[None, :]

        Requires bits <= 16.
        """
        assert self.bits <= 16, \
            f"quantize_to_int requires bits <= 16 (got {self.bits}); values would overflow int16"
        _int_dtype = torch.int16 if self.bits > 8 else torch.int8

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
                q_int = q_grouped.reshape(M, K).to(_int_dtype)
            else:
                # Asymmetric: quantize to [0, maxq], shift to signed range
                q_grouped = torch.clamp(
                    torch.round(x_grouped / group_scale[None, :, None])
                    + group_zero[None, :, None],
                    0, self.maxq)
                shift = (int(self.maxq.item()) + 1) // 2
                q_int = (q_grouped - shift).reshape(M, K).to(_int_dtype)

            # zp_correction is None: for per-group static asymmetric, the
            # correction is precomputed as static_zp_bias in ActQuantWrapper.
            return q_int, group_scale, None

        if self.static:
            # Per-tensor static scale (groupsize <= 0 fallback).
            # Collapse per-column scales to a single scalar = max over columns.
            col_scale = self.scale.to(dev).flatten()

            if self.sym:
                per_tensor_scale = col_scale.max()
                q = torch.clamp(torch.round(x / per_tensor_scale),
                                -(self.maxq + 1), self.maxq)
                q_int = q.reshape(-1, q.shape[-1]).to(_int_dtype)
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
                q_int = (q - shift).reshape(-1, q.shape[-1]).to(_int_dtype)
                per_token_scale = torch.full((M,), per_tensor_scale.item(),
                                             device=dev, dtype=col_scale.dtype)
                zp_corr_scalar = per_tensor_scale * (shift - global_zero)
                zp_correction = torch.full((M,), zp_corr_scalar.item(),
                                           device=dev, dtype=col_scale.dtype)

            return q_int, per_token_scale, zp_correction

        # Dynamic: compute per-token scales from x at runtime.
        assert self.groupsize <= 0, \
            "quantize_to_int with groupsize > 0 requires static mode"
        # find_params skips bits=16 without realint (fake-quant no-op), but
        # quantize_to_int always needs actual scales.  Force realint temporarily.
        _saved_realint = self.realint
        self.realint = True
        self.find_params(x)
        self.realint = _saved_realint

        # Extract per-token scalar scale (works for both sym and asym).
        # self.scale has shape [..., K] with repeated values along K.
        # IMPORTANT: flat_scale[:, 0] has stride K (non-contiguous).  The
        # Triton kernel indexes A_scale_ptr with stride 1, so we MUST return
        # a contiguous 1-D tensor — otherwise the kernel reads garbage.
        flat_scale = self.scale.reshape(-1, self.scale.shape[-1])
        per_token_scale = flat_scale[:, 0].contiguous()  # [M], stride-1

        if self.sym:
            q, _scale = sym_quant(x, self.scale, self.maxq)
            q_int = q.reshape(-1, q.shape[-1]).to(_int_dtype)
            zp_correction = None
        else:
            q, _scale, zero = asym_quant(x, self.scale, self.zero, self.maxq)
            # q is in [0, maxq]. Shift into signed range.
            shift = (int(self.maxq.item()) + 1) // 2
            q_int = (q - shift).reshape(-1, q.shape[-1]).to(_int_dtype)

            flat_zero = zero.reshape(-1, zero.shape[-1])
            per_token_zero = flat_zero[:, 0].contiguous()  # [M]
            # Per-token correction: scale * (shift - zero)
            zp_correction = per_token_scale * (shift - per_token_zero)

        self.free()

        return q_int, per_token_scale, zp_correction

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
        if self.bits == 16 and not self.realint:
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
        self.pre_quantizer = ActQuantizer()  # quantize before rotation (e.g. before R4)
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        # Integer GEMM with capped accumulator
        self.use_int_gemm = False
        self._ig_compare = False  # compare int_gemm vs float GEMM
        # Semi-int GEMM diagnostic kernel
        self.use_semi_int_gemm = False
        self.semi_int_mask = '111111'
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
        # Equalization: per-channel factors for down_proj (set by --eq)
        self.eq_factors = None
        # Calibration intermediates (set when _calibrating=True)
        self._calibrating = False
        self._cal_input = None     # post-rotation, pre-quantization
        self._cal_output = None    # post-matmul, pre-output-quantization
        # R4 stats collector (set by r4_stats.setup_r4_stats())
        self._r4_collector = None

    def prepare_int_gemm(self, w_bits, w_sym=True, w_group_size=-1,
                         acc_bits=32, acc_block_k=32, use_triton=True,
                         acc_wrap=False, acc_dtype='float',
                         gscaler_parsed=None, lsb_mac_shift=0,
                         t1_msb_scan=False):
        """Pre-compute integer weight representation for capped-accumulator GEMM."""
        from int_acc_gemm import prepare_int_weights
        self.use_int_gemm = True
        self.t1_msb_scan = t1_msb_scan
        self.acc_bits = acc_bits
        self.acc_block_k = acc_block_k
        self.acc_wrap = acc_wrap
        self.acc_dtype = acc_dtype
        self.int_gemm_use_triton = use_triton
        self.w_group_size = w_group_size
        self.lsb_mac_shift = lsb_mac_shift

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

        # Factor out gscaler shift bias: prescale w_scale by 2^bias so the
        # kernel accumulates larger values in tier-2.  The bias is applied
        # back (as 2^-bias) after the K-loop.  Corrections above use the
        # original w_scale, so this must come last.
        # Only prescale for integer tier-2 — float types handle the small
        # scales natively, and fp16 would overflow with large prescaling.
        # Note: fixed-point frac_bits (int32p4) are handled separately in the
        # kernel — they scale the contribution by 2^frac inside the K-loop
        # and undo via integer right-shift after the loop, BEFORE the gscaler
        # float multiply.  They are NOT folded into w_scale prescaling.
        if (gscaler_parsed is not None and gscaler_parsed['bias'] >= 1
                and acc_dtype.startswith('int')):
            self.w_shift_bias = gscaler_parsed['bias']
            self.w_scale = self.w_scale * (2.0 ** self.w_shift_bias)
        else:
            self.w_shift_bias = 0

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

        # Map each activation group to its weight group scale.
        # Use un-prescaled w_scale (undo gscaler shift bias) because this
        # correction is applied in float outside the kernel.
        w_scale_orig = self.w_scale
        w_shift = getattr(self, 'w_shift_bias', 0)
        if w_shift > 0:
            w_scale_orig = w_scale_orig * (2.0 ** (-w_shift))
        if self.w_group_size > 0:
            act_group_starts = torch.arange(n_act_groups, device=dev) * G
            w_group_idx = act_group_starts // self.w_group_size  # [n_act_groups]
            w_scale_per_ag = w_scale_orig[:, w_group_idx]  # [N, n_act_groups]
        else:
            w_scale_per_ag = w_scale_orig.unsqueeze(1).expand(N, n_act_groups)

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
                w_scale_dev = w_scale_orig.to(dev) if w_scale_orig.dim() == 1 else w_scale_orig[:, 0].to(dev)
                cross_per_ag = w_zp_dev.unsqueeze(1) * w_scale_dev.unsqueeze(1) * G  # [N, 1]
                self.static_zp_bias += (a_zp_corr.unsqueeze(0) * cross_per_ag).sum(dim=1)

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16 or self.quantizer.realint:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16 or self.out_quantizer.realint:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        if self.use_int_gemm:
            overflow = 'wrap' if self.acc_wrap else 'saturate'
            str_ += f'\nInt GEMM: acc_bits={self.acc_bits}, block_k={self.acc_block_k}, overflow={overflow}'

        return str_

    def forward(self, x):
        x_dtype = x.dtype
        _r4c = self._r4_collector

        # Point A: raw input (before pre_quantizer/rotation)
        if _r4c is not None and _r4c.active:
            _r4c.point_a.update(x)

        # Pre-rotation quantization (e.g. quantize input before R4 Hadamard)
        if self.pre_quantizer.bits < 16 or self.pre_quantizer.realint:
            x = self.pre_quantizer(x).to(x_dtype)
            self.pre_quantizer.free()

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

        # Equalization: per-channel scaling (after rotation, before quantization)
        if self.eq_factors is not None:
            x = x / self.eq_factors.to(device=x.device, dtype=x.dtype)

        # Point B: after rotation/equalization, before input quantizer
        if _r4c is not None and _r4c.active:
            _r4c.point_b.update(x)
            _x_pre_quant = x

        if self._calibrating:
            self._cal_input = x  # post-rotation/eq, pre-quantization

        if self.use_int_gemm and self.quantizer.bits <= 16:
            # Integer GEMM path: quantize activations to int8/int16 inside the kernel
            from int_acc_gemm import int_gemm_capped
            _ig_kwargs = dict(
                w_int=self.w_int, w_scale=self.w_scale,
                act_quantizer=self.quantizer, acc_bits=self.acc_bits,
                block_k=self.acc_block_k, w_group_size=self.w_group_size,
                bias=self.bias, use_triton=self.int_gemm_use_triton,
                acc_wrap=self.acc_wrap, w_zp_correction=self.w_zp_correction,
                w_zp=self.w_zp, w_zp_cross=self.w_zp_cross,
                acc_dtype=getattr(self, 'acc_dtype', 'float'),
                w_shift_bias=getattr(self, 'w_shift_bias', 0),
                lsb_mac_shift=getattr(self, 'lsb_mac_shift', 0),
                global_scale_fp32=float(getattr(self, 'hwscale_global', 1.0)),
            )
            if getattr(self, 't1_msb_scan', False):
                _ig_kwargs['_t1_scan_label'] = getattr(self.quantizer, '_sd_name', '??')
            x_pre = x  # save pre-quantization input for sd_check
            x = int_gemm_capped(x_float=x, **_ig_kwargs)

            # Static asymmetric per-group zero-point correction (precomputed bias)
            # Must be added BEFORE bf16 cast to avoid catastrophic cancellation
            if self.quantizer.static and self.static_zp_bias is not None:
                x = x + self.static_zp_bias.to(device=x.device, dtype=x.dtype)
            x = x.to(x_dtype)

            # t1_msb_scan: detect tier-1 accumulator overflow
            if getattr(self, 't1_msb_scan', False):
                _ref_kwargs = dict(_ig_kwargs)
                _ref_kwargs['acc_bits'] = 32
                _ref_kwargs['acc_wrap'] = False
                _ref_kwargs['lsb_mac_shift'] = 0
                _ref_kwargs['_t1_scan_label'] = ''  # no sub-scan on reference call
                x_ref = int_gemm_capped(x_float=x_pre, **_ref_kwargs)
                if self.quantizer.static and self.static_zp_bias is not None:
                    x_ref = x_ref + self.static_zp_bias.to(device=x_ref.device, dtype=x_ref.dtype)
                x_ref = x_ref.to(x_dtype)
                diff = (x.float() - x_ref.float())
                rms_ref = (x_ref.float() ** 2).mean().sqrt().item() + 1e-12
                rmse = (diff ** 2).mean().sqrt().item()
                nrmse = rmse / rms_ref
                max_abs = diff.abs().max().item()
                layer_name = getattr(self.quantizer, '_sd_name', '??')
                if max_abs > 1e-6:
                    print(f"[t1_msb_scan] MISMATCH {layer_name}: "
                          f"NRMSE={nrmse:.4e}, max_abs_diff={max_abs:.4e}, "
                          f"rms_ref={rms_ref:.4e}, n_diff={int((diff.abs() > 1e-6).sum().item())}",
                          flush=True)
                    if nrmse > 1e-4:
                        raise RuntimeError(
                            f"[t1_msb_scan] Tier-1 overflow in {layer_name} "
                            f"(NRMSE={nrmse:.4e})")

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

            # ig_compare: compare int_gemm vs normal fake-quant GEMM
            if self._ig_compare:
                q = self.quantizer
                # Only switch to dynamic when static scales aren't broadcast-
                # compatible with x [..., K] — specifically, per-group scales
                # ([n_groups], hw_align) don't broadcast.  Per-tensor [1]
                # (hw_accurate) and per-column [K] (down_proj path) do broadcast
                # and should be used as-is for a true apples-to-apples comparison
                # (same quantization as int_gemm, only GEMM path differs).
                needs_dynamic = q.static and q.groupsize > 0
                if needs_dynamic:
                    saved = (q.static, q.groupsize,
                             q.scale.clone() if q.scale is not None else None,
                             q.zero.clone() if q.zero is not None else None)
                    q.static = False
                    q.groupsize = -1
                x_fq = x_pre.clone()
                if q.bits < 16 or q.realint:
                    x_fq = q(x_fq).to(x_dtype)
                    q.free()
                x_fq = self.module(x_fq).to(x_dtype)
                # Restore quantizer state (no-op when we didn't switch)
                if needs_dynamic:
                    q.static, q.groupsize = saved[0], saved[1]
                    q.scale = saved[2]
                    if saved[3] is not None:
                        q.zero = saved[3]

                # Check error and report midpoint for problematic layers
                if not getattr(self, '_ig_logged', False):
                    self._ig_logged = True
                    ref_norm = x_fq.float().abs().max().item()
                    full_err = (x.float() - x_fq.float()).abs().max().item() / ref_norm if ref_norm > 0 else 0

                    if full_err > 0.1 and self.w_zp is not None and self.w_zp.numel() > 0:
                        wzp = self.w_zp.float()
                        # w_int is centered: range [-midpoint, maxq-midpoint]
                        # w_zp = midpoint - gptq_zero, so gptq_zero = midpoint - w_zp
                        midpoint = int(-self.w_int.min().item())  # e.g. 8 for 4-bit
                        gptq_zero = midpoint - wzp
                        print(f"[ig_cmp] {q._sd_name}: err={full_err:.2e}  "
                              f"midpoint={midpoint}  "
                              f"gptq_zero  mean={gptq_zero.mean().item():.2f}  "
                              f"range=[{gptq_zero.min().item():.1f}, {gptq_zero.max().item():.1f}]  "
                              f"w_zp  abs_mean={wzp.abs().mean().item():.2f}  "
                              f"max={wzp.abs().max().item():.1f}", flush=True)
                    elif full_err > 0.1:
                        print(f"[ig_cmp] {q._sd_name}: err={full_err:.2e}  (no w_zp)", flush=True)

                x = x_fq  # use float path for correct PPL
        elif self.use_semi_int_gemm:
            # Semi-int GEMM diagnostic path
            from semi_int_gemm import semi_int_gemm
            x = semi_int_gemm(
                x, self.w_int, self.w_scale,
                self.quantizer, self.semi_int_mask,
                acc_bits=self.acc_bits, block_k=self.acc_block_k,
                w_group_size=self.w_group_size, bias=self.bias,
                acc_wrap=self.acc_wrap, w_zp=self.w_zp,
                acc_dtype=getattr(self, 'acc_dtype', 'float'),
                w_shift_bias=getattr(self, 'w_shift_bias', 0),
                w_zp_correction=self.w_zp_correction,
                w_zp_cross=self.w_zp_cross,
                module_weight=self.module.weight.data,
                module=self.module,
            )
            # Static asymmetric per-group zero-point correction (precomputed bias)
            # Must be added BEFORE bf16 cast to avoid catastrophic cancellation.
            # Skip for int8 layers where the in-kernel a_zp correction handles this.
            _skip_static_zp = (self.quantizer.static
                               and getattr(self.quantizer, 'groupsize', -1) > 0
                               and not self.quantizer.sym
                               and self.quantizer.bits <= 8)
            if self.quantizer.static and self.static_zp_bias is not None and not _skip_static_zp:
                x = x + self.static_zp_bias.to(device=x.device, dtype=x.dtype)
            x = x.to(x_dtype)
        else:
            # Original fake-quant path
            if self.quantizer.bits < 16 or self.quantizer.realint:  # Quantize, if needed
                # self.quantizer.find_params(x)  # QuaRot 源码，现修改在 quantizer.forward 函数中
                x = self.quantizer(x).to(x_dtype)
                self.quantizer.free()

            # Point C: after input quantization, before linear
            if _r4c is not None and _r4c.active:
                _r4c.point_c.update(x)
                _r4c.cross_bc.update(_x_pre_quant, x)

            x = self.module(x).to(x_dtype)

        # Point D: output of linear layer + end-to-end SQNR
        if _r4c is not None and _r4c.active:
            _r4c.point_d.update(x)
            # End-to-end SQNR: compare quantized output vs unquantized reference
            with torch.no_grad():
                _x_ref = self.module(_x_pre_quant).to(x_dtype)
            _r4c.cross_ref.update(_x_ref, x)
            _r4c.batch_count += 1
            _r4c.check_budget()

        if self._calibrating:
            self._cal_output = x  # post-matmul, pre-output-quantization

        if self.out_quantizer.bits < 16 or self.out_quantizer.realint:  # Quantize the output, if needed
            # self.out_quantizer.find_params(x) # QuaRot 源码，现修改在 quantizer.forward 函数中
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


_ATTN_PROJS = ('q_proj', 'k_proj', 'v_proj')
_ATTN_PROJS_O = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
_MLP_PROJS = ('gate_proj', 'up_proj', 'down_proj')


def should_quant_out(name, mode):
    """Return True if layer *name* should get output quantization under *mode*.

    Modes:
        none   – nothing
        up     – up_proj only
        mlp    – gate_proj, up_proj, down_proj
        spec   – all except q/k/v_proj
        speco  – all except q/k/v/o_proj
        all    – every layer (except lm_head)
        r4     – pre-rotation quantizer on down_proj (uses pre_quantizer, not out_quantizer)
        res    – residual quantizers only (no out_quantizer)
        mm     – Q quantizer in attention only (no out_quantizer)
        ex     – all + res + mm
    """
    if mode in ('none', 'r4', 'res', 'mm'):
        return False  # these modes don't use out_quantizer
    if 'lm_head' in name:
        return False
    if mode == 'up':
        return 'up_proj' in name
    if mode == 'mlp':
        return any(p in name for p in _MLP_PROJS)
    if mode == 'spec':
        return not any(p in name for p in _ATTN_PROJS)
    if mode == 'speco':
        return not any(p in name for p in _ATTN_PROJS_O)
    if mode in ('all', 'ex'):
        return True
    return False


def should_quant_pre(name, mode):
    """Return True if layer *name* should get pre-rotation quantization under *mode*."""
    if mode in ('r4', 'ex') and 'down_proj' in name:
        return True
    return False


def needs_residual_quant(mode):
    """Return True if mode requires residual add quantizers."""
    return mode in ('res', 'ex')


def needs_mm_quant(mode):
    """Return True if mode requires Q quantizer in attention."""
    return mode in ('mm', 'ex')


def align_scales_to_groups(scale, zero, maxq, group_size, sym):
    """Convert per-column calibration scales to per-group, expanded back to [K].

    Takes per-column scales [K] and returns [K] scales where groups of
    ``group_size`` consecutive columns share the same (worst-case) scale.
    This simulates the hardware constraint that scales must factor out of
    dot products.

    Parameters
    ----------
    scale : Tensor [K]  — per-column scales from calibration
    zero  : Tensor [K]  — per-column zeros from calibration
    maxq  : Tensor or float — max quantization level
    group_size : int    — target group size (typically w_groupsize)
    sym   : bool        — symmetric quantization

    Returns
    -------
    (scale_aligned [K], zero_aligned [K])
    """
    scale = scale.flatten()
    zero = zero.flatten()
    K = scale.shape[0]

    # Determine effective group size
    G = min(group_size, K)
    if K % G != 0:
        G = K  # fall back to per-tensor (1 group)
    n_groups = K // G

    if sym:
        group_scale = scale.reshape(n_groups, G).max(dim=1)[0]
        group_zero = torch.zeros(n_groups, dtype=zero.dtype, device=zero.device)
    else:
        # Reconstruct representable range per column, then envelope per group
        col_min = -(zero * scale)
        col_max = (maxq - zero) * scale
        group_min = col_min.reshape(n_groups, G).min(dim=1)[0]
        group_max = col_max.reshape(n_groups, G).max(dim=1)[0]
        group_scale = (group_max - group_min) / maxq
        group_zero = torch.round(-group_min / group_scale)
        dead = (group_min == 0) & (group_max == 0)
        group_scale[dead] = 1.0
        group_zero[dead] = 0.0

    # Expand back to [K]
    scale_aligned = group_scale.repeat_interleave(G)[:K]
    zero_aligned = group_zero.repeat_interleave(G)[:K]
    return scale_aligned, zero_aligned


def setup_residual_quantizers(model):
    """Patch each decoder layer to quantize the result of each residual add.

    Adds two ActQuantizer instances per layer:
      layer._attn_res_quantizer  – after attention residual add
      layer._mlp_res_quantizer   – after MLP residual add

    The patched forward applies quantization at both residual points.
    """
    from model_utils import get_layers
    layers = get_layers(model)
    for layer in layers:
        attn_rq = ActQuantizer()
        attn_rq.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
        attn_rq.realint = True
        mlp_rq = ActQuantizer()
        mlp_rq.configure(bits=16, groupsize=-1, sym=True, clip_ratio=1.0)
        mlp_rq.realint = True
        # Store as plain attributes to avoid accelerate dispatch issues
        object.__setattr__(layer, '_attn_res_quantizer', attn_rq)
        object.__setattr__(layer, '_mlp_res_quantizer', mlp_rq)

        orig_forward = layer.forward

        def make_patched_forward(lay, arq, mrq):
            def patched_forward(hidden_states, *args, **kwargs):
                # Ensure residual quantizers are on the correct device
                dev = hidden_states.device
                for _rq in (arq, mrq):
                    if _rq.maxq.device != dev:
                        _rq.maxq = _rq.maxq.to(dev)
                        if _rq.scale is not None:
                            _rq.scale = _rq.scale.to(dev)
                        if _rq.zero is not None:
                            _rq.zero = _rq.zero.to(dev)

                residual = hidden_states
                hidden_states = lay.input_layernorm(hidden_states)
                attn_out = lay.self_attn(
                    hidden_states=hidden_states, **kwargs)
                hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out
                hidden_states = residual + hidden_states
                # Quantize after attention residual add
                if arq.bits < 16 or arq.realint:
                    lay._attn_res_pre = hidden_states  # for calibration collection
                    if not arq.static:
                        arq.find_params(hidden_states)
                    hidden_states = arq(hidden_states).to(hidden_states.dtype)
                    arq.free()

                residual = hidden_states
                hidden_states = lay.post_attention_layernorm(hidden_states)
                hidden_states = lay.mlp(hidden_states)
                hidden_states = residual + hidden_states
                # Quantize after MLP residual add
                if mrq.bits < 16 or mrq.realint:
                    lay._mlp_res_pre = hidden_states  # for calibration collection
                    if not mrq.static:
                        mrq.find_params(hidden_states)
                    hidden_states = mrq(hidden_states).to(hidden_states.dtype)
                    mrq.free()

                return hidden_states
            return patched_forward

        layer.forward = make_patched_forward(layer, attn_rq, mlp_rq)


def link_adjacent_quantizers(model):
    """Detect quantizer→quantizer chains and link scales.

    For Llama: no direct adjacencies exist (residual, LN, SiLU, eltwise-mul
    always intervene). This is a no-op placeholder for future architectures
    where layer outputs may feed directly into another layer's input.
    """
    # Future: walk model graph or use a predefined adjacency map
    # to find (layer_A.out_quantizer → layer_B.quantizer) pairs
    # and set layer_B.quantizer.shared_scale_source = layer_A.out_quantizer
    pass


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
        gscaler=None,
        nvfp4=False,
        scalewise=False,
        hwscale_spec=None,
        layer_act_scale=None,
    ):
        # FP4 is symmetric, per-channel, 4-bit; MSE search disabled (its loop
        # uses uniform-integer rounding that doesn't apply to FP4 levels).
        if nvfp4:
            bits = 4
            sym = True
            perchannel = True
            mse = False
        self.nvfp4 = nvfp4
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.gscaler = gscaler
        if nvfp4:
            # Max FP4 code magnitude is 12 → scale = amax / 12 in find_params.
            self.maxq = torch.tensor(12)
        elif sym:
            self.maxq = torch.tensor(2**(bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

        self.groupsize = groupsize
        self.static_groups = static_groups

        # ── scalewise: pre-round combined hwscale inside the GPTQ loop ──
        self.scalewise = bool(scalewise) and (hwscale_spec is not None) \
            and (layer_act_scale is not None) and (bits < 16)
        self.hwscale_spec = hwscale_spec
        # Per-input-channel activation scale tensor (length K). Stored as fp32
        # and detached on configure; sliced in find_params per weight group.
        if layer_act_scale is None:
            self.layer_act_scale = None
        else:
            self.layer_act_scale = layer_act_scale.detach().to(torch.float32)
        # Layer-wide hwscale global scalar — set lazily by init_scalewise(W).
        self._scalewise_global = None
        # Cursor advanced by find_params to track which K-window is active.
        self._k_start = 0

    @torch.no_grad()
    def init_scalewise(self, W, w_groupsize):
        """Precompute the per-layer hwscale global scalar before fasterquant.

        Mirrors the runtime hwscale snap (see ``snap_to_hwscale``): one global
        FP32 scalar absorbs the bulk of the magnitude, the per-group residual
        is what M<m>S<s> bias=0 represents. The global is shared across all
        K-groups within a layer, so we compute it from the FULL pre-GPTQ
        combined scale tensor here, then reuse it inside every per-group
        find_params call.

        Requires the act-scale grouping to match ``w_groupsize`` — at runtime
        the deployed combined scales are stored at this granularity.
        """
        if not self.scalewise:
            return
        if w_groupsize is None or w_groupsize <= 0:
            raise ValueError(
                "scalewise requires a positive weight groupsize that matches "
                "the runtime hwscale grouping (acc_block_k). Got "
                f"groupsize={w_groupsize}.")
        K = W.shape[1]
        if K % w_groupsize != 0:
            raise ValueError(
                f"K={K} not divisible by w_groupsize={w_groupsize}")
        if self.layer_act_scale.numel() != K:
            raise ValueError(
                f"layer_act_scale length {self.layer_act_scale.numel()} "
                f"does not match weight K dim {K}")
        n_groups = K // w_groupsize
        # Float per-group weight scale, shape [n_out, n_groups], using the
        # same sym/asym/MSE-disabled logic find_params uses by default
        # (compute_initial: scale = amax / maxq for sym; abs-range / maxq
        # for asym). MSE search isn't used here — it'd give a different
        # global per group which is fine since global is computed from the
        # full distribution anyway.
        Wg = W.float().reshape(W.shape[0], n_groups, w_groupsize)
        if self.sym or getattr(self, 'nvfp4', False):
            wmax = Wg.abs().amax(dim=2).clamp(min=1e-5)
            w_scale = wmax / float(self.maxq)
        else:
            wmin = Wg.amin(dim=2)
            wmax = Wg.amax(dim=2)
            w_scale = (wmax - wmin).clamp(min=1e-5) / float(self.maxq)
        a_per_group = self.layer_act_scale.to(W.device).reshape(
            n_groups, w_groupsize).amax(dim=1)              # [n_groups]
        raw = w_scale * a_per_group.unsqueeze(0)            # [n_out, n_groups]
        self._scalewise_global = float(
            hwscale_global_from_spec(self.hwscale_spec, raw_scale=raw))
        # Pre-build the bias-0 spec used inside find_params.
        self._scalewise_bias0_spec = {
            'mantissa_bits': self.hwscale_spec['mantissa_bits'],
            'shift_bits': self.hwscale_spec['shift_bits'],
            'bias': 0,
            'spec': self.hwscale_spec['spec'],
        }
        self._k_start = 0

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

        if self.scalewise and self._scalewise_global is not None:
            # Pre-round the merged (group_scale × act_scale) to its M<m>S<s>
            # representation here so the GPTQ loop's Hessian-aware error
            # compensation absorbs scale-rounding error along with weight
            # rounding. Mirrors snap_to_hwscale() but reuses a precomputed
            # per-layer global so the global is consistent across all groups.
            # Caller (fasterquant) sets ``_k_start`` to the K-position of this
            # weight slice before calling find_params.
            x_dev = self.scale.device
            k_end = min(self._k_start + shape[-1], self.layer_act_scale.numel())
            a_window = self.layer_act_scale[self._k_start:k_end].to(x_dev)
            a_group = a_window.amax().clamp(min=1e-12)        # scalar
            raw = self.scale * a_group                        # [n_out]
            scaled = raw / self._scalewise_global
            stored = snap_scale_to_gscaler(scaled, self._scalewise_bias0_spec)
            combined_q = stored * self._scalewise_global
            # Effective per-group weight scale that GPTQ will quantize against:
            # equals combined_q / a_group, so dequant * a_group reproduces the
            # rounded combined scale exactly at runtime.
            self.scale = combined_q / a_group
            # Asymmetric: zero point was computed against the un-rounded
            # scale; rebuild it for the new scale so the int code lands at
            # the right level.
            if not self.sym:
                self.zero = torch.round(-xmin / self.scale.clamp(min=1e-12))
        elif self.gscaler is not None:
            self.scale = snap_scale_to_gscaler(self.scale, self.gscaler)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x, stochastic=False):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if getattr(self, 'nvfp4', False):
                return fp4_quant_dequant(x, self.scale).to(x_dtype)
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
