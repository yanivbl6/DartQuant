"""Diagnostic GEMM with toggleable precision stages.

Mirrors int_gemm_capped_reference logic in float, with each precision-loss
stage independently toggleable via a bitmask.

Usage:
    --semi_int_gemm <mask>    e.g. "11111100" (=full), "00000000" (=none), "10000000" (=fake)

Bitmask stages (bits 0-5: precision stages, bits 6-7: diagnostic overrides):
    Bit 0: Act quantization (round + clamp + cast via quantize_to_int)
    Bit 1: Int16 hi/lo decomposition (split into two int8 GEMMs)
    Bit 2: Dot product rounding (partial.round())
    Bit 3: Tier-1 accumulator cap (clamp/wrap to acc_bits)
    Bit 4: Weight ZP correction rounding + re-cap
    Bit 5: Tier-2 integer accumulation (round + clamp)
    Bit 6: Flat matmul — bypass kloop, dequant everything, single matmul
    Bit 7: FQ-act — use sym_quant (fake-quant style) instead of quantize_to_int

Diagnostic keywords:
    flatfq  = 10000010  act quant (quantize_to_int) + flat matmul
    fqact   = 10000001  act quant (sym_quant) + kloop
    fqflat  = 10000011  act quant (sym_quant) + flat matmul  (= exact fake_quant replica)

The 2x2 matrix {quantize_to_int vs sym_quant} x {kloop vs flat_matmul}
isolates whether the discrepancy is in activation quantization or in the
weight decomposition / kloop logic.
"""

import logging
import math
from typing import Optional

import torch

from int_acc_gemm import (
    parse_acc_dtype,
    _decompose_int16_to_int8,
)


def _t2_acc_dtype_info(acc_dtype_str):
    """Return (torch_dtype, int_clamp_max, int_clamp_min) for tier-2 accumulator."""
    result = parse_acc_dtype(acc_dtype_str)
    kind, bits = result[0], result[1]
    if kind == 'int':
        max_val = 2 ** (bits - 1) - 1
        return torch.int64, max_val, -max_val - 1
    elif kind == 'bfloat':
        return torch.bfloat16, None, None
    elif kind == 'float' and bits == 16:
        return torch.float16, None, None
    return torch.float32, None, None


# ---------------------------------------------------------------------------
# Bitmask constants
# ---------------------------------------------------------------------------

MASK_ACT_QUANT = 0       # Activation quantization (round+clamp+cast)
MASK_INT16_DECOMP = 1    # Int16 hi/lo decomposition
MASK_DOT_ROUND = 2       # Dot product rounding
MASK_ACC_CAP = 3          # Tier-1 accumulator cap
MASK_WZP_ROUND_CAP = 4   # Weight ZP correction round + re-cap
MASK_T2_INT = 5           # Tier-2 integer accumulation
MASK_FLAT_MATMUL = 6      # Bypass kloop: dequant everything, single matmul
MASK_FQ_ACT = 7           # Use sym_quant instead of quantize_to_int
MASK_FQ_CAL = 8           # Use fake-quant calibration (strip intgemm from cal tag)
MASK_SANITY = 9           # Bypass everything: run quantizer.forward() + module(x)

NUM_BITS = 10

_KEYWORDS = {
    'full':    '1111110000',  # all precision stages, no diagnostic overrides
    'none':    '0' * NUM_BITS,    # pure float (no act quant, kloop)
    'fake':    '1' + '0' * (NUM_BITS - 1),  # act quant only via quantize_to_int
    'flatfq':  '1000001010',  # quantize_to_int + flat matmul + fq_cal
    'fqact':   '1000000110',  # sym_quant + kloop + fq_cal
    'fqflat':  '1000001110',  # sym_quant + flat matmul + fq_cal (= exact fake_quant replica)
    'fakefq':  '1000000010',  # quantize_to_int + kloop + fq_cal (= fake but with fq calibration)
    'sanity':  '0000000011',  # bypass everything: quantizer.forward() + module(x) + fq_cal
}

_BIT_NAMES = [
    'act_quant', 'int16_decomp', 'dot_round',
    'acc_cap', 'wzp_round_cap', 't2_int',
    'flat_matmul', 'fq_act', 'fq_cal', 'sanity',
]


def parse_mask(mask_str: str) -> str:
    """Parse mask string: keyword or binary string. Returns NUM_BITS-char string of '0'/'1'."""
    s = mask_str.strip().lower()
    if s in _KEYWORDS:
        return _KEYWORDS[s]
    # Accept old shorter masks by padding with '0's
    if len(s) in (6, 8, 9) and all(c in '01' for c in s):
        return s.ljust(NUM_BITS, '0')
    if len(s) != NUM_BITS or not all(c in '01' for c in s):
        raise ValueError(
            f"Invalid semi_int_gemm mask: '{mask_str}'. "
            f"Must be {NUM_BITS}-char binary string or keyword: {list(_KEYWORDS.keys())}")
    return s


def describe_mask(mask: str) -> str:
    """Human-readable description of which stages are enabled."""
    parts = []
    for i, name in enumerate(_BIT_NAMES):
        if i < len(mask) and mask[i] == '1':
            parts.append(name)
    return '+'.join(parts) if parts else 'none'


def _bit(mask, idx):
    return idx < len(mask) and mask[idx] == '1'


# ---------------------------------------------------------------------------
# Weight dequantization helper
# ---------------------------------------------------------------------------

def _dequant_weights(w_int, w_scale, w_group_size, w_zp, K):
    """Reconstruct float weights: W_fq = (w_int + w_zp) * w_scale."""
    N = w_int.shape[0]
    if w_group_size > 0 and w_scale.dim() == 2:
        n_groups = w_scale.shape[1]
        G = w_group_size
        w_adj = w_int.float().reshape(N, n_groups, G)
        if w_zp is not None and w_zp.dim() == 2:
            w_adj = w_adj + w_zp.unsqueeze(2)
        w_float = (w_adj * w_scale.unsqueeze(2)).reshape(N, -1)[:, :K]
    else:
        w_adj = w_int.float()
        if w_zp is not None:
            if w_zp.dim() == 1:
                w_adj = w_adj + w_zp.unsqueeze(1)
            else:
                w_adj = w_adj + w_zp
        if w_scale.dim() == 1:
            w_float = w_adj * w_scale.unsqueeze(1)
        else:
            w_float = w_adj * w_scale
    return w_float


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------

_recon_logged = set()  # track which layers have been logged


def semi_int_gemm(
    x_float: torch.Tensor,
    w_int: torch.Tensor,
    w_scale: torch.Tensor,
    act_quantizer,
    mask: str,
    acc_bits: int = 32,
    block_k: int = 32,
    w_group_size: int = -1,
    bias: Optional[torch.Tensor] = None,
    acc_wrap: bool = False,
    w_zp: Optional[torch.Tensor] = None,
    acc_dtype: str = 'float',
    w_shift_bias: int = 0,
    w_zp_correction: Optional[torch.Tensor] = None,
    w_zp_cross: Optional[torch.Tensor] = None,
    module_weight: Optional[torch.Tensor] = None,
    module: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Diagnostic GEMM with toggleable precision stages."""
    orig_shape = x_float.shape
    K = orig_shape[-1]
    x_2d = x_float.reshape(-1, K)
    M = x_2d.shape[0]
    N = w_int.shape[0]

    # ------------------------------------------------------------------
    # Bit 9: Sanity — run the exact fake_quant code path
    # ------------------------------------------------------------------
    if _bit(mask, MASK_SANITY):
        q = act_quantizer
        # Save quantizer state (per-group conversion may have changed it)
        _saved_gs = getattr(q, 'groupsize', -1)
        _saved_scale = q.scale.clone() if q.scale is not None else None
        _saved_zero = q.zero.clone() if q.zero is not None else None
        # Restore per-column scale from per-group so forward() broadcasts correctly
        if _saved_gs > 0 and q.static:
            q.scale = q.scale.to(x_float.device).repeat_interleave(_saved_gs)[:K]
            if q.zero is not None and q.zero.numel() > 0:
                q.zero = q.zero.to(x_float.device).repeat_interleave(_saved_gs)[:K]
            q.groupsize = -1

        x_dtype = x_float.dtype
        x_out = x_float

        # --- Diagnostic: log state on first call per layer shape ---
        _sanity_key = (N, K)
        if _sanity_key not in _recon_logged:
            _recon_logged.add(_sanity_key)
            logging.warning(
                "[SANITY] shape=[%d,%d] x_dtype=%s q.bits=%d q.realint=%s "
                "q.static=%s q.groupsize=%d q.scale=%s(%s) "
                "module.weight=%s(%s) x_in: mean=%.4e max=%.4e",
                N, K, x_dtype, q.bits, q.realint,
                q.static, q.groupsize,
                list(q.scale.shape) if q.scale is not None else 'None',
                q.scale.dtype if q.scale is not None else '-',
                list(module.weight.shape), module.weight.dtype,
                x_float.float().abs().mean().item(),
                x_float.float().abs().max().item())

        if q.bits < 16 or q.realint:
            x_out = q(x_out).to(x_dtype)
            q.free()

        assert module is not None, "sanity mode requires module= to be passed"

        # --- Diagnostic: check for NaN/Inf after quantizer ---
        if _sanity_key in _recon_logged and x_out.isnan().any():
            logging.error("[SANITY] NaN AFTER quantizer! shape=[%d,%d]", N, K)
        if _sanity_key in _recon_logged and x_out.isinf().any():
            logging.error("[SANITY] Inf AFTER quantizer! shape=[%d,%d]", N, K)

        x_out = module(x_out).to(x_dtype)

        # --- Diagnostic: check output ---
        if (N, K) in _recon_logged:
            _mn = x_out.float().abs().mean().item()
            _mx = x_out.float().abs().max().item()
            if _mn != _mn or _mx > 1e6:  # NaN or huge
                logging.error("[SANITY] BAD output! shape=[%d,%d] mean=%.4e max=%.4e", N, K, _mn, _mx)

        # Restore quantizer state
        q.groupsize = _saved_gs
        q.scale = _saved_scale
        q.zero = _saved_zero
        return x_out

    # ------------------------------------------------------------------
    # Resolve activation quantization method
    # ------------------------------------------------------------------
    do_act_quant = _bit(mask, MASK_ACT_QUANT)
    use_fq_act = _bit(mask, MASK_FQ_ACT)     # sym_quant instead of quantize_to_int
    use_flat = _bit(mask, MASK_FLAT_MATMUL)   # bypass kloop

    # ------------------------------------------------------------------
    # No act quantization — dequantize weights and do plain float matmul
    # ------------------------------------------------------------------
    if not do_act_quant:
        w_float = _dequant_weights(w_int, w_scale, w_group_size, w_zp, K)
        output = x_2d.float() @ w_float.t()
        if bias is not None:
            output = output + bias.unsqueeze(0)
        return output.reshape(*orig_shape[:-1], N)

    # ------------------------------------------------------------------
    # Activation quantization
    # ------------------------------------------------------------------
    if use_fq_act:
        # Bit 7: use sym_quant (same code path as fake_quant's forward()).
        # Reconstruct per-column scale from per-group scale so the quantization
        # is identical to what ActQuantizer.forward() would do after hw_align.
        q = act_quantizer
        if q.static and getattr(q, 'groupsize', -1) > 0:
            G = q.groupsize
            group_scale = q.scale.to(x_2d.device)
            col_scale = group_scale.repeat_interleave(G)[:K]
        elif q.static:
            col_scale = q.scale.to(x_2d.device).flatten()
            if col_scale.shape[0] == 1:
                col_scale = col_scale.expand(K)
        else:
            q.find_params(x_2d)
            col_scale = q.scale.to(x_2d.device)

        from quant_utils import sym_quant
        a_q, _ = sym_quant(x_2d, col_scale, q.maxq.to(x_2d.device))
        _int_dtype = torch.int16 if q.bits > 8 else torch.int8
        a_int = a_q.reshape(-1, K).to(_int_dtype)
        a_zp_correction = None

        # Derive the same group_scale / token_scale as quantize_to_int would
        if q.static and getattr(q, 'groupsize', -1) > 0:
            a_group_scale = group_scale
            a_group_size = G
            a_token_scale = torch.empty(0, device=x_float.device)
            use_act_groups = True
        else:
            a_group_scale = None
            a_group_size = -1
            if q.static:
                per_tensor = col_scale.max()
                a_token_scale = torch.full((M,), per_tensor.item(),
                                           device=x_2d.device, dtype=col_scale.dtype)
            else:
                flat = q.scale.reshape(-1, q.scale.shape[-1])
                a_token_scale = flat[:, 0].contiguous()
            use_act_groups = False
        if not q.static:
            q.free()
    else:
        # Bit 0 only: use quantize_to_int (original path)
        a_int, a_scale, a_zp_correction = act_quantizer.quantize_to_int(x_2d)

        use_act_groups = (getattr(act_quantizer, 'groupsize', -1) > 0
                          and act_quantizer.static)
        if use_act_groups:
            a_group_scale = a_scale
            a_group_size = act_quantizer.groupsize
            a_token_scale = torch.empty(0, device=x_float.device)
        else:
            a_group_scale = None
            a_group_size = -1
            a_token_scale = a_scale

    is_int16 = (a_int.dtype == torch.int16)
    a_work = a_int.float()

    # ------------------------------------------------------------------
    # Bit 6: Flat matmul — dequant everything, single matmul
    # ------------------------------------------------------------------
    if use_flat:
        # Dequant activations back to float
        if use_act_groups and a_group_scale is not None:
            G = a_group_size
            a_fq = a_work.reshape(M, -1, G) * a_group_scale[None, :, None]
            a_fq = a_fq.reshape(M, K)
        elif a_token_scale.numel() > 0:
            a_fq = a_work * a_token_scale.unsqueeze(1)
        else:
            a_fq = a_work

        # Dequant weights
        w_float = _dequant_weights(w_int, w_scale, w_group_size, w_zp, K)

        # --- A/B diagnostic (once per layer shape) ---
        _layer_key = (N, K)
        if _layer_key not in _recon_logged:
            _recon_logged.add(_layer_key)
            # Weight reconstruction check
            if module_weight is not None:
                W_ref = module_weight.float()
                w_err = (w_float - W_ref).abs()
                print(f"[DIAG w] shape=[{N},{K}]  w_err_max={w_err.max().item():.4e}  "
                      f"w_err_mean={w_err.mean().item():.4e}  w_ref={W_ref.abs().mean().item():.4e}  "
                      f"w_scale={w_scale.dtype}  w_zp={w_zp.dtype if w_zp is not None else None}",
                      flush=True)

            # Activation quantization check: compare quantize_to_int vs quantizer.forward()
            q = act_quantizer
            _saved_gs = getattr(q, 'groupsize', -1)
            _saved_sc = q.scale.clone() if q.scale is not None else None
            _saved_zr = q.zero.clone() if q.zero is not None else None

            # Print scales BEFORE restoring
            _q2i_scale = q.scale.to(x_2d.device).float()
            print(f"[DIAG s] shape=[{M},{K}]  bits={q.bits}  gs={_saved_gs}  "
                  f"q2i_scale: shape={list(_q2i_scale.shape)}  "
                  f"min={_q2i_scale.min().item():.6e}  max={_q2i_scale.max().item():.6e}  "
                  f"mean={_q2i_scale.mean().item():.6e}",
                  flush=True)

            if _saved_gs > 0 and q.static:
                q.scale = q.scale.to(x_2d.device).repeat_interleave(_saved_gs)[:K]
                if q.zero is not None and q.zero.numel() > 0:
                    q.zero = q.zero.to(x_2d.device).repeat_interleave(_saved_gs)[:K]
                q.groupsize = -1

            # Print scales AFTER restoring
            _fq_scale = q.scale.to(x_2d.device).float()
            print(f"[DIAG s] shape=[{M},{K}]  bits={q.bits}  gs={q.groupsize}  "
                  f"fq_scale:  shape={list(_fq_scale.shape)}  "
                  f"min={_fq_scale.min().item():.6e}  max={_fq_scale.max().item():.6e}  "
                  f"mean={_fq_scale.mean().item():.6e}",
                  flush=True)

            # Run forward in float32
            if q.bits < 16 or q.realint:
                a_ref_f32 = q(x_float.float()).float()
                q.free()
            else:
                a_ref_f32 = x_float.float()
            q.groupsize = _saved_gs
            q.scale = _saved_sc
            q.zero = _saved_zr
            a_ref_2d = a_ref_f32.reshape(-1, K)

            # Also compute quantize_to_int result in float32 for comparison
            a_q2i_f32 = a_work.clone()  # int values as float
            if use_act_groups and a_group_scale is not None:
                _G = a_group_size
                a_q2i_dq = a_q2i_f32.reshape(M, -1, _G) * a_group_scale[None, :, None]
                a_q2i_dq = a_q2i_dq.reshape(M, K)
            else:
                a_q2i_dq = a_q2i_f32 * a_token_scale.unsqueeze(1) if a_token_scale.numel() > 0 else a_q2i_f32

            a_err = (a_q2i_dq - a_ref_2d).abs()
            # Find worst element
            _worst_idx = a_err.argmax().item()
            _wm, _wk = _worst_idx // K, _worst_idx % K
            _wg = _wk // a_group_size if (use_act_groups and a_group_size > 0) else -1
            _wi = _wk % a_group_size if (use_act_groups and a_group_size > 0) else _wk
            # Manual replication of quantize_to_int for this one element
            _x_val = x_2d[_wm, _wk].float()
            _s_val = a_group_scale[_wg].float()
            _manual_div = _x_val / _s_val
            _manual_round = torch.round(_manual_div)
            # Also check what quantize_to_int ACTUALLY computed via the grouped path
            _x_grouped_check = x_2d.reshape(M, -1, a_group_size)
            _x_at_pos = _x_grouped_check[_wm, _wg, _wi].float()
            _div_at_pos = _x_at_pos / _s_val
            print(f"[DIAG a] shape=[{M},{K}]  bits={q.bits}  "
                  f"act_err_max={a_err.max().item():.4e}  "
                  f"worst@[{_wm},{_wk}](g={_wg},i={_wi}): "
                  f"a_int={a_work[_wm,_wk].item():.0f}  "
                  f"scale={_s_val.item():.6e}  "
                  f"x={_x_val.item():.6e}  "
                  f"x_grouped={_x_at_pos.item():.6e}  "
                  f"div={_manual_div.item():.6f}  "
                  f"round={_manual_round.item():.0f}  "
                  f"div_grouped={_div_at_pos.item():.6f}  "
                  f"fq={a_ref_2d[_wm,_wk].item():.6e}",
                  flush=True)

            # Also compare full outputs (both float32, using module.weight for both)
            if module is not None and module_weight is not None:
                W_f32 = module_weight.float()
                out_ref = (a_ref_2d @ W_f32.t())
                out_q2i = (a_q2i_dq @ W_f32.t())
                o_err = (out_q2i - out_ref).abs()
                print(f"[DIAG o] shape=[{M},{N}]  out_err_max={o_err.max().item():.4e}  "
                      f"out_err_mean={o_err.mean().item():.4e}  out_ref={out_ref.abs().mean().item():.4e}",
                      flush=True)

        output = a_fq @ w_float.t()
        if bias is not None:
            output = output + bias.unsqueeze(0)

        # ZP corrections for asymmetric activations
        if a_zp_correction is not None and w_zp_correction is not None:
            output = output + a_zp_correction.unsqueeze(1) * w_zp_correction.unsqueeze(0)
        if a_zp_correction is not None and w_zp_cross is not None:
            output = output + a_zp_correction.unsqueeze(1) * w_zp_cross.unsqueeze(0)

        return output.reshape(*orig_shape[:-1], N)

    # ------------------------------------------------------------------
    # Stage 1: Int16 hi/lo decomposition
    # ------------------------------------------------------------------
    if _bit(mask, MASK_INT16_DECOMP) and is_int16 and do_act_quant:
        a_hi, a_lo = _decompose_int16_to_int8(a_int)

        out_hi = _semi_int_kloop(
            a_hi.float(), w_int, w_scale, mask, acc_bits, block_k,
            w_group_size, acc_wrap, w_zp, acc_dtype, w_shift_bias,
            a_group_scale=a_group_scale, a_group_size=a_group_size,
            M=M, N=N, K=K)
        out_lo = _semi_int_kloop(
            a_lo.float(), w_int, w_scale, mask, acc_bits, block_k,
            w_group_size, acc_wrap, w_zp, acc_dtype, w_shift_bias,
            a_group_scale=a_group_scale, a_group_size=a_group_size,
            M=M, N=N, K=K)

        output = out_hi * 256.0 + out_lo
    else:
        # Single pass (no decomposition)
        output = _semi_int_kloop(
            a_work, w_int, w_scale, mask, acc_bits, block_k,
            w_group_size, acc_wrap, w_zp, acc_dtype, w_shift_bias,
            a_group_scale=a_group_scale, a_group_size=a_group_size,
            M=M, N=N, K=K)

    # ------------------------------------------------------------------
    # Post-loop: apply remaining scales (always float, same as int_gemm)
    # ------------------------------------------------------------------
    if a_group_size <= 0:
        output = output * a_token_scale.unsqueeze(1)

    if w_group_size <= 0:
        output = output * w_scale.unsqueeze(0)

    if bias is not None:
        output = output + bias.unsqueeze(0)

    # ------------------------------------------------------------------
    # Post-kernel ZP corrections (always float, same as int_gemm_capped)
    # ------------------------------------------------------------------
    if a_zp_correction is not None and w_zp_correction is not None:
        output = output + a_zp_correction.unsqueeze(1) * w_zp_correction.unsqueeze(0)

    if a_zp_correction is not None and w_zp_cross is not None:
        output = output + a_zp_correction.unsqueeze(1) * w_zp_cross.unsqueeze(0)

    return output.reshape(*orig_shape[:-1], N)


# ---------------------------------------------------------------------------
# K-block loop (core of the reference kernel)
# ---------------------------------------------------------------------------

def _semi_int_kloop(
    a_work: torch.Tensor,      # [M, K] float (quantized-as-float or raw scaled)
    w_int: torch.Tensor,       # [N, K] int8
    w_scale: torch.Tensor,     # [N] or [N, n_groups]
    mask: str,
    acc_bits: int,
    block_k: int,
    w_group_size: int,
    acc_wrap: bool,
    w_zp: Optional[torch.Tensor],
    acc_dtype: str,
    w_shift_bias: int,
    a_group_scale: Optional[torch.Tensor] = None,
    a_group_size: int = -1,
    M: int = 0, N: int = 0, K: int = 0,
) -> torch.Tensor:
    """K-block loop mirroring int_gemm_capped_reference with bitmask toggles."""

    # Tier-1 accumulator range
    acc_max = 2 ** (acc_bits - 1) - 1
    acc_min = -acc_max - 1
    acc_range = acc_max - acc_min + 1

    # Tier-2 accumulator setup
    t2_dtype, t2_clamp_max, t2_clamp_min = _t2_acc_dtype_info(acc_dtype)
    t2_is_int = t2_clamp_max is not None
    _, _, t2_frac_bits = parse_acc_dtype(acc_dtype)
    t2_frac_scale = float(1 << t2_frac_bits) if t2_frac_bits > 0 else 1.0

    if _bit(mask, MASK_T2_INT) and t2_is_int:
        output = torch.zeros(M, N, dtype=torch.int64, device=a_work.device)
    else:
        output = torch.zeros(M, N, dtype=torch.float32, device=a_work.device)

    for k_start in range(0, K, block_k):
        k_end = min(k_start + block_k, K)

        a_block = a_work[:, k_start:k_end]
        w_block = w_int[:, k_start:k_end]

        # Dot product
        partial = a_block.float() @ w_block.float().t()

        # Stage 2: Dot product rounding
        if _bit(mask, MASK_DOT_ROUND):
            partial = partial.round()

        # Stage 3: Tier-1 accumulator cap
        if _bit(mask, MASK_ACC_CAP):
            if acc_wrap:
                partial = ((partial.long() - acc_min) % acc_range + acc_min).float()
            else:
                partial = partial.clamp(acc_min, acc_max)

        # Stage 4: Weight ZP correction
        if w_zp is not None:
            a_block_sum = a_block.float().sum(dim=1)  # [M]
            if _bit(mask, MASK_ACT_QUANT):
                a_block_sum = a_block_sum.round()

            if w_group_size > 0 and w_zp.dim() == 2:
                g_idx = k_start // w_group_size
                w_zp_g = w_zp[:, g_idx]
            else:
                w_zp_g = w_zp

            correction = a_block_sum.unsqueeze(1) * w_zp_g.unsqueeze(0)
            if _bit(mask, MASK_WZP_ROUND_CAP):
                correction = correction.round()
            partial = partial + correction

            # Re-cap after correction
            if _bit(mask, MASK_WZP_ROUND_CAP) and _bit(mask, MASK_ACC_CAP):
                if acc_wrap:
                    partial = ((partial.long() - acc_min) % acc_range + acc_min).float()
                else:
                    partial = partial.clamp(acc_min, acc_max)

        # Per-group activation scale
        if a_group_size > 0 and a_group_scale is not None:
            a_g_idx = k_start // a_group_size
            partial = partial * a_group_scale[a_g_idx]

        # Weight group scale
        if w_group_size > 0:
            g_idx = k_start // w_group_size
            contrib = partial.float() * w_scale[:, g_idx].unsqueeze(0)
        else:
            contrib = partial.float()

        # Stage 5: Tier-2 accumulation
        if _bit(mask, MASK_T2_INT) and t2_is_int:
            if t2_frac_bits > 0:
                output += (contrib * t2_frac_scale).round().long()
            else:
                output += contrib.round().long()
            output = output.clamp(t2_clamp_min, t2_clamp_max)
        else:
            output += contrib.to(output.dtype)

    # Convert to float, undo shifts
    total_shift = t2_frac_bits + w_shift_bias
    output = output.float()
    if total_shift > 0:
        output *= 2.0 ** (-total_shift)

    return output
