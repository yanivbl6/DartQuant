"""Softmax Output Quantization (SMQ).

Simulates hardware softmax with limited output bits by replacing
torch.nn.functional.scaled_dot_product_attention with a custom
implementation that quantizes the softmax output.

Quantization formula:
    scale = 1 / (2**bits - 1)
    out = clamp(round(softmax_out / scale), 0, 2**bits - 1) * scale
"""

import math
import functools
import torch


_original_sdpa = None


def quantized_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, *, _smq_bits=8,
):
    """Drop-in replacement for torch.nn.functional.scaled_dot_product_attention
    that quantizes the softmax output to _smq_bits."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if is_causal:
        assert attn_mask is None, "Cannot use both is_causal and attn_mask"
        causal = torch.ones(L, S, dtype=torch.bool, device=query.device).triu(
            diagonal=S - L + 1)
        attn_weight = attn_weight.masked_fill(causal, float('-inf'))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float('-inf'))
        else:
            attn_weight = attn_weight + attn_mask

    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)

    # SMQ: quantize softmax output
    maxq = 2 ** _smq_bits - 1
    smq_scale = 1.0 / maxq
    attn_weight = torch.clamp(torch.round(attn_weight / smq_scale), 0, maxq) * smq_scale

    attn_weight = attn_weight.to(query.dtype)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


def enable_smq(bits):
    """Replace torch SDPA globally with quantized version."""
    global _original_sdpa
    _original_sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = functools.partial(
        quantized_scaled_dot_product_attention, _smq_bits=bits)


def smq_tag(bits):
    """Return tag suffix for non-trivial SMQ values."""
    if bits <= 0:
        return ""
    return f"_smq{bits}"
