"""
Utilities for loading GGUF pre-quantized weights into a HuggingFace model.

Reuses the GGUF→HF name mapping from load_gguf_test.py.
"""
import re
import logging
import numpy as np
import torch
from gguf import GGUFReader, dequantize


# ── GGUF → HuggingFace tensor name mapping for Llama ──

GGUF_TO_HF_STATIC = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

GGUF_TO_HF_LAYER = {
    "attn_norm.weight":   "input_layernorm.weight",
    "attn_q.weight":      "self_attn.q_proj.weight",
    "attn_k.weight":      "self_attn.k_proj.weight",
    "attn_v.weight":      "self_attn.v_proj.weight",
    "attn_out.weight":    "self_attn.o_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight":    "post_attention_layernorm.weight",
    "ffn_gate.weight":    "mlp.gate_proj.weight",
    "ffn_up.weight":      "mlp.up_proj.weight",
    "ffn_down.weight":    "mlp.down_proj.weight",
}

SKIP_TENSORS = {"rope_freqs.weight"}

# GGUF tensor suffixes that need Q/K reverse-permutation
_QK_PERMUTED = {"attn_q.weight", "attn_k.weight"}

# ── GGUF quantization type specs (for --quant_warnings) ──

GGUF_TYPE_SPECS = {
    'Q4_0': {'bits': 4, 'group_size': 32,  'sym': True},
    'Q4_1': {'bits': 4, 'group_size': 32,  'sym': False},
    'Q4_K': {'bits': 4, 'group_size': 256, 'sym': False},
    'Q5_0': {'bits': 5, 'group_size': 32,  'sym': True},
    'Q5_1': {'bits': 5, 'group_size': 32,  'sym': False},
    'Q5_K': {'bits': 5, 'group_size': 256, 'sym': False},
    'Q6_K': {'bits': 6, 'group_size': 256, 'sym': False},
    'Q8_0': {'bits': 8, 'group_size': 32,  'sym': True},
    'Q8_K': {'bits': 8, 'group_size': 256, 'sym': False},
    'F32':  {'bits': 32, 'group_size': 0,  'sym': None},
    'F16':  {'bits': 16, 'group_size': 0,  'sym': None},
}


def gguf_name_to_hf(name):
    """Convert a GGUF tensor name to the corresponding HuggingFace state_dict key."""
    if name in SKIP_TENSORS:
        return None
    if name in GGUF_TO_HF_STATIC:
        return GGUF_TO_HF_STATIC[name]
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        if suffix in GGUF_TO_HF_LAYER:
            return f"model.layers.{layer_idx}.{GGUF_TO_HF_LAYER[suffix]}"
    return None


def _reverse_permute(weights, n_head):
    """Reverse the Q/K row permutation applied by llama.cpp's convert.py.

    During GGUF creation, Q and K weight rows are interleaved by head-halves
    for efficient RoPE computation.  This undoes that permutation back to HF order.
    """
    return (weights.reshape(n_head, weights.shape[0] // n_head // 2, 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


def _dequantize_tensor(gguf_tensor):
    """Dequantize a GGUF tensor to a float32 numpy array."""
    qtype = gguf_tensor.tensor_type.name
    if qtype in ("F32", "F16"):
        return np.array(gguf_tensor.data, dtype=np.float32).reshape(gguf_tensor.shape)
    return dequantize(gguf_tensor.data, gguf_tensor.tensor_type)


def _check_quant_mismatch(gguf_name, qtype_name, w_bits, w_groupsize, w_sym):
    """Emit warnings if GGUF tensor spec mismatches the quantization config."""
    spec = GGUF_TYPE_SPECS.get(qtype_name)
    if spec is None or spec['sym'] is None:
        return  # F32/F16 or unknown type — skip

    warnings = []
    if spec['bits'] != w_bits:
        warnings.append(f"bits={spec['bits']} vs w_bits={w_bits}")
    if w_groupsize > 0 and spec['group_size'] > 0 and spec['group_size'] != w_groupsize:
        warnings.append(f"group_size={spec['group_size']} vs w_groupsize={w_groupsize}")
    if spec['sym'] != w_sym:
        sym_str = "sym" if spec['sym'] else "asym"
        cfg_str = "sym" if w_sym else "asym"
        warnings.append(f"{sym_str} vs config {cfg_str}")

    if warnings:
        logging.warning("GGUF quant mismatch [%s] type=%s: %s",
                        gguf_name, qtype_name, ", ".join(warnings))


def load_gguf_weights(model, gguf_path, quant_warnings=False,
                      w_bits=4, w_groupsize=128, w_sym=True):
    """Load dequantized GGUF weights into a HuggingFace model.

    Replaces model weights in-place, preserving the original dtype.
    Automatically reverses the Q/K row permutation applied by llama.cpp.
    Returns a dict mapping HF key -> GGUF quant type name.
    """
    logging.info("Loading GGUF weights from: %s", gguf_path)
    reader = GGUFReader(gguf_path)
    gguf_tensors = {t.name: t for t in reader.tensors}

    sd = model.state_dict()
    has_output = "output.weight" in gguf_tensors
    replaced = {}
    skipped = []

    # Read head counts from model config for Q/K reverse permutation
    config = model.config
    n_heads = getattr(config, 'num_attention_heads', None)
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)

    for gguf_name, gguf_tensor in gguf_tensors.items():
        hf_name = gguf_name_to_hf(gguf_name)
        if hf_name is None:
            skipped.append(gguf_name)
            continue

        if hf_name not in sd:
            logging.warning("GGUF tensor %s -> HF key %s not found in model, skipping",
                            gguf_name, hf_name)
            continue

        qtype_name = gguf_tensor.tensor_type.name

        if quant_warnings:
            _check_quant_mismatch(gguf_name, qtype_name, w_bits, w_groupsize, w_sym)

        # Dequantize to float32 numpy
        data = _dequantize_tensor(gguf_tensor)

        # Handle transposition (GGUF stores 2D weights transposed vs HF)
        target_shape = sd[hf_name].shape
        if data.shape != target_shape and data.ndim == 2:
            data = data.T
        if data.shape != target_shape:
            logging.warning("Shape mismatch for %s: GGUF %s vs HF %s, skipping",
                            gguf_name, data.shape, target_shape)
            continue

        # Reverse Q/K row permutation from llama.cpp's convert.py
        suffix = gguf_name.split(".", 2)[-1] if "." in gguf_name else gguf_name
        if suffix in _QK_PERMUTED and n_heads is not None:
            n_head = n_heads if "attn_q" in gguf_name else n_kv_heads
            data = _reverse_permute(data, n_head)

        # Convert to torch tensor, preserving original dtype
        original_dtype = sd[hf_name].dtype
        new_weight = torch.from_numpy(data.copy()).to(original_dtype)

        # Replace in model (navigate to the actual parameter)
        _set_parameter(model, hf_name, new_weight)
        replaced[hf_name] = qtype_name

    # Handle tied embeddings: if no output.weight in GGUF, copy token_embd to lm_head
    if not has_output and "token_embd.weight" in gguf_tensors and "lm_head.weight" in sd:
        if "model.embed_tokens.weight" in replaced:
            embed_param = sd["model.embed_tokens.weight"]
            # Re-read the now-updated embed_tokens weight
            embed_data = _dequantize_tensor(gguf_tensors["token_embd.weight"])
            if embed_data.shape != sd["lm_head.weight"].shape and embed_data.ndim == 2:
                embed_data = embed_data.T
            lm_dtype = sd["lm_head.weight"].dtype
            _set_parameter(model, "lm_head.weight",
                           torch.from_numpy(embed_data.copy()).to(lm_dtype))
            replaced["lm_head.weight"] = replaced["model.embed_tokens.weight"]
            logging.info("Tied embeddings: copied token_embd -> lm_head.weight")

    logging.info("GGUF weights loaded: %d replaced, %d skipped (%s)",
                 len(replaced), len(skipped), ", ".join(skipped))
    return replaced


def _set_parameter(model, dotted_name, new_tensor):
    """Set a parameter in a model by its dotted state_dict key."""
    parts = dotted_name.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    param_name = parts[-1]
    old_param = getattr(obj, param_name)
    setattr(obj, param_name, torch.nn.Parameter(new_tensor, requires_grad=old_param.requires_grad))
