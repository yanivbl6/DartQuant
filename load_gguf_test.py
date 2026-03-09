"""
Load a GGUF file and optionally validate its weights against the HuggingFace model.

Usage:
  python load_gguf_test.py <file.gguf>              # inspect metadata + tensors
  python load_gguf_test.py <file.gguf> --validate    # compare against HF model
"""
import sys
import re
import argparse
import numpy as np
from gguf import GGUFReader, dequantize


# ── GGUF → HuggingFace tensor name mapping for Llama ──

GGUF_TO_HF_STATIC = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

GGUF_TO_HF_LAYER = {
    "attn_norm.weight":  "input_layernorm.weight",
    "attn_q.weight":     "self_attn.q_proj.weight",
    "attn_k.weight":     "self_attn.k_proj.weight",
    "attn_v.weight":     "self_attn.v_proj.weight",
    "attn_out.weight":    "self_attn.o_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight":   "post_attention_layernorm.weight",
    "ffn_gate.weight":   "mlp.gate_proj.weight",
    "ffn_up.weight":     "mlp.up_proj.weight",
    "ffn_down.weight":   "mlp.down_proj.weight",
}

SKIP_TENSORS = {"rope_freqs.weight"}


def gguf_name_to_hf(name: str) -> str | None:
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


# ── Inspect mode ──

def inspect_gguf(path: str):
    reader = GGUFReader(path)

    print(f"=== GGUF: {path} ===")
    print(f"Metadata entries: {len(reader.fields)}")
    for name, field in reader.fields.items():
        if len(field.data) <= 8:
            try:
                val = field.parts[field.data[0]]
                val = val.tobytes().decode("utf-8", errors="replace") if hasattr(val, "tobytes") else val
            except Exception:
                val = "<binary>"
        else:
            val = f"<{len(field.data)} elements>"
        print(f"  {name}: {val}")

    print(f"\nTensors: {len(reader.tensors)}")
    total_bytes = 0
    for t in reader.tensors:
        size = t.n_bytes
        total_bytes += size
        print(f"  {t.name:60s}  shape={list(t.shape):20s}  type={t.tensor_type.name:12s}  {size/1e6:.2f} MB")

    print(f"\nTotal tensor data: {total_bytes / 1e9:.3f} GB")
    print("Load OK!")


# ── Validate mode ──

def validate_gguf(gguf_path: str, model_name: str, hf_token: str | None):
    import torch

    print(f"=== Validating GGUF against HF model ===")
    print(f"GGUF:  {gguf_path}")
    print(f"Model: {model_name}")

    # Load GGUF
    print("\nLoading GGUF file...")
    reader = GGUFReader(gguf_path)
    gguf_tensors = {t.name: t for t in reader.tensors}
    print(f"  {len(gguf_tensors)} tensors loaded")

    # Load HF model
    print("Loading HuggingFace model...")
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32,
                                              token=hf_token, low_cpu_mem_usage=True)
    hf_sd = model.state_dict()
    print(f"  {len(hf_sd)} tensors in state_dict")

    # Check for tied embeddings (no output.weight in GGUF)
    has_output = "output.weight" in gguf_tensors
    tied_embeddings = not has_output
    if tied_embeddings:
        print("  Note: no output.weight in GGUF → tied embeddings (will compare token_embd with lm_head too)")

    # Compare
    print(f"\n{'GGUF Tensor':<45s} {'Type':<8s} {'Shape Match':<12s} {'MSE':>12s} {'MaxErr':>12s} {'Status'}")
    print("-" * 105)

    matched = 0
    skipped = 0
    failed = 0
    results = []

    for gguf_name, gguf_tensor in gguf_tensors.items():
        hf_name = gguf_name_to_hf(gguf_name)
        if hf_name is None:
            skipped += 1
            continue

        if hf_name not in hf_sd:
            print(f"  {gguf_name:<45s} {'?':<8s} {'MISSING':>12s}")
            failed += 1
            continue

        # Dequantize GGUF tensor
        qtype = gguf_tensor.tensor_type.name
        if qtype in ("F32", "F16"):
            gguf_data = np.array(gguf_tensor.data, dtype=np.float32).reshape(gguf_tensor.shape)
        else:
            gguf_data = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)

        hf_data = hf_sd[hf_name].float().numpy()

        # GGUF may store 2D weights transposed relative to HF
        if gguf_data.shape != hf_data.shape and gguf_data.ndim == 2:
            gguf_data = gguf_data.T

        shape_ok = gguf_data.shape == hf_data.shape
        if not shape_ok:
            print(f"  {gguf_name:<45s} {qtype:<8s} {'MISMATCH':>12s}  gguf={gguf_data.shape} hf={hf_data.shape}")
            failed += 1
            continue

        diff = gguf_data - hf_data
        mse = float(np.mean(diff ** 2))
        max_err = float(np.max(np.abs(diff)))

        # Thresholds: F32 should be exact, quantized types have error
        if qtype == "F32":
            ok = max_err < 1e-5
        else:
            ok = mse < 0.01  # generous threshold for 4-bit quant

        status = "OK" if ok else "WARN"
        if not ok:
            failed += 1
        else:
            matched += 1

        results.append((gguf_name, qtype, shape_ok, mse, max_err, status))
        print(f"  {gguf_name:<45s} {qtype:<8s} {'OK':<12s} {mse:12.2e} {max_err:12.4f} {status}")

    # Check tied embeddings against lm_head
    if tied_embeddings and "token_embd.weight" in gguf_tensors:
        gguf_tensor = gguf_tensors["token_embd.weight"]
        qtype = gguf_tensor.tensor_type.name
        if qtype in ("F32", "F16"):
            gguf_data = np.array(gguf_tensor.data, dtype=np.float32).reshape(gguf_tensor.shape)
        else:
            gguf_data = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)

        lm_head = hf_sd["lm_head.weight"].float().numpy()
        if gguf_data.shape != lm_head.shape and gguf_data.ndim == 2:
            gguf_data = gguf_data.T
        if gguf_data.shape == lm_head.shape:
            diff = gguf_data - lm_head
            mse = float(np.mean(diff ** 2))
            max_err = float(np.max(np.abs(diff)))
            ok = mse < 0.01
            status = "OK" if ok else "WARN"
            print(f"  {'token_embd→lm_head (tied)':<45s} {qtype:<8s} {'OK':<12s} {mse:12.2e} {max_err:12.4f} {status}")

    print(f"\n=== Summary: {matched} matched, {skipped} skipped, {failed} failed ===")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Inspect or validate GGUF files")
    parser.add_argument("gguf_path", help="Path to .gguf file")
    parser.add_argument("--validate", action="store_true", help="Validate against HF model")
    parser.add_argument("--model", default=None,
                        help="HF model name or path (default: auto-detect from GGUF metadata)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token")
    args = parser.parse_args()

    if args.validate:
        # Auto-detect model path if not given
        model_name = args.model
        if model_name is None:
            # Try to read from GGUF metadata
            reader = GGUFReader(args.gguf_path)
            if "general.name" in reader.fields:
                field = reader.fields["general.name"]
                name = field.parts[field.data[0]].tobytes().decode("utf-8")
                print(f"Detected model name from GGUF: {name}")
            # Default to the 1B model path used in this project
            from experiment_config import MODEL_MAP
            model_name = MODEL_MAP.get("1b")
            if model_name is None:
                print("ERROR: Could not determine model path. Use --model.")
                sys.exit(1)
            print(f"Using model path: {model_name}")

        success = validate_gguf(args.gguf_path, model_name, args.hf_token)
        sys.exit(0 if success else 1)
    else:
        inspect_gguf(args.gguf_path)


if __name__ == "__main__":
    main()
