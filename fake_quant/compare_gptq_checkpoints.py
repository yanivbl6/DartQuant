#!/usr/bin/env python3
"""Compare GPTQ checkpoint weights with vs without int_gemm.

CPU-only — no model loading, just state_dict comparison.
"""
import torch
import torch.nn.functional as F
import os

CKPT_ROOT = "/data/users/yanivbl/gptq_checkpoints"

PAIRS = [
    {
        "label": "Pair 1: wAsym (bug scenario)",
        "A": "baseline_Llama-3.2-1B-Instruct_w0a8k8v8_g128_aAsym_wAsym_kAsym_vAsym_kvex8_projex16_imitate-Q4-K-S",
        "B": "baseline_Llama-3.2-1B-Instruct_w0a8k8v8_g128_aAsym_wAsym_kAsym_vAsym_kvex8_projex16_intgemm_acc16_bk128_imitate-Q4-K-S",
        "subdir": "Llama-3.2-1B-Instruct_w0",
    },
    {
        "label": "Pair 2: wSym + pwl (control)",
        "A": "baseline_Llama-3.2-1B-Instruct_w4a8k8v8_g128_aAsym_wSym_kSym_vSym_kvex8_projex15_pwl",
        "B": "baseline_Llama-3.2-1B-Instruct_w4a8k8v8_g128_aAsym_wSym_kSym_vSym_kvex8_projex15_pwl_intgemm_acc16_bk128",
        "subdir": "Llama-3.2-1B-Instruct_w4",
    },
]


def load_sd(ckpt_name, subdir):
    d = os.path.join(CKPT_ROOT, ckpt_name, subdir)
    files = sorted(f for f in os.listdir(d) if f.endswith(".pth"))
    sd = {}
    for f in files:
        sd.update(torch.load(os.path.join(d, f), map_location="cpu"))
    return sd


def compare(pair):
    print(f"\n{'='*80}")
    print(f"  {pair['label']}")
    print(f"  A (no intgemm): ...{pair['A'][-60:]}")
    print(f"  B (intgemm):    ...{pair['B'][-60:]}")
    print(f"{'='*80}")

    sd_a = load_sd(pair["A"], pair["subdir"])
    sd_b = load_sd(pair["B"], pair["subdir"])

    # Only compare weight tensors in model.layers
    keys = sorted(
        k for k in sd_a
        if k in sd_b and ".weight" in k and "model.layers." in k and sd_a[k].dim() == 2
    )

    n_bad = 0
    first_bad = None
    print(f"\n{'Layer':<65s} {'cos':>8s} {'maxdiff':>10s} {'reldiff':>10s}  {'std_A':>10s} {'std_B':>10s}")
    print("-" * 120)

    for k in keys:
        a = sd_a[k].float().flatten()
        b = sd_b[k].float().flatten()
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        maxdiff = (a - b).abs().max().item()
        denom = a.abs().max().item()
        reldiff = maxdiff / denom if denom > 0 else 0.0
        std_a = a.std().item()
        std_b = b.std().item()

        flag = ""
        if cos < 0.999 or reldiff > 0.1:
            flag = " *** BAD ***"
            n_bad += 1
            if first_bad is None:
                first_bad = k

        # Print short key
        short = k.replace("model.layers.", "L").replace(".self_attn.", ".sa.").replace(".mlp.", ".mlp.")
        print(f"{short:<65s} {cos:>8.6f} {maxdiff:>10.3e} {reldiff:>10.3e}  {std_a:>10.4e} {std_b:>10.4e}{flag}")

    print(f"\nSummary: {len(keys)} layers compared, {n_bad} flagged as divergent")
    if first_bad:
        print(f"First divergent layer: {first_bad}")
    else:
        print("All layers look similar — GPTQ weights are NOT corrupted by int_gemm")


if __name__ == "__main__":
    for pair in PAIRS:
        try:
            compare(pair)
        except Exception as e:
            print(f"\nSkipping {pair['label']}: {e}")
