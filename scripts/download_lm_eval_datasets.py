#!/usr/bin/env python
"""Download all required datasets to the HF cache if not already present.

Covers both PPL/calibration datasets (wikitext2, ptb, c4) and lm-eval
benchmark datasets (hellaswag, piqa, arc, winogrande, etc.).

Run with: python scripts/download_lm_eval_datasets.py
Datasets are stored under $HF_HOME/datasets (default: /data/data/huggingface/datasets).

NOTE: c4 train is very large. The script only probes it via streaming to
verify connectivity. The actual c4 .arrow cache files used by
fake_quant/data_utils.py must already be present at the expected paths.
"""
import os

# Ensure we can reach HuggingFace Hub for downloading
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import datasets

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# PPL evaluation / calibration datasets (used by fake_quant/data_utils.py)
PPL_TASKS = [
    ("wikitext", "wikitext-2-raw-v1", {}),
    ("ptb_text_only", "penn_treebank", {"trust_remote_code": True}),
    ("allenai/c4", "en", {"split": "validation"}),
    ("allenai/c4", "en", {"split": "train", "streaming": True}),
]

# lm-eval benchmark datasets
LM_EVAL_TASKS = [
    ("Rowan/hellaswag", None, {}),
    ("baber/piqa", None, {}),
    ("allenai/ai2_arc", "ARC-Easy", {}),
    ("allenai/ai2_arc", "ARC-Challenge", {}),
    ("allenai/winogrande", "winogrande_xl", {}),
    ("EleutherAI/lambada_openai", None, {}),
    ("allenai/social_i_qa", None, {"trust_remote_code": True}),
    ("allenai/openbookqa", None, {}),
] + [("cais/mmlu", s, {}) for s in MMLU_SUBJECTS]


def download_task(path, name, kwargs):
    """Download a single dataset task. Returns (label, ok, message)."""
    label = f"{path}" + (f" ({name})" if name else "")
    extra = {k: v for k, v in kwargs.items() if k not in ("streaming",)}
    split_info = kwargs.get("split", "")
    if split_info:
        label += f" [{split_info}]"
    try:
        if kwargs.get("streaming"):
            # For huge datasets (c4 train): iterate a few batches to
            # populate the cache, then stop.
            ds = datasets.load_dataset(path, name, streaming=True, **extra)
            count = 0
            for _ in ds:
                count += 1
                if count >= 2:
                    break
            print(f"  OK (streaming probe): {label}")
        else:
            datasets.load_dataset(path, name, **extra)
            print(f"  OK: {label}")
        return True
    except Exception as e:
        if "offline" in str(e).lower():
            print(f"  MISSING (offline): {label} — re-run without HF_DATASETS_OFFLINE=1")
        else:
            print(f"  FAILED: {label}: {e}")
        return False


def main():
    cache_dir = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
    )
    print(f"Cache dir: {cache_dir}")

    print("\n=== PPL / calibration datasets ===")
    for path, name, kwargs in PPL_TASKS:
        download_task(path, name, kwargs)

    print("\n=== lm-eval benchmark datasets ===")
    for path, name, kwargs in LM_EVAL_TASKS:
        download_task(path, name, kwargs)


if __name__ == "__main__":
    main()
