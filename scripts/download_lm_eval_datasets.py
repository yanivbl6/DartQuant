#!/usr/bin/env python
"""Download lm-eval datasets to the HF cache if not already present.

Run with: HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/download_lm_eval_datasets.py
Datasets are stored under $HF_HOME/datasets (default: /data/data/huggingface/datasets).
"""
import datasets
import os

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

TASKS = [
    ("Rowan/hellaswag", None, {}),
    ("baber/piqa", None, {}),
    ("allenai/ai2_arc", "ARC-Easy", {}),
    ("allenai/ai2_arc", "ARC-Challenge", {}),
    ("allenai/winogrande", "winogrande_xl", {}),
    ("EleutherAI/lambada_openai", None, {}),
    ("allenai/social_i_qa", None, {"trust_remote_code": True}),
    ("allenai/openbookqa", None, {}),
] + [("cais/mmlu", s, {}) for s in MMLU_SUBJECTS]


def main():
    cache_dir = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
    )
    print(f"Cache dir: {cache_dir}")

    for path, name, kwargs in TASKS:
        label = f"{path}" + (f" ({name})" if name else "")
        try:
            datasets.load_dataset(path, name, **kwargs)
            print(f"  OK: {label}")
        except Exception as e:
            if "offline" in str(e).lower():
                print(f"  MISSING (offline): {label} — re-run without HF_DATASETS_OFFLINE=1")
            else:
                print(f"  FAILED: {label}: {e}")


if __name__ == "__main__":
    main()
