#!/usr/bin/env python
"""Download all required datasets to the HF cache if not already present.

Covers both PPL/calibration datasets (wikitext2, ptb, c4) and lm-eval
benchmark datasets (hellaswag, piqa, arc, winogrande, etc.).

Usage:
    python scripts/download_lm_eval_datasets.py            # download missing
    python scripts/download_lm_eval_datasets.py --check     # check for corruption
    python scripts/download_lm_eval_datasets.py --fix       # remove corrupt caches & re-download

Datasets are stored under $HF_HOME/datasets (default: /data/data/huggingface/datasets).

NOTE: c4 train is very large. The script only probes it via streaming to
verify connectivity. The actual c4 .arrow cache files used by
fake_quant/data_utils.py must already be present at the expected paths.
"""
import argparse
import glob
import os
import shutil

# Ensure we can reach HuggingFace Hub for downloading
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import datasets

# Root used by fake_quant/data_utils.py for direct arrow file loading
DATA_UTILS_ROOT = '/data/data/huggingface/datasets'

# Arrow file patterns that data_utils.py needs at runtime (relative to DATA_UTILS_ROOT)
PPL_ARROW_PATTERNS = [
    ('wikitext2',  'wikitext/wikitext-2-raw-v1/**/wikitext-test.arrow'),
    ('wikitext2',  'wikitext/wikitext-2-raw-v1/**/wikitext-train.arrow'),
    ('ptb',        'ptb_text_only/penn_treebank/**/ptb_text_only-test.arrow'),
    ('ptb',        'ptb_text_only/penn_treebank/**/ptb_text_only-train.arrow'),
    ('c4 val',     'allenai___c4/default-*/0.0.0/*/c4-validation-*.arrow'),
    ('c4 train',   'allenai___c4/default-*/0.0.0/*/c4-train-*.arrow'),
]

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


def _cache_dir_for_task(cache_root, path, config_name):
    """Return the HF datasets cache directory for a given dataset/config.

    HuggingFace normalises 'org/dataset' -> 'org___dataset'.
    """
    normalized = path.replace("/", "___")
    config = config_name or "default"
    return os.path.join(cache_root, normalized, config)


def check_cache(cache_root, path, config_name, kwargs):
    """Check a single dataset cache for corruption.

    Tries an offline load_dataset() — this is the authoritative test,
    catching exactly the failures the user would hit at runtime.

    Returns (status, detail) where status is one of:
      'ok'        – loads successfully in offline mode
      'missing'   – no cached data found
      'corrupt'   – cache exists but load_dataset fails
    """
    # Try the real load path (offline) — this is the authoritative test
    old_offline = os.environ.get("HF_DATASETS_OFFLINE")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        extra = {k: v for k, v in kwargs.items() if k not in ("streaming",)}
        datasets.load_dataset(path, config_name, **extra)
        return 'ok', ''
    except Exception as e:
        err = str(e)
        detail = err.split('\n')[0][:120]

        # Distinguish missing (never downloaded) from corrupt (broken cache)
        cache_dir = _cache_dir_for_task(cache_root, path, config_name)
        normalized = path.replace("/", "___")
        parent = os.path.join(cache_root, normalized)
        has_any_cache = os.path.isdir(cache_dir) or (
            os.path.isdir(parent) and os.listdir(parent))

        if not has_any_cache and ("offline" in err.lower() or "cache" in err.lower()):
            return 'missing', 'not cached'

        # Cache exists but is broken — add diagnostic info
        extras = []
        if os.path.isdir(parent):
            arrows = glob.glob(os.path.join(parent, '**', '*.arrow'), recursive=True)
            locks = glob.glob(os.path.join(parent, '**', '*.lock'), recursive=True)
            if not arrows:
                extras.append('no arrow files')
            if locks:
                extras.append(f'{len(locks)} lock file(s)')
        if extras:
            detail += f' [{", ".join(extras)}]'
        return 'corrupt', detail
    finally:
        if old_offline is not None:
            os.environ["HF_DATASETS_OFFLINE"] = old_offline
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)


def purge_cache(cache_root, path, config_name):
    """Remove the cache directory for a dataset/config."""
    cache_dir = _cache_dir_for_task(cache_root, path, config_name)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        return True
    return False


def download_task(path, name, kwargs):
    """Download a single dataset task. Returns True on success."""
    label = f"{path}" + (f" ({name})" if name else "")
    extra = {k: v for k, v in kwargs.items() if k not in ("streaming",)}
    split_info = kwargs.get("split", "")
    if split_info:
        label += f" [{split_info}]"
    try:
        if kwargs.get("streaming"):
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


ALL_TASKS = PPL_TASKS + LM_EVAL_TASKS


def _task_label(path, name, kwargs):
    label = f"{path}" + (f" ({name})" if name else "")
    split_info = kwargs.get("split", "")
    if split_info:
        label += f" [{split_info}]"
    return label


def check_ppl_arrows():
    """Check PPL dataset arrow files at their runtime paths.

    data_utils.py loads these directly via Dataset.from_file(),
    not through the standard HF cache.
    """
    print(f"PPL arrow root: {DATA_UTILS_ROOT}\n")
    ok = True
    for label, pattern in PPL_ARROW_PATTERNS:
        matches = glob.glob(os.path.join(DATA_UTILS_ROOT, pattern), recursive=True)
        if matches:
            print(f"  OK: {label} ({len(matches)} file(s))")
        else:
            print(f"  MISSING: {label} — no files matching {pattern}")
            ok = False
    return ok


def run_check(cache_root):
    """Check all dataset caches for corruption. Returns list of corrupt tasks."""
    print("=== PPL / calibration datasets (direct arrow files) ===")
    ppl_ok = check_ppl_arrows()

    print(f"\n=== lm-eval benchmark datasets (HF cache: {cache_root}) ===\n")
    corrupt = []
    for path, name, kwargs in LM_EVAL_TASKS:
        label = _task_label(path, name, kwargs)
        if kwargs.get("streaming"):
            print(f"  SKIP (streaming): {label}")
            continue
        status, detail = check_cache(cache_root, path, name, kwargs)
        if status == 'ok':
            print(f"  OK: {label}")
        elif status == 'missing':
            print(f"  MISSING: {label}")
            corrupt.append((path, name, kwargs))
        else:
            print(f"  CORRUPT: {label} — {detail}")
            corrupt.append((path, name, kwargs))

    if not ppl_ok:
        print("\nPPL datasets have missing arrow files.")
        print(f"  Download them to {DATA_UTILS_ROOT} (see data_utils.py)")
    return corrupt


def run_fix(cache_root):
    """Check caches, purge corrupt ones, and re-download."""
    print("=== Checking for corruption ===")
    corrupt = run_check(cache_root)

    if not corrupt:
        print("\nAll caches OK. Nothing to fix.")
        return

    print(f"\n=== Fixing {len(corrupt)} dataset(s) ===\n")
    fixed = 0
    for path, name, kwargs in corrupt:
        label = _task_label(path, name, kwargs)
        purged = purge_cache(cache_root, path, name)
        if purged:
            print(f"  Purged cache: {label}")
        if download_task(path, name, kwargs):
            fixed += 1

    failed = len(corrupt) - fixed
    print(f"\n=== Fixed {fixed}/{len(corrupt)} dataset(s)" +
          (f", {failed} still failing" if failed else "") + " ===")


def run_download(cache_root):
    """Original download-all behavior."""
    print(f"Cache dir: {cache_root}")

    print("\n=== PPL / calibration datasets ===")
    for path, name, kwargs in PPL_TASKS:
        download_task(path, name, kwargs)

    print("\n=== lm-eval benchmark datasets ===")
    for path, name, kwargs in LM_EVAL_TASKS:
        download_task(path, name, kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Download and verify HF dataset caches')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true',
                       help='Check caches for corruption without downloading')
    group.add_argument('--fix', action='store_true',
                       help='Purge corrupt caches and re-download them')
    args = parser.parse_args()

    cache_root = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
    )

    if args.check:
        corrupt = run_check(cache_root)
        if corrupt:
            print(f"\n{len(corrupt)} dataset(s) need fixing. Run with --fix to repair.")
    elif args.fix:
        run_fix(cache_root)
    else:
        run_download(cache_root)


if __name__ == "__main__":
    main()
