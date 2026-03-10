"""
Persistent JSON-based result cache for evaluation runs.

Each mode (baseline / quarot / dart) gets its own cache file at:
    data/cached_results/<mode>_results.pb

The file stores a flat dict whose keys describe individual results:
    ppl/<dataset>          -> float   (perplexity value)
    lm_eval/<task>/acc     -> float   (accuracy, 0-100)
    lm_eval/acc_avg        -> float   (average accuracy)

Usage:
    cache = ResultCache("data/cached_results/quarot_results.pb")
    if not cache.has("ppl/wikitext2"):
        ppl = run_ppl_eval(...)
        cache.set("ppl/wikitext2", ppl)
    cache.save()
"""

import json
import logging
import os


class ResultCache:
    def __init__(self, path: str, overwrite: bool = False):
        self.path = path
        self.data: dict = {}
        if not overwrite and os.path.isfile(path):
            try:
                with open(path, 'r') as f:
                    self.data = json.load(f)
                logging.info(f"Loaded {len(self.data)} cached results from {path}")
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not read cache {path}: {e}. Starting fresh.")
                self.data = {}
        else:
            if overwrite and os.path.isfile(path):
                logging.info(f"Overwrite mode: ignoring existing cache at {path}")

    def has(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value

    def save(self):
        """Persist current results to disk (atomic write)."""
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)
        logging.info(f"Cache saved to {self.path} ({len(self.data)} entries)")

    def keys(self):
        return self.data.keys()
