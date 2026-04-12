#!/usr/bin/env python3
"""
Visualize DartQuant experiment results from cached .pb files.

Usage:
    python show_results.py                       # summary table (all runs, with FP16 baseline)
    python show_results.py --nbl                 # summary table, no FP16 baseline
    python show_results.py data/cached_results/quarot*.pb  # only matching files
"""

import argparse
import glob
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'cached_results')


# ── Palette (ANSI 256-color) ────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[38;5;114m"
YELLOW  = "\033[38;5;221m"
RED     = "\033[38;5;203m"
CYAN    = "\033[38;5;75m"
MAGENTA = "\033[38;5;176m"
GRAY    = "\033[38;5;245m"
WHITE   = "\033[38;5;255m"
BG_HDR  = "\033[48;5;236m"


# Smooth gradient from best (green) to worst (red), 7 ANSI-256 color stops.
_GRADIENT = [114, 150, 186, 221, 215, 209, 203]


def _gradient_color(t):
    """Return an ANSI-256 color code for t in [0, 1] (0=best, 1=worst)."""
    t = max(0.0, min(1.0, t))
    idx = t * (len(_GRADIENT) - 1)
    return _GRADIENT[min(round(idx), len(_GRADIENT) - 1)]


def _color_delta(delta, is_ppl_col):
    """Return a colored string for a delta value."""
    if is_ppl_col and abs(delta) > 999:
        return f"{RED}{BOLD}{'INVALID':>8}{RESET}"
    sign = "+" if delta >= 0 else ""
    s = f"{sign}{delta:.2f}"
    if is_ppl_col:
        t = max(0.0, min(1.0, delta / 2.0)) if delta >= 0 else 0.0
        t_good = max(0.0, min(1.0, -delta / 2.0)) if delta < 0 else 0.0
    else:
        t = max(0.0, min(1.0, -delta / 0.05)) if delta <= 0 else 0.0
        t_good = max(0.0, min(1.0, delta / 0.05)) if delta > 0 else 0.0
    if abs(delta) < 1e-4:
        clr = GRAY
    elif t > 0:
        clr = f"\033[38;5;{_gradient_color(t)}m"
    else:
        clr = f"\033[38;5;{_gradient_color(0.0)}m" if t_good > 0.5 else f"\033[38;5;{_GRADIENT[2]}m"
    return f"{clr}{s:>8}{RESET}"


def color_val(val, best, worst, fmt=".2f", is_ppl=False):
    """Color a numeric value on a smooth gradient from best (green) to worst (red)."""
    if val is None:
        return f"{GRAY}{'—':>8}{RESET}"
    if is_ppl and val > 999:
        return f"{RED}{BOLD}{'INVALID':>8}{RESET}"
    s = f"{val:{fmt}}"
    span = abs(worst - best) if (best is not None and worst is not None) else 0
    if span < 1e-9:
        return f"{GREEN}{BOLD}{s:>8}{RESET}"
    # t=0 is best, t=1 is worst
    if is_ppl:  # lower is better
        t = (val - best) / span
    else:       # higher is better
        t = (best - val) / span
    c = _gradient_color(t)
    bold = BOLD if t < 0.05 else ""
    return f"\033[38;5;{c}m{bold}{s:>8}{RESET}"


_MODE_COLORS = {"full": CYAN, "baseline": YELLOW, "quarot": MAGENTA, "dart": GREEN}

# Metric keyword aliases for --draw (lowercase key → column header)
_METRIC_ALIASES = {
    "wikitext": "PPL↓ wikitext2", "wikitext2": "PPL↓ wikitext2", "wiki": "PPL↓ wikitext2",
    "ptb": "PPL↓ ptb",
    "c4": "PPL↓ c4",
    "piqa": "Piqa",
    "hellaswag": "Hellaswag", "hs": "Hellaswag", "ws": "Hellaswag",
    "arce": "Arc Easy", "arc_easy": "Arc Easy",
    "arcc": "Arc Challenge", "arc_challenge": "Arc Challenge",
    "winogrande": "Winogrande", "wino": "Winogrande",
    "lambada": "Lambada Openai",
    "siqa": "Social Iqa", "social_iqa": "Social Iqa",
    "obqa": "Openbookqa", "openbookqa": "Openbookqa",
    "mmlu": "MMLU",
    "avg": "Avg↑",
}


def _print_delta_table(title, rows, matrix, labels, runs, cols, is_ppl, col_w,
                       label_w=None):
    """Print a delta table.

    *rows* is a list of (run_idx, baseline_idx) pairs.
    """
    if not rows:
        return
    if label_w is None:
        label_w = max(len(labels[r]) for r, _ in rows) + 2
    label_w = max(label_w, len("Run") + 2)
    n_cols = len(cols)
    widths = [label_w] + [col_w] * n_cols

    print()
    print(f"  {BOLD}{title}{RESET}")
    print("┌" + "┬".join("─" * w for w in widths) + "┐")
    hdr = f"│{BG_HDR}{BOLD}{WHITE}{'Run':<{label_w}}{RESET}│"
    for c in cols:
        hdr += f"{BG_HDR}{BOLD}{WHITE}{c[:col_w-2]:>{col_w-1}} {RESET}│"
    print(hdr)
    print(hline(widths))

    for run_idx, bl_idx in rows:
        mode = runs[run_idx][2]
        lbl_color = _MODE_COLORS.get(mode, WHITE)
        row_str = f"│{lbl_color}{BOLD}{labels[run_idx]:<{label_w}}{RESET}│"
        for c in range(n_cols):
            val, bl_val = matrix[run_idx][c], matrix[bl_idx][c]
            if val is not None and bl_val is not None:
                row_str += f"{_color_delta(val - bl_val, is_ppl[c])}  │"
            else:
                row_str += f"{GRAY}{'—':>8}{RESET}  │"
        print(row_str)

    print("└" + "┴".join("─" * w for w in widths) + "┘")


def _print_compare(runs, matrix, labels, cols, is_ppl, col_w, compare_expr,
                   label_w=None):
    """Print comparison tables grouping runs by a key token.

    Three modes, inferred from the compare value:

    * **Numeric suffix** (``g128``, ``smq16``): key = alpha prefix, groups by
      all tokens sharing that prefix (``g32``, ``g64``, ``g128``).
    * **Hyphenated variant** (``imitate-Q4-K-S``): key = everything up to the
      last ``-``, groups by the suffix (``S``, ``M``, ...).
    * **Binary** (``wAsym``, ``pwl``): presence vs absence. Tokens that
      exclusively co-occur with the compare token are auto-stripped for pairing.
    """
    from collections import defaultdict

    cmp_lower = compare_expr.lower()

    # ── Parse compare value ────────────────────────────────────────────
    m_numeric = re.match(r'^([a-zA-Z_-]+?)(\d+)$', compare_expr)
    if m_numeric:
        # Numeric suffix: g128 → key prefix "g", pattern g\d+
        key_prefix = m_numeric.group(1).lower()
        key_pattern = re.compile(f'^{re.escape(key_prefix)}\\d+$', re.IGNORECASE)
        mode = "numeric"
    elif '-' in compare_expr:
        # Hyphenated variant: imitate-Q4-K-S → prefix "imitate-Q4-K-"
        last_dash = compare_expr.rfind('-')
        key_prefix = compare_expr[:last_dash + 1].lower()
        key_pattern = re.compile(f'^{re.escape(key_prefix)}.+$', re.IGNORECASE)
        mode = "variant"
    else:
        # Binary: wAsym, pwl → presence vs absence
        key_pattern = re.compile(f'^{re.escape(compare_expr)}$', re.IGNORECASE)
        mode = "binary"

    # ── Helpers ────────────────────────────────────────────────────────
    def _norm_base(path):
        return _normalize_bk(
            os.path.basename(path).replace("_results.pb", "").lower())

    def _get_key_token(path):
        for t in _norm_base(path).split('_'):
            if key_pattern.match(t):
                return t
        return None

    # ── Group non-full runs by key value ───────────────────────────────
    groups = defaultdict(list)  # {key_value_or_None: [run_idx]}
    for i, (label, data, rmode, path) in enumerate(runs):
        if rmode == "full":
            continue
        token = _get_key_token(path)
        if mode == "binary":
            groups[cmp_lower if token else None].append(i)
        else:
            if token is not None:
                groups[token].append(i)

    if cmp_lower not in groups:
        print(f"\n  {YELLOW}Compare: no runs found with baseline value "
              f"'{compare_expr}'{RESET}")
        avail = [k for k in groups if k is not None]
        if avail:
            print(f"  {DIM}Available values: {', '.join(sorted(avail))}{RESET}")
        return None

    # ── Build strip patterns for pairing key ───────────────────────────
    # Always strip the key token.  In binary mode, also strip tokens that
    # exclusively co-occur with the baseline group (e.g. "wrap" always
    # accompanies "wAsym").
    strip_patterns = [key_pattern]

    if mode == "binary":
        baseline_indices = groups[cmp_lower]
        other_indices = [idx for g, idxs in groups.items()
                         if g != cmp_lower for idx in idxs]
        if baseline_indices and other_indices:
            # Tokens present in ALL baseline runs but NONE of the other runs
            bl_token_sets = [set(_norm_base(runs[i][3]).split('_'))
                             for i in baseline_indices]
            ot_token_sets = [set(_norm_base(runs[i][3]).split('_'))
                             for i in other_indices]
            bl_common = bl_token_sets[0]
            for ts in bl_token_sets[1:]:
                bl_common &= ts
            ot_all = set()
            for ts in ot_token_sets:
                ot_all |= ts
            exclusive = bl_common - ot_all
            for tok in exclusive:
                strip_patterns.append(
                    re.compile(f'^{re.escape(tok)}$', re.IGNORECASE))

    def _pair_key(path):
        tokens = _norm_base(path).split('_')
        return '_'.join(t for t in tokens
                        if not any(p.match(t) for p in strip_patterns))

    baseline_by_key = {_pair_key(runs[idx][3]): idx
                       for idx in groups[cmp_lower]}

    # ── Collect pairs grouped by base config ───────────────────────────
    config_groups = defaultdict(list)
    for gval in sorted(groups, key=lambda v: (v is None, v or "")):
        if gval == cmp_lower:
            continue
        key_label = gval if gval is not None else f"no {compare_expr}"
        for idx in groups[gval]:
            pk = _pair_key(runs[idx][3])
            if pk in baseline_by_key:
                config_groups[pk].append((idx, baseline_by_key[pk], key_label))

    if not config_groups:
        print(f"\n  {YELLOW}Compare: no matching pairs found for "
              f"'{compare_expr}'{RESET}")
        for gval, indices in sorted(groups.items(),
                                    key=lambda x: (x[0] is None, x[0] or "")):
            gl = gval if gval is not None else f"no {compare_expr}"
            print(f"  {DIM}{gl}: {len(indices)} runs{RESET}")
        return None

    # ── Print one sub-table per base config & build return data ────────
    def _pad_key(kv):
        return re.sub(r'(\d+)', lambda m: m.group(1).zfill(4), kv)

    sorted_configs = sorted(config_groups.items(),
                            key=lambda item: labels[item[1][0][1]])

    # Build key_values list with display names (convert None → "no <expr>")
    no_label = f"no {compare_expr}"
    all_key_values = sorted(groups.keys(), key=lambda v: (v is None, _pad_key(v or "")))
    display_key_values = [no_label if v is None else v for v in all_key_values]
    result_configs = []

    for pk, cfg_pairs in sorted_configs:
        cfg_pairs.sort(key=lambda x: _pad_key(x[2]))
        bl_label = labels[cfg_pairs[0][1]]
        bl_idx = cfg_pairs[0][1]
        delta_rows = [(idx, bl_idx) for idx, bl_idx, _ in cfg_pairs]
        _print_delta_table(
            f"Δ vs {compare_expr}  {DIM}({bl_label})",
            delta_rows, matrix, labels, runs, cols, is_ppl, col_w,
            label_w=label_w)
        result_configs.append({
            "baseline_label": bl_label,
            "baseline_idx": bl_idx,
            "pair_key": pk,
            "pairs": [{"key_label": kl, "run_idx": ri, "bl_idx": bi}
                       for ri, bi, kl in cfg_pairs],
        })

    return {
        "baseline_value": cmp_lower,
        "key_values": display_key_values,
        "configs": result_configs,
        "mode": mode,
    }


def _wrap_label(s, max_chars=15):
    """Wrap a label string into multiple lines at word boundaries."""
    words = s.replace("_", " ").split()
    lines, line = [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        lines.append(line)
    return "\n".join(lines)


def _collect_metric_values(configs, matrix, col_idx):
    """Collect all non-None metric values across configs for y-limit computation."""
    vals = []
    for cfg in configs:
        v = matrix[cfg["baseline_idx"]][col_idx]
        if v is not None:
            vals.append(v)
        for pair in cfg["pairs"]:
            v = matrix[pair["run_idx"]][col_idx]
            if v is not None:
                vals.append(v)
    return vals


def _set_focused_ylim(ax, vals, ppl_col):
    """Set y-axis limits focused on the actual data range."""
    if not vals:
        return
    vmin, vmax = min(vals), max(vals)
    margin = max((vmax - vmin) * 0.3, 0.01)
    if ppl_col:
        ax.set_ylim(vmax + margin, vmin - margin)
    else:
        ax.set_ylim(vmin - margin, vmax + margin)


def _make_title(col_header, baseline_value, common_sub):
    """Build a chart title, appending common subtitle if present."""
    title = f"{col_header}  (compare: {baseline_value})"
    if common_sub:
        title += f"\n{common_sub}"
    return title


def _save_fig(fig, fig_dir, metric_kw, baseline_value, chart_type):
    """Save figure to disk and print path."""
    fname = f"{metric_kw}_{baseline_value}_{chart_type}.png".replace("/", "_")
    path = os.path.join(fig_dir, fname)
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  {DIM}Saved {path}{RESET}")


def _draw_bar_chart(configs, matrix, col_idx, ppl_col, key_values,
                    baseline_value, key_colors, col_header, common_sub,
                    short_labels, fig_dir, metric_kw):
    """Draw and save a clustered bar chart for one metric."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(configs) * 1.5 + 2), 5))
    n_keys = len(key_values)
    bar_width = 0.8 / n_keys

    for ci, cfg in enumerate(configs):
        bl_val = matrix[cfg["baseline_idx"]][col_idx]
        if bl_val is not None:
            ax.bar(ci + key_values.index(baseline_value) * bar_width,
                   bl_val, bar_width, color=key_colors[baseline_value],
                   label=baseline_value if ci == 0 else "")
        for pair in cfg["pairs"]:
            kl = pair["key_label"]
            val = matrix[pair["run_idx"]][col_idx]
            if val is not None and kl in key_colors:
                ki = key_values.index(kl)
                ax.bar(ci + ki * bar_width, val, bar_width,
                       color=key_colors[kl],
                       label=kl if ci == 0 else "")

    _set_focused_ylim(ax, _collect_metric_values(configs, matrix, col_idx),
                      ppl_col)

    wrapped = [_wrap_label(l) for l in short_labels]
    ax.set_xticks([ci + bar_width * (n_keys - 1) / 2
                   for ci in range(len(configs))])
    ax.set_xticklabels(wrapped, rotation=0, ha="center", fontsize=7)
    ax.set_ylabel(col_header)
    ax.set_title(_make_title(col_header, baseline_value, common_sub),
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, fig_dir, metric_kw, baseline_value, "bar")


def _draw_line_chart(configs, matrix, col_idx, ppl_col, key_values,
                     baseline_value, col_header, common_sub, short_labels,
                     fig_dir, metric_kw):
    """Draw and save a line chart for one metric (numeric compare, ≥3 values)."""
    import matplotlib.pyplot as plt

    x_nums = []
    for kv in key_values:
        m = re.search(r'(\d+)', kv)
        x_nums.append(int(m.group(1)) if m else 0)

    fig, ax = plt.subplots(figsize=(max(6, len(key_values) + 2), 5))
    cmap = plt.cm.get_cmap("tab10", max(len(configs), 3))

    for ci, cfg in enumerate(configs):
        y_vals = [None] * len(key_values)
        bl_ki = key_values.index(baseline_value)
        y_vals[bl_ki] = matrix[cfg["baseline_idx"]][col_idx]
        for pair in cfg["pairs"]:
            if pair["key_label"] in key_values:
                ki = key_values.index(pair["key_label"])
                y_vals[ki] = matrix[pair["run_idx"]][col_idx]
        xy = [(x, y) for x, y in zip(x_nums, y_vals) if y is not None]
        if len(xy) >= 2:
            xs, ys = zip(*xy)
            lbl = short_labels[ci] if ci < len(short_labels) else cfg["baseline_label"]
            ax.plot(xs, ys, marker='o', label=lbl, color=cmap(ci))

    ax.set_xlabel(re.sub(r'\d+', '*', baseline_value))
    ax.set_ylabel(col_header)
    ax.set_title(_make_title(col_header, baseline_value, common_sub),
                 fontsize=10)
    ax.set_xticks(x_nums)
    ax.set_xticklabels(key_values, fontsize=8)
    if ppl_col:
        ax.invert_yaxis()
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save_fig(fig, fig_dir, metric_kw, baseline_value, "line")


def _draw_figures(compare_data, runs, matrix, labels, cols, is_ppl,
                  draw_metrics):
    """Generate bar charts (and line charts for numeric keys) from compare data."""
    import matplotlib
    matplotlib.use("Agg")

    fig_dir = os.path.join(SCRIPT_DIR, "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    baseline_value = compare_data["baseline_value"]
    key_values = compare_data["key_values"]
    configs = compare_data["configs"]
    cmp_mode = compare_data["mode"]

    # Resolve metric keywords to column indices
    metric_indices = []
    for kw in draw_metrics:
        col_header = _METRIC_ALIASES.get(kw.lower())
        if col_header is None:
            print(f"  {YELLOW}--draw: unknown metric '{kw}'. "
                  f"Available: {', '.join(sorted(_METRIC_ALIASES.keys()))}{RESET}")
            continue
        try:
            col_idx = cols.index(col_header)
        except ValueError:
            print(f"  {YELLOW}--draw: column '{col_header}' not found in table{RESET}")
            continue
        metric_indices.append((kw, col_header, col_idx))

    if not metric_indices:
        return

    # Build a color map for key values
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("tab10", max(len(key_values), 3))
    key_colors = {kv: cmap(i) for i, kv in enumerate(key_values)}

    # Factor out common tags from config labels
    config_labels = [cfg["baseline_label"] for cfg in configs]
    common_sub, short_labels = _factor_labels(config_labels)

    for metric_kw, col_header, col_idx in metric_indices:
        ppl_col = is_ppl[col_idx]

        _draw_bar_chart(configs, matrix, col_idx, ppl_col, key_values,
                        baseline_value, key_colors, col_header, common_sub,
                        short_labels, fig_dir, metric_kw)

        if cmp_mode == "numeric" and len(key_values) >= 3:
            _draw_line_chart(configs, matrix, col_idx, ppl_col, key_values,
                             baseline_value, col_header, common_sub,
                             short_labels, fig_dir, metric_kw)


def _extract_model_name(path):
    """Extract model name from a result .pb filename.

    E.g. 'baseline_Llama-3.2-1B-Instruct_w4a8k8v8_..._results.pb'
    -> 'Llama-3.2-1B-Instruct'
    """
    base = os.path.basename(path).replace("_results.pb", "")
    base = re.sub(r'^static_', '', base)
    # Strip mode prefix
    base = re.sub(r'^(full|baseline|quarot|dart)_', '', base)
    # Model name is everything before the first _w\d (quant tag start)
    m = re.match(r'^(.+?)_w\d', base)
    return m.group(1) if m else ""


def _extract_gguf_tag(path):
    """Extract GGUF tag (e.g. 'gguf-Q4-K-M') from a result filename, or None."""
    base = os.path.basename(path)
    m = re.search(r'(gguf-[A-Za-z0-9-]+)', base)
    return m.group(1) if m else None


def _extract_imitate_tag(path):
    """Extract imitate tag (e.g. 'imitate-Q4-K-M') from a result filename,
    and return the corresponding gguf tag (e.g. 'gguf-Q4-K-M'), or None."""
    base = os.path.basename(path)
    m = re.search(r'imitate-([A-Za-z0-9-]+)', base)
    return f"gguf-{m.group(1)}" if m else None


def _normalize_bk(tag):
    """Normalize the block-size (bk) token relative to group-size (g).

    - If bk equals g, omit it (redundant).
    - If bk is absent but g is present, insert bk32 (the hardware default).
    """
    g_match = re.search(r'(?:^|_)(g(\d+))(?:_|$)', tag)
    if not g_match:
        return tag
    g_val = g_match.group(2)
    bk_match = re.search(r'(?:^|_)(bk(\d+))(?:_|$)', tag)
    if bk_match:
        if bk_match.group(2) == g_val:
            # bk == g -> redundant, remove it
            tag = tag.replace(bk_match.group(1), '').replace('__', '_').strip('_')
        # else bk != g -> keep it (interesting case)
    else:
        # No bk token -> insert default bk32 after the g token (only for intgemm runs)
        if g_val != "32" and 'intgemm' in tag:
            tag = tag.replace(g_match.group(1), f'{g_match.group(1)}_bk32')
    return tag


def parse_filename(path):
    """Extract mode and quant config from filename."""
    base = os.path.basename(path).replace("_results.pb", "")

    is_static = "static" in base
    base = base.replace("static_", "") if is_static else base

    # e.g. "quarot_w4a8k4v4_g128_aAsym_wSym_kSym_vSym"
    # or legacy "baseline_results.pb" -> "baseline"
    parts = base.split("_", 1)

    mode = parts[0]
    tag = parts[1] if len(parts) > 1 else ""
    tag = _normalize_bk(tag)

    if is_static:
        tag += ", static"

    return mode, tag


def short_label(mode, tag):
    """Build a compact display label from mode + quant tag."""
    if not tag:
        return mode
    # Extract key numbers: w4a8k4v4_g128_...
    m = re.match(r"w(\d+)a(\d+)k(\d+)v(\d+)_g(\d+)_(.+)", tag)
    if m:
        w, a, k, v, g, sym_part = m.groups()
        sym = "sym" if "wSym" in sym_part else "asym"
        label = f"{mode} W{w}A{a}K{k}V{v} g{g} {sym}"
        return label
    # Legacy format: w8a8k8v8_wg128_kg128_vg128_...
    m2 = re.match(r"w(\d+)a(\d+)k(\d+)v(\d+)_wg(\d+)", tag)
    if m2:
        w, a, k, v, g = m2.groups()
        sym = "sym" if "wSym" in tag else "asym"
        return f"{mode} W{w}A{a}K{k}V{v} g{g} {sym}"
    return f"{mode} {tag}"


def _factor_labels(labels):
    """Split labels into (common_subtitle, short_labels).

    Tokenizes each label on whitespace/underscores/commas, finds tokens that
    appear in ALL labels (multiset intersection), factors those out as a
    subtitle, and returns only the differing tokens per row.
    """
    from collections import Counter

    if len(labels) <= 1:
        return "", labels

    tokenized = [re.split(r'[\s_,]+', l) for l in labels]
    tokenized = [[t for t in tokens if t] for tokens in tokenized]

    # Multiset intersection: tokens present in every label
    common = None
    for tokens in tokenized:
        c = Counter(tokens)
        common = (common & c) if common is not None else c

    if not common:
        return "", labels

    # Build subtitle using the first label's token order
    remaining = +common
    subtitle_parts = []
    for t in tokenized[0]:
        if remaining[t] > 0:
            subtitle_parts.append(t)
            remaining[t] -= 1

    # Build short labels: keep only the non-common tokens, in original order
    short = []
    for i, tokens in enumerate(tokenized):
        remaining = +common
        unique = []
        for t in tokens:
            if remaining[t] > 0:
                remaining[t] -= 1
            else:
                unique.append(t)
        short.append(" ".join(unique) if unique else "")

    return " ".join(subtitle_parts), short


def load_results(paths):
    """Load all .pb files, returns list of (label, data, mode, path) tuples.
    Sorted: FP full first, then GGUF full, then alphabetically by label."""
    runs = []
    for p in sorted(paths):
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        mode, tag = parse_filename(p)
        label = short_label(mode, tag)

        runs.append((label, data, mode, p))
    # Sort: FP full (no gguf) first, then GGUF full, then the rest
    def _pad_g(s):
        """Zero-pad gXX -> g0XX so g32 sorts before g128."""
        return re.sub(r'g(\d+)', lambda m: f'g{int(m.group(1)):04d}', s)

    def _sort_key(r):
        _, _, mode, path = r
        gguf_tag = _extract_gguf_tag(path)
        if mode == "full" and not gguf_tag:
            return (0, _pad_g(r[0]))
        if mode == "full" and gguf_tag:
            return (1, _pad_g(gguf_tag))
        return (2, _pad_g(r[0]))
    runs.sort(key=_sort_key)
    return runs


def hline(widths, char="─", left="├", mid="┼", right="┤"):
    return left + mid.join(char * w for w in widths) + right


def print_summary(runs, show_delta=False, compare_expr=None, draw_metrics=None,
                   fast=False):
    """Print a pretty summary table."""
    if not runs:
        print("No result files found.")
        return

    # ── Collect metrics ─────────────────────────────────────────────────
    ppl_datasets = ["wikitext2", "ptb", "c4"]
    lm_tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge",
                "winogrande", "lambada_openai", "social_iqa", "openbookqa"]
    mmlu_agg = ["mmlu"]
    summary_metric = "acc_avg"

    # All column headers
    cols = []
    cols += [f"PPL↓ {d}" for d in ppl_datasets]
    cols += [t.replace("_", " ").title() for t in lm_tasks]
    cols += ["MMLU"]
    cols += ["Avg↑"]

    # Build data matrix — factor out common prefix from labels
    # Exclude "full" runs from factoring so they don't dilute the common prefix
    full_indices = {i for i, r in enumerate(runs) if r[2] == "full"}
    non_full_labels = [r[0] for i, r in enumerate(runs) if i not in full_indices]
    common_prefix, short_non_full = _factor_labels(non_full_labels)
    # Reconstruct labels list with full runs keeping their original label
    labels = []
    nf_idx = 0
    for i, (raw_label, _, mode, path) in enumerate(runs):
        if i in full_indices:
            gguf_tag = _extract_gguf_tag(path)
            if gguf_tag:
                # e.g. "gguf-Q4-K-M" -> "GGUF Q4-K-M baseline"
                labels.append(f"GGUF {gguf_tag.replace('gguf-', '')} baseline")
            else:
                labels.append("FP16 baseline")
        else:
            labels.append(short_non_full[nf_idx])
            nf_idx += 1
    matrix = []  # matrix[run_idx][col_idx] = value or None
    is_ppl = []  # True for PPL columns

    for label, data, mode, _path in runs:
        row = []
        for d in ppl_datasets:
            row.append(data.get(f"ppl/{d}"))
            if len(is_ppl) < len(ppl_datasets):
                is_ppl.append(True)
        for t in lm_tasks:
            row.append(data.get(f"lm_eval/{t}/acc"))
            if len(is_ppl) < len(ppl_datasets) + len(lm_tasks):
                is_ppl.append(False)
        for t in mmlu_agg:
            row.append(data.get(f"lm_eval/{t}/acc"))
            if len(is_ppl) < len(ppl_datasets) + len(lm_tasks) + len(mmlu_agg):
                is_ppl.append(False)
        # Compute Avg from the displayed lm_eval columns (not cached acc_avg)
        lm_start = len(ppl_datasets)
        lm_end = lm_start + len(lm_tasks) + len(mmlu_agg)
        lm_vals = [v for v in row[lm_start:lm_end] if v is not None]
        avg_val = round(sum(lm_vals) / len(lm_vals), 2) if lm_vals else None
        row.append(avg_val)
        if len(is_ppl) < len(cols):
            is_ppl.append(False)
        matrix.append(row)

    # --fast: keep only PPL columns
    if fast:
        n_ppl = len(ppl_datasets)
        cols = cols[:n_ppl]
        is_ppl = is_ppl[:n_ppl]
        matrix = [row[:n_ppl] for row in matrix]

    n_cols = len(cols)
    n_runs = len(runs)

    # Find best/worst per column (for PPL columns, exclude values with
    # delta > 10 from the FP16 baseline so outliers don't squash the scale)
    full_idx = None
    for i, (_, _, mode, path) in enumerate(runs):
        if mode == "full" and not _extract_gguf_tag(path):
            full_idx = i
            break

    bests = []
    worsts = []
    for c in range(n_cols):
        vals = [matrix[r][c] for r in range(n_runs) if matrix[r][c] is not None]
        if is_ppl[c] and full_idx is not None and matrix[full_idx][c] is not None:
            bl_val = matrix[full_idx][c]
            vals = [v for v in vals if abs(v - bl_val) <= 10]
        if not vals:
            bests.append(None)
            worsts.append(None)
        elif is_ppl[c]:
            b, w = min(vals), max(vals)
            # Clamp PPL color span: 0.2–1.0 points per gradient color
            n_colors = len(_GRADIENT)
            min_span = 0.2 * n_colors
            max_span = 1.0 * n_colors
            span = w - b
            mid = (b + w) / 2
            if span < min_span:
                b, w = mid - min_span / 2, mid + min_span / 2
            elif span > max_span:
                b, w = mid - max_span / 2, mid + max_span / 2
            bests.append(b)
            worsts.append(w)
        else:
            bests.append(max(vals))
            worsts.append(min(vals))

    # ── Render table ────────────────────────────────────────────────────
    label_w = max(len(l) for l in labels) + 2
    col_w = 10
    widths = [label_w] + [col_w] * n_cols

    # Title
    print()
    print(f"  {BOLD}{CYAN}DartQuant Experiment Results{RESET}")
    if common_prefix:
        print(f"  {WHITE}{common_prefix}{RESET}  {DIM}({len(runs)} runs){RESET}")
    else:
        print(f"  {DIM}{len(runs)} runs found{RESET}")
    print()

    # Top border
    print("┌" + "┬".join("─" * w for w in widths) + "┐")

    # Header row
    hdr = f"│{BG_HDR}{BOLD}{WHITE}{'Run':<{label_w}}{RESET}│"
    for i, c in enumerate(cols):
        short = c[:col_w-2]
        hdr += f"{BG_HDR}{BOLD}{WHITE}{short:>{col_w-1}} {RESET}│"
    print(hdr)

    # Header separator
    print(hline(widths))

    # Data rows
    for r in range(n_runs):
        label = labels[r]
        mode = runs[r][2]

        lbl_color = _MODE_COLORS.get(mode, WHITE)
        row_str = f"│{lbl_color}{BOLD}{label:<{label_w}}{RESET}│"

        for c in range(n_cols):
            val = matrix[r][c]
            colored = color_val(val, bests[c], worsts[c], fmt=".2f", is_ppl=is_ppl[c])
            row_str += f"{colored}  │"
        print(row_str)

        # Separator between runs (not after last)
        if r < n_runs - 1:
            next_mode = runs[r + 1][2]
            # Thick divider after the last full-precision baseline row
            if mode == "full" and next_mode != "full":
                print(hline(widths, char="━", left="┠", mid="╋", right="┨"))
            elif mode == "full" and next_mode == "full":
                # Thin divider between baselines (FP16 vs GGUF)
                print(hline(widths, char="─", left="├", mid="┼", right="┤"))
            else:
                print(hline(widths, char="┄", left="├", mid="┼", right="┤"))

    # Bottom border
    print("└" + "┴".join("─" * w for w in widths) + "┘")

    # ── Delta vs FP16 baseline ──────────────────────────────────────────
    if show_delta and full_idx is not None and n_runs > 1:
        delta_rows = [(r, full_idx) for r in range(n_runs) if r != full_idx]
        _print_delta_table("Δ vs FP16", delta_rows, matrix, labels, runs,
                           cols, is_ppl, col_w, label_w=label_w)

    if compare_expr:
        compare_data = _print_compare(runs, matrix, labels, cols, is_ppl, col_w,
                                      compare_expr, label_w=label_w)
        if draw_metrics and compare_data:
            _draw_figures(compare_data, runs, matrix, labels, cols, is_ppl,
                          draw_metrics)

    # Legend
    print()
    grad_bar = "".join(f"\033[38;5;{c}m█{RESET}" for c in _GRADIENT)
    print(f"  {grad_bar}  Best → Worst   {GRAY}—{RESET} Missing")
    print(f"  PPL↓ = lower is better   Acc↑ = higher is better")
    print()


def _find_best_baseline(model_name, matched_paths):
    """Find the full-precision baseline(s) for a model that best match the
    existing result set.  Scores each candidate by how many filename tokens
    it shares with the matched files."""
    from collections import Counter

    candidates = glob.glob(os.path.join(RESULTS_DIR, f"full_{model_name}_*_results.pb"))
    if not candidates:
        return []



    # Tokenize matched paths (non-full only) to build a reference bag
    ref_tokens = Counter()
    for p in matched_paths:
        base = os.path.basename(p).replace("_results.pb", "")
        ref_tokens.update(re.split(r'[\s_,]+', base))

    # Score each candidate by token overlap with the reference set
    scored = []
    for c in candidates:
        base = os.path.basename(c).replace("_results.pb", "")
        tokens = set(re.split(r'[\s_,]+', base))
        score = sum(ref_tokens[t] for t in tokens)
        scored.append((score, c))

    scored.sort(reverse=True)
    # Return the best-matching baseline; if there are ties (e.g. with/without
    # pwl), include all that share the top score
    best_score = scored[0][0]          
    return [os.path.join(RESULTS_DIR, f"full_{model_name}_w16a16k16v16_g128_aAsym_kAsym_vAsym_kvex8_projex15_results.pb")]
    return [c for s, c in scored if s == best_score]


def _split_respecting_brackets(s, delim):
    """Split string on *delim*, but not inside ``[...]`` brackets."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == delim and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return parts


def _match_filter(filename, expr):
    """Test whether *filename* matches a filter expression.

    Syntax (case-insensitive, matched against basename):
        keyword          substring match
        a*b              glob-style wildcard within a term
        a,b              AND  (all terms must match)
        a|b              NOT  (b must NOT match)
        [a,b|c]          OR-group  (at least one condition must hold)

    Inside brackets, ``,`` means OR and ``|`` means OR-NOT:
        [t2int32,t2int24|t2]  →  t2int32 OR t2int24 OR (NOT t2)

    Example: "g128,[t2int32,t2int24|t2]"
        must contain 'g128' AND (contain 't2int32' OR 't2int24' OR NOT 't2')

    Multiple CLI arguments are joined with AND automatically.
    """
    import fnmatch
    name = filename.lower()

    def _term_matches(term):
        """Check if a single keyword/glob matches the filename."""
        term = term.strip()
        if not term:
            return False
        if '*' in term or '?' in term:
            return fnmatch.fnmatch(name, f'*{term}*')
        return term in name

    def _eval_bracket_group(content):
        """Evaluate ``[a,b|c]`` → True if any include matches OR any exclude
        does NOT match."""
        parts = content.split('|')
        inc_terms = [t.strip() for t in parts[0].split(',') if t.strip()]
        exc_parts = parts[1:]

        # Any include term matching is enough
        if any(_term_matches(t) for t in inc_terms):
            return True

        # Any exclude term NOT present counts as a pass
        for exc in exc_parts:
            for t in exc.split(','):
                t = t.strip()
                if t and not _term_matches(t):
                    return True

        return False

    # Split on | first (respecting brackets) to separate include/exclude
    parts = _split_respecting_brackets(expr.lower(), '|')
    include_expr = parts[0]           # everything before the first |
    exclude_terms = parts[1:]         # everything after each |

    # Include: split on commas (respecting brackets) for AND
    and_terms = [t.strip() for t in _split_respecting_brackets(include_expr, ',')
                 if t.strip()]
    if not and_terms:
        return True

    # All AND terms must match; bracket groups use OR semantics internally
    for term in and_terms:
        if term.startswith('[') and term.endswith(']'):
            if not _eval_bracket_group(term[1:-1]):
                return False
        else:
            if not _term_matches(term):
                return False

    # No exclude term may match
    for exc in exclude_terms:
        for t in exc.split(','):
            t = t.strip()
            if t and _term_matches(t):
                return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DartQuant experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python show_results.py                           # all runs + FP16 baseline
  python show_results.py dart                      # substring: *dart*
  python show_results.py dart pwl                  # AND: *dart* AND *pwl*
  python show_results.py "static,pwl"              # AND: *static* AND *pwl*
  python show_results.py "dart|static"             # NOT: *dart* but NOT *static*
  python show_results.py "static,pwl|w8"           # AND+NOT: *static* AND *pwl* but NOT *w8*
  python show_results.py "g128,[t2int32,t2int24|t2]"  # OR-group: g128 AND (t2int32 OR t2int24 OR NOT t2)
  python show_results.py "static*pwl"              # glob: *static*pwl*
  python show_results.py --nbl                     # exclude FP16 baseline
  python show_results.py -d                        # show delta vs FP16 baseline
  python show_results.py -c pwl                    # compare runs with/without 'pwl'
  python show_results.py data/cached_results/quarot*.pb  # literal paths (shell glob)""",
    )
    parser.add_argument("-d", "--delta", action="store_true",
                        help="Show delta table vs FP16 baseline")
    parser.add_argument("-c", "--compare", type=str, default=None, metavar="EXPR",
                        help="Compare runs matching EXPR vs runs not matching it (delta table)")
    parser.add_argument("--draw", type=str, default=None, metavar="METRICS",
                        help="Comma-separated metrics to plot (requires -c). "
                             "E.g.: C4,MMLU,avg")
    parser.add_argument("-F", "--fast", action="store_true",
                        help="Show only PPL columns (hide accuracy columns)")
    parser.add_argument("--nbl", "--no-baseline", action="store_true",
                        dest="no_baseline",
                        help="Exclude the FP16 full-precision baseline")
    parser.add_argument("filters", nargs="*", default=[],
                        help="Filter expressions or literal paths")
    args = parser.parse_args()

    if args.draw and not args.compare:
        parser.error("--draw requires --compare (-c)")

    draw_metrics = [m.strip() for m in args.draw.split(",")] if args.draw else None

    # ── Gather files ────────────────────────────────────────────────────
    all_results = glob.glob(os.path.join(RESULTS_DIR, "*_results.pb"))

    if not args.filters:
        paths = all_results
    else:
        # Check if filters are literal file paths or search patterns
        literal = [f for f in args.filters if os.path.isfile(f)]
        patterns = [f for f in args.filters if not os.path.isfile(f)]

        if literal and not patterns:
            paths = literal
        else:
            # Each CLI arg is an AND clause; within each, commas add more AND,
            # pipes exclude terms, and * is a glob wildcard.
            paths = []
            for p in all_results:
                base = os.path.basename(p)
                if all(_match_filter(base, pat) for pat in (patterns or args.filters)):
                    paths.append(p)

    if not paths:
        print("No result files found.")
        return

    # ── Discover models ─────────────────────────────────────────────────
    models = sorted(set(_extract_model_name(p) for p in paths) - {""})
    if models:
        print(f"\n  {DIM}Models: {', '.join(models)}{RESET}")

    # ── Auto-include best-matching FP16 baseline per model ──────────────
    if not args.no_baseline:
        path_set = set(paths)
        non_full = [p for p in paths if not os.path.basename(p).startswith("full_")]
        for model in models:
            for bl in _find_best_baseline(model, non_full or paths):
                if bl not in path_set:
                    paths.append(bl)
                    path_set.add(bl)

        # Auto-include GGUF full-precision baselines when GGUF or imitate runs are present
        gguf_tags = set()
        for p in list(paths):
            tag = _extract_gguf_tag(p)
            if tag and not os.path.basename(p).startswith("full_"):
                gguf_tags.add(tag)
            imitate_tag = _extract_imitate_tag(p)
            if imitate_tag:
                gguf_tags.add(imitate_tag)
        for model in models:
            for tag in sorted(gguf_tags):
                for candidate in glob.glob(os.path.join(RESULTS_DIR, f"full_{model}_*_{tag}_results.pb")):
                    if candidate not in path_set:
                        paths.append(candidate)
                        path_set.add(candidate)

    runs = load_results(paths)

    if args.no_baseline:
        runs = [r for r in runs if r[2] != "full"]

    print_summary(runs, show_delta=args.delta, compare_expr=args.compare,
                  draw_metrics=draw_metrics, fast=args.fast)


if __name__ == "__main__":
    main()
