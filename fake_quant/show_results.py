#!/usr/bin/env python3
"""
Visualize DartQuant experiment results from cached .pb files.

Usage:
    python show_results.py -s                    # summary table (all runs, with FP16 baseline)
    python show_results.py -s --nbl              # summary table, no FP16 baseline
    python show_results.py -s /tmp/quarot*.pb    # only matching files
"""

import argparse
import glob
import json
import os
import re
import sys


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


def color_val(val, best, worst, fmt=".2f", is_ppl=False):
    """Color a numeric value on a smooth gradient from best (green) to worst (red)."""
    if val is None:
        return f"{GRAY}{'—':>8}{RESET}"
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
        short.append(" ".join(unique) if unique else labels[i])

    return " ".join(subtitle_parts), short


def load_results(paths):
    """Load all .pb files, returns list of (label, data, mode) tuples.
    Sorted so 'full' always comes first, then alphabetically by label."""
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

        runs.append((label, data, mode))
    # Sort: full first, then alphabetically by label
    runs.sort(key=lambda r: (0 if r[2] == "full" else 1, r[0]))
    return runs


def hline(widths, char="─", left="├", mid="┼", right="┤"):
    return left + mid.join(char * w for w in widths) + right


def print_summary(runs):
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
    raw_labels = [r[0] for r in runs]
    common_prefix, labels = _factor_labels(raw_labels)
    matrix = []  # matrix[run_idx][col_idx] = value or None
    is_ppl = []  # True for PPL columns

    for label, data, mode in runs:
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

    n_cols = len(cols)
    n_runs = len(runs)

    # Find best/worst per column
    bests = []
    worsts = []
    for c in range(n_cols):
        vals = [matrix[r][c] for r in range(n_runs) if matrix[r][c] is not None]
        if not vals:
            bests.append(None)
            worsts.append(None)
        elif is_ppl[c]:
            bests.append(min(vals))
            worsts.append(max(vals))
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
 

        # Color the label by mode
        mode_colors = {"full": CYAN, "baseline": YELLOW, "quarot": MAGENTA, "dart": GREEN}
        lbl_color = mode_colors.get(mode, WHITE)
        row_str = f"│{lbl_color}{BOLD}{label:<{label_w}}{RESET}│"

        for c in range(n_cols):
            val = matrix[r][c]
            colored = color_val(val, bests[c], worsts[c], fmt=".2f", is_ppl=is_ppl[c])
            row_str += f"{colored}  │"
        print(row_str)

        # Separator between runs (not after last)
        if r < n_runs - 1:
            # Thick divider after the full-precision baseline
            if mode == "full":
                print(hline(widths, char="━", left="┠", mid="╋", right="┨"))
            else:
                print(hline(widths, char="┄", left="├", mid="┼", right="┤"))

    # Bottom border
    print("└" + "┴".join("─" * w for w in widths) + "┘")

    # ── Delta vs FP16 baseline ──────────────────────────────────────────
    full_idx = None
    for i, (_, _, mode) in enumerate(runs):
        if mode == "full":
            full_idx = i
            break

    if full_idx is not None and n_runs > 1:
        print()
        print(f"  {BOLD}Delta vs FP16 baseline:{RESET}")
        print("┌" + "┬".join("─" * w for w in widths) + "┐")
        hdr = f"│{BG_HDR}{BOLD}{WHITE}{'Δ vs FP16':<{label_w}}{RESET}│"
        for i, c in enumerate(cols):
            short = c[:col_w-2]
            hdr += f"{BG_HDR}{BOLD}{WHITE}{short:>{col_w-1}} {RESET}│"
        print(hdr)
        print(hline(widths))

        for r in range(n_runs):
            if r == full_idx:
                continue
            label = labels[r]
            mode = runs[r][2]
            lbl_color = {"full": CYAN, "baseline": YELLOW, "quarot": MAGENTA, "dart": GREEN}.get(mode, WHITE)
            row_str = f"│{lbl_color}{BOLD}{label:<{label_w}}{RESET}│"

            for c in range(n_cols):
                val = matrix[r][c]
                base_val = matrix[full_idx][c]
                if val is not None and base_val is not None:
                    delta = val - base_val
                    sign = "+" if delta >= 0 else ""
                    s = f"{sign}{delta:.2f}"
                    # Map delta to gradient: 0 = neutral (gray), large bad = red, large good = green
                    if is_ppl[c]:
                        # PPL: positive delta = worse, negative = better
                        t = max(0.0, min(1.0, delta / 2.0)) if delta >= 0 else 0.0
                        t_good = max(0.0, min(1.0, -delta / 2.0)) if delta < 0 else 0.0
                    else:
                        # Acc: negative delta = worse, positive = better
                        t = max(0.0, min(1.0, -delta / 0.05)) if delta <= 0 else 0.0
                        t_good = max(0.0, min(1.0, delta / 0.05)) if delta > 0 else 0.0
                    if abs(delta) < 1e-4:
                        clr = GRAY
                    elif t > 0:
                        clr = f"\033[38;5;{_gradient_color(t)}m"
                    else:
                        clr = f"\033[38;5;{_gradient_color(0.0)}m" if t_good > 0.5 else f"\033[38;5;{_GRADIENT[2]}m"
                    row_str += f"{clr}{s:>8}{RESET}  │"
                else:
                    row_str += f"{GRAY}{'—':>8}{RESET}  │"
            print(row_str)

        print("└" + "┴".join("─" * w for w in widths) + "┘")

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

    candidates = glob.glob(f"/tmp/full_{model_name}_*_results.pb")
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
    return [c for s, c in scored if s == best_score]


def _match_filter(filename, expr):
    """Test whether *filename* matches a filter expression.

    Syntax (case-insensitive, matched against basename):
        keyword          substring match
        a*b              glob-style wildcard within a term
        a,b              AND  (all terms must match)
        a|b              NOT  (b must NOT match)

    Example: "static,pwl|w8"  means  must contain 'static' AND 'pwl' but NOT 'w8'

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

    # Split on | first to separate include terms from exclude terms
    parts = expr.lower().split('|')
    include_expr = parts[0]           # everything before the first |
    exclude_terms = parts[1:]         # everything after each |

    # Include: split on commas for AND
    include_terms = [t.strip() for t in include_expr.split(',') if t.strip()]
    if not include_terms:
        return True

    # All include terms must match
    if not all(_term_matches(t) for t in include_terms):
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
  python show_results.py -s                       # all runs + FP16 baseline
  python show_results.py -s dart                   # substring: *dart*
  python show_results.py -s dart pwl               # AND: *dart* AND *pwl*
  python show_results.py -s "static,pwl"           # AND: *static* AND *pwl*
  python show_results.py -s "dart|static"          # NOT: *dart* but NOT *static*
  python show_results.py -s "static,pwl|w8"        # AND+NOT: *static* AND *pwl* but NOT *w8*
  python show_results.py -s "static*pwl"           # glob: *static*pwl*
  python show_results.py -s --nbl                  # exclude FP16 baseline
  python show_results.py -s /tmp/quarot*.pb        # literal paths (shell glob)""",
    )
    parser.add_argument("-s", "--summary", action="store_true",
                        help="Show summary table of all results")
    parser.add_argument("--nbl", "--no-baseline", action="store_true",
                        dest="no_baseline",
                        help="Exclude the FP16 full-precision baseline")
    parser.add_argument("filters", nargs="*", default=[],
                        help="Filter expressions or literal paths")
    args = parser.parse_args()

    if not args.summary:
        parser.print_help()
        return

    # ── Gather files ────────────────────────────────────────────────────
    all_results = glob.glob("/tmp/*_results.pb")

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

    runs = load_results(paths)

    if args.no_baseline:
        runs = [(l, d, m) for l, d, m in runs if m != "full"]

    print_summary(runs)


if __name__ == "__main__":
    main()
