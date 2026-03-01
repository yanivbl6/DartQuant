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


def color_val(val, best, worst, fmt=".2f", is_ppl=False):
    """Color a numeric value: green=best, red=worst, yellow=middle."""
    if val is None:
        return f"{GRAY}{'—':>8}{RESET}"
    s = f"{val:{fmt}}"
    if is_ppl:  # lower is better
        if val == best:
            return f"{GREEN}{BOLD}{s:>8}{RESET}"
        elif val == worst:
            return f"{RED}{s:>8}{RESET}"
        else:
            return f"{YELLOW}{s:>8}{RESET}"
    else:  # higher is better
        if val == best:
            return f"{GREEN}{BOLD}{s:>8}{RESET}"
        elif val == worst:
            return f"{RED}{s:>8}{RESET}"
        else:
            return f"{YELLOW}{s:>8}{RESET}"


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

    # Build data matrix
    labels = [r[0] for r in runs]
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
                    if is_ppl[c]:
                        # For PPL, positive delta = worse
                        sign = "+" if delta >= 0 else ""
                        clr = RED if delta > 0.1 else (GREEN if delta < -0.1 else GRAY)
                    else:
                        sign = "+" if delta >= 0 else ""
                        clr = GREEN if delta > 0.1 else (RED if delta < -0.1 else GRAY)
                    s = f"{sign}{delta:.2f}"
                    row_str += f"{clr}{s:>8}{RESET}  │"
                else:
                    row_str += f"{GRAY}{'—':>8}{RESET}  │"
            print(row_str)

        print("└" + "┴".join("─" * w for w in widths) + "┘")

    # Legend
    print()
    print(f"  {GREEN}■{RESET} Best   {YELLOW}■{RESET} Mid   {RED}■{RESET} Worst   {GRAY}—{RESET} Missing")
    print(f"  PPL↓ = lower is better   Acc↑ = higher is better")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DartQuant experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python show_results.py -s                    # all runs, with FP16 baseline
  python show_results.py -s --nbl              # exclude FP16 baseline
  python show_results.py -s /tmp/quarot*.pb    # specific files only""",
    )
    parser.add_argument("-s", "--summary", action="store_true",
                        help="Show summary table of all results")
    parser.add_argument("--nbl", "--no-baseline", action="store_true",
                        dest="no_baseline",
                        help="Exclude the FP16 full-precision baseline")
    parser.add_argument("files", nargs="*", default=[],
                        help="Specific .pb files (default: /tmp/*_results.pb)")
    args = parser.parse_args()

    if not args.summary:
        parser.print_help()
        return

    # Gather files
    if args.files:
        paths = list(args.files)
        # Auto-include the full baseline if not excluded and not already present
        if not args.no_baseline:
            full_candidates = glob.glob("/tmp/full_*_results.pb")
            for fc in full_candidates:
                if fc not in paths:
                    paths.append(fc)
    else:
        paths = glob.glob("/tmp/*_results.pb")

    if not paths:
        print("No result files found.")
        return

    runs = load_results(paths)

    if args.no_baseline:
        runs = [(l, d, m) for l, d, m in runs if m != "full"]

    print_summary(runs)


if __name__ == "__main__":
    main()
