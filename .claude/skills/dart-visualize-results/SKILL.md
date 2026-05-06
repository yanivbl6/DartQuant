---
name: dart-visualize-results
description: Generate presentation-quality figures from DartQuant cached results via fake_quant/show_results.py's --draw / --compare modes, and (when the script's visualization features fall short) iteratively extend show_results.py and update this skill. Use when the user asks to "make a figure", "draw a chart", "plot X vs Y", "visualize", "for the slide deck", "compare X graphically", or otherwise wants images they'd ship outside the terminal. Distinct from dart-ppl-status (that one reads numbers; this one renders pictures). Args are usually a filter expression + a compare key.
---

# dart-visualize-results

This skill drives `fake_quant/show_results.py` when the goal is **a figure for external use**, not a number for the user to read inline. The script already implements bar + line chart rendering via `--draw` / `-c`, but the rendering code has gaps (axis-limit heuristics, chart-type forcing, ordering) that often need a small code change before the figure is presentation-ready. So this skill is *both* a wrapper and a maintenance loop for `show_results.py`'s plotting code.

**When you change the visualization code, update the "Known gaps" section of this skill in the same edit.** The skill drifts behind the script otherwise. This is the explicit reason the body of this skill is more detailed than the other dart skills.

## Decision tree — is this skill the right one?

- User wants the **answer in the chat** (numbers, deltas, "did X work") → `dart-ppl-status`.
- User wants a **picture they can paste into a deck or share** → this skill.
- User asks for both → run `dart-ppl-status` first (numbers are fast), then run this skill.

## The iteration loop

The default flow is **try → inspect → adjust → present**. Don't render once and report. Three rounds is normal.

### Step 1 — pre-flight without `--draw`

Before generating figures, run `show_results.py` *without* `--draw` to see what's in scope:

```bash
cd /workspace/DartQuant/fake_quant && python show_results.py "<filter>" -c "<key>" -F
```

Read the comparison table that prints. Check:

| Check | Why it matters |
|---|---|
| **Row count** | Bar charts get unreadable past ~10-12 bars per metric. If the filter pulls 20+ runs, tighten it before drawing. |
| **PPL range** | If min and max differ by >20, the auto y-limit caps at min+10 (see `_set_focused_ylim` at [show_results.py:378](fake_quant/show_results.py#L378)) and runs above that vanish. Either tighten the filter or plan a code-side y-limit override. |
| **Compare mode** | The `-c` parser auto-detects `numeric` / `variant` / `wildcard` / `binary` (see [show_results.py:147](fake_quant/show_results.py#L147)). Bar charts work in all four; line charts only render in `numeric` / `wildcard` modes *and only with ≥3 key values* (see [show_results.py:697](fake_quant/show_results.py#L697)). |
| **Missing baseline** | FP16 baseline auto-includes unless `--nbl`. Confirm it's in the table. |

### Step 2 — render

```bash
cd /workspace/DartQuant/fake_quant && python show_results.py "<filter>" -c "<key>" --draw <metrics>
```

Metric keywords (from `_METRIC_ALIASES` at [show_results.py:91](fake_quant/show_results.py#L92)):
- PPL: `wiki` / `wikitext` (= wikitext2), `ptb`, `c4`
- Accuracy: `mmlu`, `piqa`, `hs` (hellaswag), `arce`, `arcc`, `wino`, `lambada`, `siqa`, `obqa`, `avg`

Multiple comma-separated metrics produce a multi-panel `summary_<label>.png` (when ≥2 panels). Single-metric runs only produce per-chart files — no summary.

Output paths:
- Per-metric bar chart: `figures/<metric>_<key>_bar.png`
- Per-metric line chart (if applicable): `figures/<metric>_<key>_line.png`
- Multi-panel summary: `figures/summary_<key>.png`
- "All runs" mode (`-c '*'`): `figures/subfigures/<metric>_all_bar.png` + `figures/summary_all.png`

All written from `_draw_figures` at [show_results.py:619](fake_quant/show_results.py#L619).

### Step 3 — inspect and report back

Open the figure(s) yourself or report the path to the user with markdown links so they render clickable in the IDE:

```markdown
[summary_t2int.png](figures/summary_t2int.png)
```

Always link the summary first. Link individual sub-charts only if the summary doesn't tell the whole story (e.g., user wants to embed just the MMLU panel).

If the figure has problems — squashed y-axis, too many bars, wrong chart type, awkward ordering — go to Step 4.

### Step 4 — fix, either filter-side or via `--viz` / code-side

The script has an open-ended **`--viz K=V,K=V`** flag for visualization-only tunables — affects figure rendering only, never the comparison or table logic. New iteration knobs go here, not as new top-level argparse flags. See `parse_viz` at [show_results.py:91](fake_quant/show_results.py#L92).

| Problem | Filter-side fix | `--viz` knob | Code-side change |
|---|---|---|---|
| Too many bars | Tighten filter (`-c "v68,M4S4,t2int"` → narrower) | — | — |
| Y-axis clips outliers | `--nbl` (drop FP16 if it's the cap-trigger) | `--viz ymin=N,ymax=M` (all panels), `--viz ppl_ymax=N` / `acc_ymax=N` (per-panel-type) | Implemented — read in `_set_focused_ylim` at [show_results.py:418](fake_quant/show_results.py#L418). PPL inversion preserved. **Use `ppl_ymax`** (not the global `ymax`) when capping a PPL outlier in a multi-metric panel that includes accuracy — global `ymax=35` will also clip MMLU/avg around 35, hiding good accuracy. |
| Want line, got bar | Use a numeric or wildcard `-c` if compatible (`-c 't2int*'` instead of `-c 'wAsym'`) | (proposed: `--viz line=true`) | Edit gate in `_draw_figures` around [show_results.py:745](fake_quant/show_results.py#L745) to read `viz.get('line')`; document the new key here. |
| Want bar, got both | `--bar` flag exists (predates `--viz`) | — | — |
| X-axis order weird | — | (proposed: `--viz sort=numeric`) | Sort `key_values` in `_draw_figures` when `viz.get('sort') == 'numeric'`. |
| Colors collide past 10 keys | — | (proposed: `--viz palette=viridis`) | Branch on `viz.get('palette')` in `_draw_figures` color setup. |

When a code-side change is needed, follow the **"Adding a new viz key"** recipe below and update this skill body with the new key in the same edit. Tag it with the date so future-you knows the script has moved.

### Adding a new viz key

The whole point of `--viz` is to keep the visualization namespace open without bloating argparse. The recipe:

1. **Read the key** in the relevant render function (e.g., `_render_bar_chart`, `_render_line_chart`, `_set_focused_ylim`, or `_draw_figures` for plumbing). Each already accepts a `viz` dict (default `None`); use `(viz or {}).get('your_key')`.
2. **Document the key** in two places: (a) the docstring of `parse_viz` at [show_results.py:91](fake_quant/show_results.py#L92), (b) the `--help` text of `--viz` (one line per key in the metavar / help string), (c) THIS skill body — both Step 4's table and the "Known gaps" section if applicable.
3. **No argparse changes needed.** Don't add a new `--your_flag`. Keep the surface area small.
4. **Naming rules:** lowercase, snake_case if multi-word, prefer the matplotlib-equivalent name (`ymin` not `ylim_min`, `palette` not `cmap`). Values auto-coerce in `parse_viz`: bool / int / float / str.

## Known gaps in show_results.py visualization (as of 2026-05-06)

Listed in expected-bite-order. When you close one, delete it from this list and add the corresponding `--viz` key to Step 4's table.

1. **~~No manual y-limit override~~ — closed 2026-05-06.** `--viz ymin=N,ymax=M` reads in `_set_focused_ylim` at [show_results.py:418](fake_quant/show_results.py#L418). PPL inversion preserved. ← Use this when the auto-cap (`floor(vmin/5)*5 + 10` for PPL) hides outliers.
2. **Line chart in binary/variant compare modes.** Wildcard mode now works (see #7 below). Binary (e.g. `-c wAsym`) and variant (string-valued) modes still get bar-only because their key_values are non-numeric strings, and the line gate at [show_results.py:745](fake_quant/show_results.py#L745) checks `cmp_mode in ("numeric", "wildcard")`. `--bar` forces bar-only but there's no symmetric `--viz line=true` for the inverse. Proposed key: `viz.get('line')` to override the gate (would still need a numeric x-axis source — for binary/variant modes that source doesn't exist by default, so this gap is mostly conceptual).
3. **No summary figure for single-metric runs.** `_save_summary_figure` at [show_results.py:484](fake_quant/show_results.py#L484) returns early when `len(specs) < 2`. Sometimes you want the summary's nicer labels even with one panel. Easy fix: read `viz.get('force_summary')` and skip the `< 2` early return.
4. **No x-axis ordering control.** Bar / line charts use `key_values` in insertion order. For `t2int*` wildcard the order is whatever order the result files happened to list — often not numeric. Proposed key: `viz['sort'] = 'numeric'` to sort `key_values` (and corresponding column data) by parsed wildcard value.
5. **Color stride is fixed.** `_stride = 3` at [show_results.py:706](fake_quant/show_results.py#L706) cycles through tab10. Fine for ≤10 keys, repeats colors past that. Proposed key: `viz['palette'] = 'viridis'` to switch to a perceptually-uniform palette when `len(key_values) > 10`.
6. **Title gets long when `common_sub` is long.** `_make_title` appends `common_sub` as a second line. Long shared tags (e.g., long hwscale specs) eat vertical space. Proposed key: `viz['title_max']=80` to truncate with ellipsis, or move to a figure-level suptitle.
7. **~~Line chart aborts when wildcard baseline is non-numeric, AND collapses non-pure-digit captures~~ — closed 2026-05-06.** Two related fixes in [_line_chart_x_nums](fake_quant/show_results.py#L580) and [_render_line_chart](fake_quant/show_results.py#L598). (a) The "vanilla" baseline used by wildcard mode is non-numeric — used to make the whole function return `None` (no line chart at all). Now it's skipped from the line trace and rendered as a gray dotted hline (analogous to FP16). (b) Wildcard captures like `t2int*` matching `t2int22a0` capture `"22a0"`, which fails `.isdigit()`; the old fallback then took `digits[-1]` = `"0"` so all `t2int<N>a0` collapsed to x=0. Now we extract the FIRST integer from the capture group. X-tick labels also shortened to the captured wildcard part (e.g. `22a0` instead of `t2int22a0`) so labels don't overlap.

## Footguns

- **`--draw` requires `-c`.** Without `-c`, argparse errors out (see [show_results.py:1386](fake_quant/show_results.py#L1386)). The "all runs" form is `-c '*' --draw <metric>` (see [show_results.py:1363](fake_quant/show_results.py#L1363)).
- **`-F` doesn't gate `--draw`.** `-F` only hides accuracy columns from the *table*. `--draw mmlu` still works fine even with `-F` because `--draw` looks up the column independently.
- **Figures are written outside `fake_quant/`.** They land in `figures/` at the repo root (`SCRIPT_DIR/..` from inside the script). Don't `cd` into `figures/` to find them — use the markdown link format above.
- **Rendering is `Agg` (non-interactive).** `_draw_figures` calls `matplotlib.use("Agg")` at [show_results.py:623](fake_quant/show_results.py#L623). Don't try to `plt.show()` in a code-side patch — it's a no-op.

## Don't

- Don't render and report blindly. Always pre-flight (Step 1) so the user doesn't get a chart with vanished outliers or 30 unreadable bars.
- Don't extend `show_results.py` for one-off needs without updating the Known gaps section. The next user of this skill needs to know the gap is closed.
- Don't introduce a separate `plot_results.py` — `show_results.py` is the canonical entry point for tabular AND graphical output. Keep visualization logic in this one file so filter/compare semantics stay synchronized.
- Don't strip ANSI from `show_results.py` output when running pre-flight — the colored deltas convey the table's structure (PPL vs accuracy, deltas vs absolute).
- Don't pass a literal file glob (`--draw foo*.pb`) — `--draw` is metric keywords, not paths.

## Example flows

User: "visualize v68 with M4S4 and different t2-accumulators, want a slide-ready figure"

```bash
# Step 1 — pre-flight: how many runs, what range?
cd /workspace/DartQuant/fake_quant && python show_results.py "v68,M4S4" -c "t2int*" -F
# (read the table — count runs, eyeball PPL range)

# Step 2 — render the figure
python show_results.py "v68,M4S4" -c "t2int*" --draw wiki,c4,mmlu,avg
```

Then report back with `[summary_t2int.png](figures/summary_t2int.png)` and link individual panels if needed.

User: "the y-axis is squashed, the int28 run got cut off"

→ Closed gap #1. Re-run with explicit bounds:
```bash
python show_results.py "v68,M4S4" -c "t2int*" --draw wiki,c4 --viz ymin=10,ymax=40
```
The PPL-axis inversion (lower=better at top) is preserved automatically.

User: "compare wAsym vs sym graphically across the t2int sweep"

→ `-c wAsym` is binary mode, no line chart. Two options:
- Run twice: `-c "t2int*"` filtered to `wAsym` runs, then again filtered to non-`wAsym`, and place side-by-side manually.
- Code-side: add a `--line` flag that overrides the cmp_mode gate at [show_results.py:697](fake_quant/show_results.py#L697) — gap #2 above.

## Path layout (shared across DartQuant skills)

- `data/cached_results/*_results.pb` — input to `show_results.py`.
- `figures/` (at repo root, *not* inside `fake_quant/`) — output of this skill. Gitignored.
- `fake_quant/show_results.py` — the only consumer / renderer. Edit this file when extending visualization.
