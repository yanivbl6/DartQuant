---
name: dart-ppl-status
description: Report DartQuant run results by delegating to fake_quant/show_results.py. Use when the user asks "what's the status", "did it work", "any updates", "compare X vs Y", or otherwise wants to inspect runs in data/cached_results/. Use proactively after a background bash task in /workspace/DartQuant finishes. Args may be: a filter expression for show_results.py, or empty for "all recent".
---

# dart-ppl-status

DartQuant has a canonical `fake_quant/show_results.py` that reads cached `.pb` result files, formats colored tables, supports rich filter syntax, and computes deltas. This skill is a thin wrapper over it. Don't re-implement what `show_results.py` already does — invoke it.

## Decision tree

1. **Run is complete** → `.pb` exists in `data/cached_results/` → use `show_results.py`. This is the common path.
2. **Run is in progress** → no `.pb` yet, log file is being written → grep the log directly.
3. **Tag entirely unknown** → list `data/cached_results/*.log` modified in the last hour, summarize.

## show_results.py invocation

```bash
cd /workspace/DartQuant/fake_quant && python show_results.py <FILTER> [FLAGS]
```

**Filter syntax** (from the script's docstring):
| Form | Meaning |
|---|---|
| `v67` | substring `*v67*` |
| `v67,t2int24` | AND |
| `v67\|imitate` | matches v67, NOT imitate |
| `v67*sym` | glob (one `*` allowed within a token) |
| `[t2int20,t2int22,t2int24]` | OR-group |

**Useful flags**:
- `-F` — PPL only, hide accuracy columns. Use this by default unless the user asks about MMLU/lm_eval.
- `-d` — delta vs FP16 baseline (baseline auto-included).
- `-c EXPR` — split runs into matching-vs-not, delta table. Use for "compare X vs Y" requests.
- `--nbl` — exclude FP16 baseline (rarely needed).

## Mapping user requests to commands

| User says | Run |
|---|---|
| "status of v67 runs" / "any updates" | `python show_results.py v67 -F` |
| "did wasym work?" | `python show_results.py v67,wasym -F` |
| "compare v67 to v63" | `python show_results.py "v6[3,7]" -c v67 -F` |
| "show the M4S4 t2 sweep" | `python show_results.py M4S4,t2int -F` |
| "everything done so far" / empty arg | `python show_results.py -F` |
| "show w_asym vs sym deltas" | `python show_results.py M4S4,t2int24 -c wasym -F` |

If a request doesn't fit a template, build the filter from the user's intent. The script's epilog has more examples — read [fake_quant/show_results.py:1348-1363](fake_quant/show_results.py#L1348) if a query is ambiguous.

## In-progress runs (no .pb yet)

When `show_results.py` returns "No result files found" and the user is asking about a running task:

```bash
LOG=/workspace/DartQuant/data/cached_results/v63_<tag>.log
# 1. PPL printed for finished datasets:
grep -E "WIKITEXT2 PPL|PTB PPL|C4 PPL" "$LOG" 2>/dev/null
# 2. If still mid-eval, show running PPL + percentage:
tail -c 300 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -1
```

Output one line per run: `<tag>: running, PPL≈NN.NN at NN%` or `<tag>: starting (no PPL yet, in cal/GPTQ phase)`.

## Don't

- Don't read `.pb` files directly — they're pickled. `show_results.py` is the only consumer.
- Don't read full logs into context — they include 30-100 KB of progress bars. Always `grep` or `tail -c 300` first.
- Don't strip ANSI colors from `show_results.py` output. The terminal renders them; stripping makes the table harder to scan.
- Don't hardcode v63-era reference numbers in the skill output. `show_results.py` already includes the FP16 baseline column and `-d` shows deltas; if the user wants v63 specifically, use `-c v67` against a v63-included filter.

## Path layout (shared across DartQuant skills)

- `data/cached_results/` — inference logs (`<label>.log`), cal logs (`<label>_CAL.log`), result caches (`*_results.pb`). This skill primarily reads from here.
- `data/act_scales/<model>/` — calibration `.pt` files (consumed by `dart-analyze-scales`).
- `data/runs/*.ini` — runfiles (consumed by `dart-run-experiments`).

## Example invocations

User: "any updates on the runs?"
→ `Bash({command: "cd /workspace/DartQuant/fake_quant && python show_results.py v67 -F"})`

User: "did the wasym fix actually work?"
→ `Bash({command: "cd /workspace/DartQuant/fake_quant && python show_results.py v67,wasym -F"})`

User: "compare M4S4 t2int24 between v63 and v67"
→ `Bash({command: "cd /workspace/DartQuant/fake_quant && python show_results.py 'M4S4,t2int24,v6[3,7]' -c v67 -F"})`
