---
name: dart-run-experiments
description: Drive fake_quant/run_experiments.py to launch DartQuant calibrations or inference batches over a runfile (default ../data/runs/runs_v6.ini). Use when the user asks to "run the experiments", "kick off the runfile", "rerun X", "regenerate cal", "run the ablations", or otherwise wants to execute or re-execute experiment batches. Always run a GPU availability check first. Always preview with --dry before any --recalib invocation. Args may be: a runfile path, a substring filter, or empty (use default runfile).
---

# dart-run-experiments

Wrap `fake_quant/run_experiments.py`. Two responsibilities: (a) pick the right flag combination based on user intent, (b) avoid known footguns (especially the `--gptq + no --recalib` cal mismatch from 2026-05-04).

## Always-do steps (in order)

1. **GPU check** — show what's free before suggesting a command.
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
   ```
   "Free" = `memory.used < ~500 MiB`. Note which GPUs are free.

2. **Decide flag combination** from user intent (see flag table below).

3. **`--dry` first, always — interpret the output yourself.** Run the dry preview to get the numbered queue and GPU assignment. Use it to:
   - Verify the queue matches the user's stated intent (which lines, how many).
   - Compute `--skip` lists for partial-scope runs (see `--recalib` note below).
   - Catch obvious mistakes (wrong runfile path, no lines selected, GPUs that aren't free).

   **Two entry points — pick by intent:**
   - **Inference batch** (or combined cal+inference): run from `fake_quant/`, runfile path `../data/runs/runs_v6.ini`.
     ```bash
     cd /workspace/DartQuant/fake_quant && python run_experiments.py --runfile ../data/runs/runs_v6.ini -g <GPUS> --dry [other flags]
     ```
   - **Cal-only** (e.g. user wants to regenerate cal artifacts without launching inference): run from `calibrater/`, runfile path `../data/runs/runs_v6.ini`.
     ```bash
     cd /workspace/DartQuant/calibrater && python multi_calibration.py --runfile ../data/runs/runs_v6.ini -g <GPUS> --dry [other flags]
     ```
   `multi_calibration.py` shares the same flag surface (`--runfile`, `--recalib`, `--gptq`, `-g`, `--dry`). Dry output paths are relative to the cwd you ran from — don't get confused if you `--dry` `run_experiments.py` from `fake_quant/` and see cal subprocess paths that look "wrong"; they execute with cwd=calibrater/.

   No user-confirmation step. If the dry output matches intent, proceed. If it doesn't match, pick the closest reasonable interpretation, log what you chose and why, and proceed. ABORT (with a clear log) only if the queue is empty or no GPUs are free.

4. **`--recalib` ignores runfile flags — handle scope yourself.** `--recalib` re-runs EVERY line in the runfile. Standard partial-recalib pattern: read the dry output, substring-match line labels against the user's stated intent ("the v6 ablations", "the M4S4 sweep"), build the complement as `--skip`, proceed. If multiple interpretations are plausible, pick the most specific match and log what was selected. Don't block.

5. **Launch** in background. Multi-hour cal+inference batches MUST run in background (`run_in_background: true`) — never block the conversation on them.

6. **Operating mode.** This skill is designed to run in autonomous/overnight loops. NEVER block on user input. When intent is ambiguous, pick the safest default (see "Default-on-ambiguity" rules below), log the choice, and proceed.

## Flag decision table

User says ... | Intent | Entry point + flags | Notes
---|---|---|---
"rerun the inference for the v6 runs" | results cache invalid, cal+GPTQ OK | `run_experiments.py` from `fake_quant/`, `--overwrite` | Cheapest. Just regenerates `.pb`. Respects `[DONE]/[FAST]` flags.
"the cal code changed, regenerate" | cal must be redone | `run_experiments.py` from `fake_quant/`, `--recalib` | **Re-runs EVERY line** regardless of `[DONE]` flags. Always pair with `--skip` for partial re-cal.
"just regenerate cal, don't run inference" | cal-only refresh | `multi_calibration.py` from `calibrater/`, `--recalib` | Cal-only entry point. Same flag surface as run_experiments.py.
"the GPTQ code changed" | GPTQ ckpt invalid | `run_experiments.py` from `fake_quant/`, `--recalib --gptq` | NEVER `--gptq` alone — see footgun. Same `--recalib` scope warning applies.
"preview which lines would run" | scoping | add `--dry` (to whichever entry point) | This is mandatory before any real run.
"run only the unflagged lines" | runfile in normal mode | `run_experiments.py` from `fake_quant/` (no extra flags) | Default behavior — respects `[DONE]/[FAST]/[CAL]` markers.
"re-cal only these specific lines" | partial recalib | `--recalib --skip <line-nums>` from appropriate entry point | Substring-match labels in dry output to compute `--skip`.
"quick PPL check" | speed | add `--very-fast` | wt2 only, ~3 min/run.
"normal eval, no MMLU" | speed | add `-F` / `--fast` | wt2/ptb/c4, ~7-10 min/run. Most common default.
"thorough eval with lm_eval" | full results | omit fast flags | Hours/run.

## Default-on-ambiguity rules (no user prompts)

These replace footgun "ask the user" patterns with autonomous safe defaults. Always log the rule that fired.

**`--gptq` requested without `--recalib`** ⚠️ — auto-add `--recalib`. New GPTQ regen against an old cal artifact produces bad PPL (this bit us on 2026-05-04). Adding `--recalib` does the right work; the cost is one extra cal pass. Log: "auto-added --recalib to avoid GPTQ-vs-cal mismatch".

**`--gptq` when GPTQ code didn't change** — auto-drop `--gptq`. GPTQ is deterministic from (FP16 cal + weights + args), so regenerating identical checkpoints wastes ~3 min/run. Heuristic: if `git log --since="<last cal time>" -- fake_quant/gptq_utils.py fake_quant/gptaq_utils.py` returns nothing, drop the flag. Log: "dropped --gptq, GPTQ code unchanged since last cal".

**Specified GPUs not free** — auto-substitute with the largest set of free GPUs (memory < 500 MiB) of the same count. If fewer free GPUs than requested, use whatever's free. If zero free, ABORT with a log message identifying the busy GPUs and their owning processes (`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`). Log: "GPUs 0-3 requested but 0,2 busy; running on 4,5,6,7 instead".

**Forgetting `--runfile` for runfile-mode runs** — auto-add `--runfile ../data/runs/runs_v6.ini` (the default). Log: "added default --runfile". Only fall back to legacy 7-experiment mode if the user explicitly asked for it.

**Empty trimmed queue after `--skip`** — abort with log explaining no lines matched intent. This is a hard error; better to surface than silently launch nothing.

**Multiple plausible label-match interpretations** — pick the most specific (longest substring match against user phrasing). If two are equally specific, run BOTH (the queue is just the union). Log: "intent matched 'M4S4 ablations' AND 'gguf' subsets; running union".

## GPU spec format

- Single number `1` → expands to `1,2,3,4,5,6,7` (7 experiments fan-out, only used in legacy mode without `--runfile`)
- Comma list `0,1,2,3` → exactly those GPUs, one per concurrent run
- For runfile mode: pick a comma-list with as many GPUs as you have free; `run_experiments.py` schedules across them.

## Default runfile

`../data/runs/runs_v6.ini` (relative to fake_quant/). Use this unless the user specifies a different path.

## Skip / select-runs (the standard partial-recalib workflow)

`--recalib` is all-or-nothing — it ignores `[DONE]/[FAST]/[CAL]` markers and re-runs every line. To restrict scope, pair with `--skip` (1-indexed line numbers from `--dry` print order).

Standard partial-recalib pattern (fully autonomous):
1. From the right entry point (`fake_quant/` for combined cal+inference, `calibrater/` for cal-only): `python <script> --runfile ../data/runs/runs_v6.ini --recalib --dry` → full numbered queue.
2. Read the queue. Substring-match each label against the user's stated intent.
3. Build `--skip <complement>` = the line numbers that DON'T match the intent.
4. Re-run `--dry` with `--skip` to verify the trimmed queue is non-empty.
5. If trimmed queue is non-empty, launch without `--dry` (in background). If empty, ABORT with log.

No user prompts. Log the matching decisions in plain text alongside the launch so the audit trail is visible.

## Calibration GPU

`--calibrate_gpu` defaults to the first GPU in `--gpus`. Cal runs sequentially on one GPU before the parallel inference batch. If the user wants cal on a specific GPU (e.g. to leave the bigger GPUs free for parallel inference), pass `--calibrate_gpu <N>`.

## Logging convention

Each run's log lands at `data/cached_results/<label>.log` (per the runfile entry name) and `<label>_CAL.log` for cal phase. After launching in background, point the user at these paths so they can tail.

## Path layout (shared across DartQuant skills)

- `data/act_scales/<model>/` — calibration `.pt` files (consumed by `dart-analyze-scales` skill).
- `data/gptaq_checkpoints/<tag>/<model>_w<bits>/*.pth` if `--gptaq`, else `data/gptq_checkpoints/<tag>/<model>_w<bits>/*.pth` — GPTQ checkpoints. `--gptq` runner flag deletes the matching dir.
- `data/cached_results/` — inference logs, cal logs, `*_results.pb` (consumed by `dart-ppl-status` skill).
- `data/runs/*.ini` — runfiles. `runs_v6.ini` is the default.

## Out-of-scope: ad-hoc runs not in any runfile

This skill drives `run_experiments.py` / `multi_calibration.py` over a runfile. If the user asks to launch a one-off run that isn't represented in `runs_v6.ini` (e.g. the `s1_no_fp16cal_full_*` PPL-only debug runs from 2026-05-04), the right tool is `fake_quant/Script/dart_gptq_wxaykvz.sh` directly. Don't try to force the request through `run_experiments.py`. Either:
- Surface to the user: "this run isn't in any runfile; use `bash Script/dart_gptq_wxaykvz.sh ...` directly, or add it to `runs_v6.ini` first."
- OR (if intent is unambiguous) invoke `dart_gptq_wxaykvz.sh` directly, but log clearly that you're going outside this skill's scope.

## `[ERROR]` / `[ERROR-CALIBRATE]` recovery

When a runfile line fails, `run_experiments.py` writes back `[ERROR]` (inference) or `[ERROR-CALIBRATE]` (cal phase) to that line. The skill's response when the user asks "why did <label> fail":
1. Grep the label's logs:
   - `data/cached_results/<label>.log` (inference output)
   - `data/cached_results/<label>_CAL.log` (cal phase output)
2. Look for tracebacks (`Traceback (most recent call last)`), `RuntimeError`, `AssertionError`, or `Error:` lines.
3. Surface the failing line + the relevant traceback excerpt — don't dump the whole log.

Don't auto-retry failed lines without understanding the failure first; the same failure will likely repeat.

## Don't

- Don't block on user input. Pick the safe default per "Default-on-ambiguity rules" and proceed. Log the choice.
- Don't tail running jobs in a loop — kick them off in the background and use `dart-ppl-status` for status checks.
- Don't recreate or modify the runfile. The user owns runs_v6.ini; if the queue is wrong, ABORT and log; don't try to "fix" it.
- Don't strip the runfile's `[DONE]` / `[FAST]` flags. Those are state markers the script writes back.

## Example: autonomous workflow

User: "the cal-fix is committed, rerun the v6 ablations with the new cal."
You:
1. `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` → identify free GPUs.
2. `cd /workspace/DartQuant/fake_quant && python run_experiments.py --runfile ../data/runs/runs_v6.ini -g <free-gpus> --recalib -F --dry` → full numbered queue.
3. Read the queue. Substring-match "v6 ablations" → `s1_no_fp16cal_full_M4S4_t2int24a0_*` lines. Compute `--skip` for everything else.
4. Re-run `--dry --skip <complement>` → verify trimmed queue is non-empty and contains the expected ablation labels.
5. Drop `--dry`, launch in background. Output: log paths + a one-line summary of what was launched + a pointer to `dart-ppl-status` for monitoring.

Trimmed queue empty → ABORT with log. Trimmed queue contains unexpected lines → log a warning but proceed (the user can `--skip` more next iteration).

## Example: footgun caught and auto-fixed

User: "rerun with `--overwrite --gptq`"
You:
1. Detect: `--gptq` without `--recalib`. Per default-on-ambiguity rules, auto-add `--recalib`.
2. Log: "auto-added --recalib to avoid GPTQ-vs-cal mismatch (the 2026-05-04 bug)".
3. Continue with the standard workflow (GPU check → dry → skip-complement → launch).
4. Final command logged: `... --recalib --gptq --overwrite -F` (and `--skip` if intent narrowed scope).

## Example: the footgun the user actually hit

User: "rerun with `--overwrite --gptq`"
You: "That combo regenerates GPTQ but reuses the old cal — they'll mismatch and PPL will be bad (this is the bug we fixed yesterday). You probably want `--recalib` instead. With `--recalib --gptq` if GPTQ code also changed, or just `--recalib` if only cal code changed. Want me to do the latter?"
