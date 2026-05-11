---
name: dart-shepherd-runfile
description: Shepherd a large DartQuant runfile end-to-end (cal → fast pass → full eval → comparison) over multiple hours, with adaptive self-paced status checks. Use whenever a sweep has more than ~8 runs — anything that's going to take more than ~1 hour and benefits from the cal+fast→full decoupling. Examples of triggers: "run the whole runfile", "kick off X end-to-end", "shepherd / babysit the v68 sweep", "extend results to G=32", "run all the v6X ablations". Read `dart-run-experiments` FIRST for the underlying flag semantics, footguns, and command construction — this skill builds on top of it and assumes that knowledge. Args: runfile path, sometimes accompanied by a base-runfile to copy from with one-flag substitutions.
---

# dart-shepherd-runfile

**Read `dart-run-experiments` first** — this skill assumes you understand that one's flag table, dry-preview discipline, and footguns (esp. the `--gptq + no --recalib` mismatch and the `[FAST]` / `[CAL]` semantics). This skill is what you do *on top of* that for sweeps too long to finish in one Bash invocation: orchestration of phases over hours, adaptive cadence, recovery flow.

**When this skill applies:** any runfile with more than ~8 entries, where total wall-clock will exceed ~1 hour. Below that threshold, just invoke `dart-run-experiments` once — the cal+fast→full decoupling isn't worth the overhead.

A large DartQuant runfile (~30 entries) takes most of a day to run end-to-end. The work is genuinely long, and most of it is grinding — but the right *structure* and *cadence* keep things efficient and let you catch failures within minutes rather than at the end. This skill encodes that structure so you don't reinvent it each time.

## Operating principle — minimize user gating

The shepherd's prime directive: **keep the GPUs busy on whatever lines still work**. The user invoked this to walk away for hours; treat their attention as the scarcest resource. Decisions you'd normally surface for approval (skipping broken lines, applying obvious patches, moving to the next stage with partial coverage) you instead make autonomously and surface in the end-of-phase summary.

**Defaults when failures happen mid-run:**
- Failures in some lines almost never block the rest. The runner already marks failed lines with `[ERROR]` / `[ERROR-CALIBRATE]` and skips them; the other lines complete normally. **Let it.**
- **Do not stop a launched phase to investigate.** Triage happens between phases, not during. Investigations on lines that are already errored don't make the running lines faster.
- **Move to the next stage with partial coverage.** If fast pass finishes with 41/44 [FAST] and 3 [ERROR-CALIBRATE], launch full eval on the 41 — the 3 errored lines auto-skip in default mode and don't risk anything. The full-eval pass is the long pole; don't delay it for cal-side failures unrelated to it.
- **Apply minor patches autonomously.** If a failure is obviously a small bug (one-line fix, parallel-cal race, missing per-layer override that another caller already has, etc.) and the user is reachable later, just fix it and re-cal/re-run the affected lines in parallel with the next phase. Save the long ones — anything that needs a deeper code change or judgment call about correctness — for the user.

**What still warrants stopping and asking:**
- A failure pattern that suggests the *whole run* is producing bad numbers (e.g. all PPL = 10⁴, or all cals crashing on the same step). That's not "some lines broke", it's "the path is wrong."
- A patch that changes algorithmic behavior (not just a try/except or a per-layer override mirroring an existing caller). Anything touching cal math, GPTQ weight derivation, or scale computation goes to the user.
- A decision that costs real wall-clock on a large fraction of the runfile (e.g. "regenerate GPTQ checkpoints shared by 20 already-[FAST] lines"). The risk of invalidating good work is too high for autonomous action.

**Format of the end-of-phase summary:** lead with "Phase X done, N/M lines successful." Then surface the failed labels with one-line root cause each, the patches you applied (if any), and the recommended next step. The user reads this once after the phase, not N times during it.

## The phase order — always run in this order

For a fresh runfile (no `[DONE]` flags), the right pipeline is:

```
1. Setup                  (5 min)
2. Fast pass               (~2-3 hours: cal + GPTQ + PPL only)
3. Verify fast results     (5 min: any [ERROR] flags?)
4. Full-eval pass          (~6-8 hours: lm_eval added)
5. Compare                 (5 min via dart-ppl-status)
```

**Why fast pass first, not full eval directly:**
- Fast pass writes cal artifacts and GPTQ checkpoints, validates the entire path, and gives PPL feedback in 2-3h. Errors surface fast.
- Full eval reuses those cached artifacts (no `--gptq`, no `--calibrate`). The lm_eval phase is what makes it slow; cal/GPTQ would just be re-done waste.
- If the fast pass errors on 12 lines (as happened with the fp16_calib bug on 2026-05-06), you fix and recover before sinking 8 hours into a full eval that would also fail.

**Don't combine phases.** Resist "let me just run the full eval with `--calibrate -F` and then upgrade." The decoupling is what makes recovery cheap.

## Phase 1 — Setup (always do first)

```bash
# 1. GPUs free?
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# Need all 4 (or however many you'll use) at <500 MiB.

# 2. Runfile sanity
wc -l /workspace/DartQuant/data/runs/<runfile>.ini
for f in DONE FAST CAL ERROR ERROR-CALIBRATE; do
  c=$(grep -cE "^\[$f\]" /workspace/DartQuant/data/runs/<runfile>.ini)
  echo "[$f]: $c"
done

# 3. Dry preview the fast pass — confirm cal-group count and inference-row count
cd /workspace/DartQuant/fake_quant
python run_experiments.py --runfile ../data/runs/<file>.ini -g 0,1,2,3 \
  --gptq --calibrate -F --dry 2>&1 | grep -E "=== (Calibration|[0-9]+ inference)"
```

If the runfile is a copy-with-substitution (e.g. user said "extend to G=32"), do the substitutions BEFORE the dry preview:
```bash
cp /workspace/DartQuant/data/runs/<src>.ini /workspace/DartQuant/data/runs/<dst>.ini
sed -i -E '
  s/^\[(DONE|FAST|CAL|ERROR|ERROR-CALIBRATE)\] //;   # strip flags
  s/-G 16/-G 32/g;                                    # the param swap
  s/^wonly_G16:/wonly_G32:/;                          # rename labels
  s/^g16_/g32_/                                       # rename labels
' /workspace/DartQuant/data/runs/<dst>.ini
```
Note: many lines use `--set 1`, which auto-sets `acc_block_k = groupsize`. Flag this to the user when it matters.

## Phase 2 — Fast pass

```bash
cd /workspace/DartQuant/fake_quant
nohup python run_experiments.py --runfile ../data/runs/<file>.ini -g 0,1,2,3 \
  --gptq --calibrate -F > /tmp/<runfile>_fastpass.log 2>&1 &
echo "PID: $!"
```

**Time decomposition** (Llama-3.2-1B-Instruct, 4× A6000):
- Cal phase: ~12-16 min/cal × ⌈N_cal_groups / 4⌉ batches.
  - N_cal_groups depends on runfile config diversity AND on `multi_calibration.py`'s dedup. Typical v68-class runfile: 18-21 groups for 30-32 inference rows.
  - First batch may include heavier configs (fp16_calib + GPTAQ); later batches faster as FP16 cal cache hits.
  - Sub-phases per cal: model load → R1/R2 rotation → GPTQ/GPTAQ (~10-13 min, the dominant cost) → branch eq stats → activation observation → save.
  - For `--fp16_calib` lines, post-GPTQ cal is *skipped* (legitimate, not a bug — only the FP16 cal artifact is needed).
- Inference phase (`-F`): ~7-10 min/run × ⌈32 / 4⌉ batches ≈ 56-80 min.
- **Total: ~2.5-3 hours.**

## Phase 3 — Verify fast pass

```bash
for f in DONE FAST CAL ERROR ERROR-CALIBRATE; do
  c=$(grep -cE "^\[$f\]" /workspace/DartQuant/data/runs/<runfile>.ini)
  echo "[$f]: $c"
done
```

Expect: `[FAST]: <total>`, all others 0. If `[ERROR]` or `[ERROR-CALIBRATE]` is non-zero:

1. Identify the failing labels: `grep '^\[ERROR\]' <runfile> | awk '{print $2}'`.
2. Read the log: `tail -50 /workspace/DartQuant/data/cached_results/<label>.log`. For `[ERROR-CALIBRATE]`, the cal-side log is at `data/cached_results/cal_<tag>.log` (find via `ls -lt | head` or grep the fastpass log for `Calibration FAILED for:`).
3. Look for `"Static act scales not found"` (cal artifact missing — usually a dedup-key bug like 2026-05-06 fp16_calib mishandling) or genuine tracebacks.
4. **Triage per the operating principle (above).** Bucket each failure:
   - **Trivial-patch bucket** (parallel-cal race → `ignore_errors=True`, missing per-layer override that another caller already has, obvious one-line fixes): patch it, strip `[ERROR-CALIBRATE]`, re-cal the affected groups in parallel, and **proceed to Phase 4 anyway** — don't block the long phase on the small one.
   - **Real-bug bucket** (algorithmic, shape math, something that needs design judgment): surface it in the phase summary with the traceback excerpt, recommend skipping, and **launch Phase 4 on the working lines**. The errored lines auto-skip in default mode (per dart-run-experiments' flag-semantics table) and don't risk anything.
   - **Whole-run-broken bucket** (every cal failing on the same step, all PPLs ≈ 10⁴): stop and ask. This is the only bucket that gates Phase 4.

**Default: launch Phase 4 with whatever's [FAST].** "Don't proceed until all are [FAST]" was the old rule and it was wrong — it burns hours of user attention to recover lines that the next phase will skip anyway. The right rule is: launch Phase 4 in parallel with whatever recovery you're doing on the errored subset.

## Phase 4 — Full-eval pass

```bash
cd /workspace/DartQuant/fake_quant
nohup python run_experiments.py --runfile ../data/runs/<file>.ini -g 0,1,2,3 \
  > /tmp/<runfile>_fulleval.log 2>&1 &
echo "PID: $!"
```

**Important:** Don't pass `--gptq`, `--calibrate`, `-F`, or `--overwrite`. Default mode picks up `[FAST]` lines automatically and runs full eval (PPL + lm_eval), reusing cached cal `.pt` and GPTQ checkpoints. (See dart-run-experiments's flag-semantics table — `[FAST]` is **run** in default mode, only `[DONE]` skips.)

**Don't strip `[FAST]` flags.** The runner picks them up automatically. Stripping is wasted work.

**Time decomposition:**
- 4-way batch sync, ~15-25 min/run averaged (heavier configs slower).
- ~3-4 runs/hour throughput.
- 32 runs total ≈ **6-9 hours**. lm_eval `loglikelihood requests` is the bottleneck.
- Stragglers within a batch hold up the next batch; expect periods where 3 GPUs idle waiting for 1.

## Phase 5 — Compare

```bash
cd /workspace/DartQuant/fake_quant
# substitute v_old and v_new with the version tags being compared
python show_results.py "[v_old,v_new]" -c v_new
```

Or invoke `dart-ppl-status` skill with appropriate filter. Don't reinvent: `show_results.py` does the heavy lifting.

## Adaptive wakeup cadence

You'll watch this for hours. Use `/loop` in **dynamic mode** (no interval) to self-pace via `ScheduleWakeup`. Cadence pattern:

| Phase / state | Wakeup interval | Why |
|---|---|---|
| Right after launch (cals just spawned) | **270s (4.5 min)** | Cache stays warm. Want to confirm cals progress past first GPTAQ steps and no immediate crash. |
| Pre-first-artifact (still mid-batch-1) | **270s** | The "is the path actually working" question. One more tight tick before extending. |
| First batch done, stable cadence visible | **1500s (25 min)** | Single cache miss buys a long quiet window; throughput is now estimable. |
| Mid-grind (cal phase or full-eval middle) | **1800-2400s (30-40 min)** | Stable rate; no value sampling tighter. Still close enough to catch errors within an hour. |
| Long lm_eval grind, hours of homogeneous work | **3000s (50 min)** | Maximum reasonable; runs/hr stable, no behaviors change. |
| Approaching completion (~5-15 min ETA) | **270-600s** | Catch the completion event so you can launch the next phase immediately. |
| Just before phase transition | **270s** | Cache stays warm for the launch of the next phase. |

**Rules of thumb:**
- **Don't pick 300s.** Worst-of-both: cache miss without amortizing. Either go ≤270s (warm cache) or commit to ≥1200s.
- **Tight at boundaries, loose in the middle.** The interesting moments are the start (will it work?), phase transitions (cal→inference, fast→full), and completion (need to launch next thing).
- **Each refinement should reduce the schedule, not increase it.** Once you've extended to 1800s, don't go back to 600s unless you're approaching a transition.

## Watching for trouble

| Signal | What it usually means | Action |
|---|---|---|
| `[ERROR]` / `[ERROR-CALIBRATE]` count > 0 during a phase | Cal or inference failed for some lines (runner already isolated them) | **Don't stop the running phase.** Triage at phase end per Phase 3 recipe — likely launch Phase 4 anyway, recovery runs in parallel. |
| GPU at 0% util but inference subprocs alive | Batch waiting on straggler (normal) OR process hung | Check log progress; if not advancing for 10+ min, suspect hang |
| Inference log: `"Static act scales not found"` | Cal artifact missing for that line's tag | Cal-time tag mismatch (rare with proper dedup); re-cal that line |
| Cal log ends with `"FP16 act_scales are the deployment scales; skipping post-GPTQ activation calibration."` | `--fp16_calib` path; no post-GPTQ artifact written | NORMAL for fp16dep lines. Only worry if a non-fp16dep line in the same dedup group needs the post-GPTQ artifact (means dedup key is too loose). |
| PPL = 10000+ on a t2int20a0 line | Numerical overflow in low-bit accumulator | Expected; t2int20a0 is "INVALID" in show_results.py output |

## Status-check command (use every tick)

This is the canonical one-liner for cal+inference state. Adapt the runfile name:

```bash
ps -p <LAUNCHER_PID> -o etime= 2>&1 | head -1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "cal: $(pgrep -af 'calibrate_act_scales.py' | grep -v pgrep | wc -l)"
echo "inf: $(pgrep -af 'main_for_test.py' | grep -v pgrep | wc -l)"
echo "launcher: $(pgrep -af 'run_experiments.py.*<runfile_substring>' | grep -v pgrep | wc -l)"
for f in DONE FAST CAL ERROR ERROR-CALIBRATE; do
  c=$(grep -cE "^\[$f\]" /workspace/DartQuant/data/runs/<runfile>.ini)
  echo "[$f]: $c"
done
tail -3 /tmp/<runfile>_<phase>.log 2>&1 | head -c 700
```

`tail -c 700` (byte limit) is important — these logs include 30-100 KB of carriage-return progress bars per minute.

## Don't

- Don't skip the fast pass and go straight to full eval. You'll burn 8 hours discovering an error that surfaces in 2.
- Don't strip `[FAST]` flags before the full-eval pass. Default mode picks them up; stripping is needless churn.
- Don't kill a running launcher to "restart cleaner" unless you've confirmed there's a real problem. Cal artifacts already on disk are reused; the kill mainly costs you whatever cal is mid-batch.
- Don't poll tighter than 270s during a mid-phase grind. Cache miss is wasted; nothing changes between checks anyway.
- Don't extend past the cache window during the first ~10 minutes. Early validation of the path is worth the cache-cheap tight ticks.
- Don't combine `--gptq` and `--calibrate` on a runfile that has cached GPTQ checkpoints from a prior good run. `--gptq` deletes them; `--calibrate` then has to redo cal AND a fresh GPTQ. Use only when both are intentionally being regenerated.

## Example session shape

```
T+0:00    Launch fast pass, schedule wakeup +270s
T+0:04    Status: cals progressing, no errors → wakeup +270s
T+0:11    Status: first batch ~75% GPTAQ → wakeup +270s (pre-first-artifact)
T+0:16    Status: first batch done, throughput estimable → wakeup +1500s
T+0:44    Status: mid-cal-phase, on track → wakeup +1800s
T+1:15    Status: cal phase ~70% → wakeup +1800s
T+1:46    Status: cal done, inference started → wakeup +1800s
T+2:17    Status: 24/32 [FAST] → wakeup +270s (near completion)
T+2:22    Status: 28/32, last batch in flight → wakeup +600s
T+2:32    Status: 32/32 [FAST], LAUNCH FULL-EVAL, wakeup +1800s
T+3:00    Full-eval status: 2/32 [DONE] → wakeup +1800s
T+4:00    Full-eval status: 4/32 [DONE], steady → wakeup +2400s
T+4:40    Full-eval status: 6/32 [DONE] → wakeup +3000s
... (multi-hour grind, sample every 50 min)
T+10:00   Approaching end, tighten to 600s, then 270s
T+10:30   Done; run /dart-ppl-status comparison
```

## Saving and reporting

- After each phase completes, update todos via TodoWrite.
- After the final comparison, summarize for the user: phase timings, error count, key result deltas. Use absolute PPL values (not %), and lead with the metric that signals (per `feedback_ppl_comparison`).
- If a fix to runner code happened during shepherding (e.g. dedup-key bug), call it out at the end so the user knows.
