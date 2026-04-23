#!/usr/bin/env python3
"""
Run the 7 standard DartQuant experiments in parallel.

Replaces Script/run_experiments_after_calibrate.sh.  Shares argument
definitions with calibrater/multi_calibration.py via experiment_config.

Usage:
    python run_experiments.py -m 1b -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15
    python run_experiments.py -m 1b --calibrate
    python run_experiments.py -m 3b -w 4 -a 8 -k 4 --sym --fast

Runfile mode — run diverse experiments from an external file:
    python run_experiments.py --runfile runs.txt -g 2,3,4

  Runfile format (one run per line):
    name: MODE [args for dart_gptq_wxaykvz.sh, WITHOUT -g]
    # lines starting with # or ; are comments
    # run state is tracked via a [FLAG] prefix, written automatically:
    [CAL] name: ...             calibration done, inference pending
    [FAST] name: ...            fast inference done (re-run if --fast not set)
    [DONE] name: ...            thorough inference done; skipped
    [ERROR] name: ...           inference failed (calibration OK) — retried
    [ERROR-CALIBRATE] name: ... calibration failed — retried with --calibrate
    Use --recalib to force a full re-run (calibration + inference) regardless.
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import experiment_config as cfg
import runfile_flags as rf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'cached_results')


def _read_last_line(path):
    """Read the last non-empty line from a log file (handles \\r from tqdm)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)  # end
            size = f.tell()
            if size == 0:
                return ""
            f.seek(max(0, size - 4096))
            tail = f.read().decode('utf-8', errors='replace')
        # tqdm uses \r for in-place updates; split on both \r and \n
        lines = re.split(r'[\r\n]', tail)
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                return stripped
    except (OSError, ValueError):
        pass
    return ""


def _print_status(procs, first_call=False):
    """Print / refresh a fixed-size status block (one line per slot).

    `procs` is length = capacity.  Each entry is (name, proc, log_path) or
    None for an idle slot.  The set of entries is stable across calls, so
    cursor-up `\\033[nA` safely rewrites the block in place.
    """
    term_width = shutil.get_terminal_size((120, 24)).columns
    n = len(procs)
    if not first_call:
        sys.stdout.write(f"\033[{n}A")

    for entry in procs:
        if entry is None:
            line = "  [idle]"
        else:
            name, proc, log_path = entry
            rc = proc.poll()
            if rc is None:
                tail = _read_last_line(log_path)
                if tail:
                    prefix = f"  [running] {name}: "
                    max_tail = term_width - len(prefix)
                    if len(tail) > max_tail:
                        tail = "..." + tail[-(max_tail - 3):]
                    line = prefix + tail
                else:
                    line = f"  [running] {name}: starting..."
            elif rc == 0:
                tail = _read_last_line(log_path)
                if tail:
                    prefix = f"  [done]    {name}: "
                    max_tail = term_width - len(prefix)
                    if len(tail) > max_tail:
                        tail = "..." + tail[-(max_tail - 3):]
                    line = prefix + tail
                else:
                    line = f"  [done]    {name}"
            else:
                line = f"  [FAIL]    {name} (exit code {rc})"
        # Clear rest of line in case previous line was longer
        sys.stdout.write(f"\033[K{line}\n")
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run 7 DartQuant experiments in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments launched (7 total):
  full, baseline, quarot, dart,
  baseline_static, quarot_static, dart_static

Examples:
  python run_experiments.py -m 1b -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15
  python run_experiments.py -m 1b --calibrate
  python run_experiments.py -m 3b --fast
""")
    # -m is required for standard mode but not for --runfile
    cfg.add_model_arg(parser)
    parser._option_string_actions['-m'].required = False
    cfg.add_quant_args(parser)

    parser.add_argument('-g', '--gpus', type=str, default='1',
                        help='GPU IDs: a single number S (expands to S,S+1,...,S+6) '
                             'or a comma-separated list like "1,3,5,7,2,4,6" (default: "1")')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run calibration before experiments (runfile mode: '
                             'static-act calibration; standard mode: R1/R2 training)')
    parser.add_argument('--recalib', action='store_true',
                        help='Force re-calibration AND re-inference for all selected '
                             'runfile lines, ignoring existing [DONE]/[FAST]/[CAL] flags. '
                             'Implies --calibrate in runfile mode.')
    parser.add_argument('--calibrate_gpu', type=int, default=None,
                        help='GPU for calibration (default: first of --gpus)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Ignore cached results, re-run all')
    parser.add_argument('--gptq', action='store_true',
                        help='Delete cached GPTQ checkpoint and re-quantize')
    parser.add_argument('-F', '--fast', action='store_true',
                        help='Fast mode (skip lm_eval tasks)')
    parser.add_argument('--very-fast', action='store_true',
                        help='Very fast mode (skip lm_eval, PPL on wikitext2 only)')

    parser.add_argument('--skip', type=str, default=None,
                        help='Experiments to skip (1-indexed by print order): '
                             'a single number like "4" or comma-separated like "3,6"')
    parser.add_argument('--wait', action='store_true',
                        help='Dynamic GPU scheduling: each of the N=len(--gpus) slots '
                             'polls nvidia-smi (via utils/gpu_wait.py) and launches on '
                             'the first GPU that falls below --max_used_mb. Replaces '
                             'the old --sequential (use --wait -g <single> for 1-at-a-time).')
    parser.add_argument('--max_used_mb', type=int, default=200,
                        help='Max used memory (MiB) for --wait GPU selection (default: 200)')
    parser.add_argument('--refresh', type=int, default=10,
                        help='Status refresh interval in seconds (default: 10)')
    parser.add_argument('--dry', action='store_true',
                        help='Print commands without running them')

    parser.add_argument('--runfile', type=str, default=None,
                        help='Path to a runfile with custom per-run commands. '
                             'When set, -m and quant args are ignored; runs are '
                             'read from the file instead. See module docstring.')

    return parser.parse_args()


def build_experiment_cmd(mode, quant_args, extra_flags, args):
    """Build the dart_gptq_wxaykvz.sh command (no -g; scheduler appends it)."""
    script = os.path.join(SCRIPT_DIR, 'Script', 'dart_gptq_wxaykvz.sh')
    cmd = [script, mode, '-m', args.model]

    if mode != 'full':
        cmd += quant_args
        cmd += extra_flags
    else:
        # GGUF affects even the full-precision run (replaces base weights)
        if getattr(args, 'gguf', None):
            cmd += ['--gguf', args.gguf]

    if args.overwrite:
        cmd.append('--overwrite')
    if args.gptq:
        cmd.append('--gptq')
    if args.very_fast:
        cmd.append('--very-fast')
    elif args.fast:
        cmd.append('--fast')

    return cmd


def run_calibration(model, gpu_id):
    """Run R1/R2 calibration via calibrate_model.sh."""
    script = os.path.join(SCRIPT_DIR, '..', 'calibrater', 'calibrate_model.sh')
    model_full = cfg.resolve_model(model)
    cmd = ['bash', script, '-m', model_full, '-g', str(gpu_id)]
    print(f"=== Running calibration: {' '.join(cmd)} ===")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("Calibration failed!")
        sys.exit(1)
    print()


# ── Runfile calibration (calls multi_calibration functions) ──────────────

CALIBRATER_DIR = os.path.join(SCRIPT_DIR, '..', 'calibrater')


def run_calibration_for_runfile(args):
    """Run static-act calibration for all runfile lines that need it.

    Marks each runfile line with [CAL] on success or [ERROR-CALIBRATE] on
    failure. Does NOT abort on calibration failure — the downstream
    inference phase skips [ERROR-CALIBRATE] lines via needs_inference().
    """
    sys.path.insert(0, CALIBRATER_DIR)
    from multi_calibration import (
        parse_runfile_for_calibration, ensure_r1r2, build_calibrate_cmd,
        run_batch,
    )

    # Returns list of group dicts with 'names' for per-line flag writeback.
    groups = parse_runfile_for_calibration(args.runfile, recalib=args.recalib)
    if not groups:
        print("No calibration groups need running (all static-act lines already calibrated).")
        return

    # Compute GPU list — expand scalar to len(groups)
    gpu_str = args.gpus
    parts = gpu_str.split(',')
    if len(parts) == 1:
        s = int(parts[0])
        gpus = list(range(s, s + len(groups)))
    else:
        gpus = [int(p) for p in parts]

    print(f"\n=== Calibration: {len(groups)} group(s) from {args.runfile} ===\n")

    # Ensure R1/R2 exist for dart models
    dart_models = set()
    for g in groups:
        if g['mode'] == 'dart':
            model_full = cfg.resolve_model(g['model_short'])
            dart_models.add(cfg.model_name_from_path(model_full))
    if dart_models:
        ensure_r1r2(dart_models, gpus, dry=args.dry, cwd=CALIBRATER_DIR)

    # Build calibration commands; remember which runfile names each label covers.
    jobs = []
    label_to_names = {}
    for g in groups:
        extra = list(g['extra_args'])
        if args.gptq:
            extra.append('--gptq')
        cmd = build_calibrate_cmd(g['mode'], g['model_short'], g['quant_args'], extra)
        label = f"cal:{g['mode']}:{g['quant_tag']}"
        jobs.append((label, cmd))
        label_to_names[label] = list(g['names'])

    results = run_batch(jobs, gpus, dry=args.dry, cwd=CALIBRATER_DIR)

    if args.dry:
        return

    # Mark each contributing runfile line with [CAL] or [ERROR-CALIBRATE].
    failed_labels = []
    for label, rc in results.items():
        new_flag = rf.FLAG_CAL if rc == 0 else rf.FLAG_ERROR_CAL
        for name in label_to_names.get(label, []):
            rf.mark_runfile(args.runfile, name, new_flag)
        if rc != 0:
            failed_labels.append(label)

    if failed_labels:
        print(f"\n=== Calibration FAILED for: {', '.join(failed_labels)} ===")
        print("Affected lines marked [ERROR-CALIBRATE]; inference will skip them.")
    else:
        print(f"\n=== All {len(groups)} calibration group(s) completed successfully ===\n")


# ── Runfile helpers ──────────────────────────────────────────────────────
# Flag parsing/writing lives in runfile_flags.py (shared with multi_calibration).


def _pick_gpu(args, gpus, slot_idx):
    """Select a GPU for the next slot.

    Under --wait, block on `wait_for_gpu` (via utils/gpu_wait.py) until one
    falls below `args.max_used_mb`.  Otherwise round-robin through `gpus`.
    """
    if getattr(args, 'wait', False):
        sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'utils'))
        from gpu_wait import wait_for_gpu
        return wait_for_gpu(max_used_mb=args.max_used_mb, poll_interval=30)
    return gpus[slot_idx % len(gpus)]


def _run_jobs(active_cmds, gpus, args, log_fn, on_complete=None):
    """Dynamic scheduler: up to `capacity = len(gpus)` jobs in flight.

    Fixed-size `slots` list (length = capacity) of (name, proc, log_path, log_f)
    or None for idle.  On each pass: notify completed jobs (on_complete +
    failure tally), refill any done/idle slots from the pending queue,
    redraw the status block in place, sleep, repeat.  Keeps the stable
    per-slot line count so cursor-up refresh works exactly as before.

    Args:
        active_cmds: list of (name, cmd_without_g) — non-skipped runs.
        gpus: list of GPU IDs (only len(gpus) matters under --wait).
        args: parsed CLI args (reads .wait, .max_used_mb, .refresh).
        log_fn: callable(name) -> log file path.
        on_complete: optional callable(name, returncode) called per finished run.

    Returns total number of failures.
    """
    capacity = len(gpus)
    pending = list(active_cmds)
    slots = [None] * capacity
    total_failed = 0
    launched = 0
    notified = set()
    refresh = args.refresh
    wait_mode = getattr(args, 'wait', False)

    def notify(i):
        nonlocal total_failed
        entry = slots[i]
        if entry is None:
            return
        name, proc, _, log_f = entry
        if proc.poll() is not None and name not in notified:
            notified.add(name)
            log_f.close()
            if on_complete:
                on_complete(name, proc.returncode)
            if proc.returncode != 0:
                total_failed += 1

    def launch_into(i):
        nonlocal launched
        name, cmd = pending.pop(0)
        gpu = _pick_gpu(args, gpus, launched)
        full_cmd = list(cmd) + ['-g', str(gpu)]
        log_path = log_fn(name)
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(full_cmd, stdout=log_f, stderr=subprocess.STDOUT)
        slots[i] = (name, proc, log_path, log_f)
        launched += 1

    print(f"\n=== {len(active_cmds)} runs, capacity={capacity} "
          f"(refreshing every {refresh}s) ===\n")

    first_call = True
    while pending or any(s is not None and s[0] not in notified for s in slots):
        # Notify finished slots (idempotent) and refill idle / done slots.
        for i in range(capacity):
            notify(i)
            if pending and (slots[i] is None
                            or slots[i][1].poll() is not None):
                slots[i] = None
                launch_into(i)
                if pending and wait_mode:
                    # Inter-launch cooldown so the previous allocation
                    # shows up in nvidia-smi before the next wait_for_gpu.
                    time.sleep(refresh)

        # Redraw fixed-height status block
        status = [(s[0], s[1], s[2]) if s is not None else None for s in slots]
        _print_status(status, first_call=first_call)
        first_call = False

        # Exit if everything is done
        if not pending and all(s is None or s[1].poll() is not None
                               for s in slots):
            for i in range(capacity):
                notify(i)
            break

        time.sleep(refresh)

    return total_failed


def run_from_file(args):
    """Run experiments defined in a runfile."""
    runfile = args.runfile
    fast_mode = bool(args.fast or args.very_fast)
    calibrate_requested = bool(args.calibrate or args.recalib)

    entries = rf.parse_runfile(runfile)
    if not entries:
        print("No runs found in runfile.")
        return

    # --- Optional calibration before experiments ---
    # Calibration phase marks each touched line with [CAL] / [ERROR-CALIBRATE].
    if calibrate_requested:
        run_calibration_for_runfile(args)
        gptq_for_experiments = False   # consumed by calibration
        # Re-parse to pick up freshly-written [CAL] / [ERROR-CALIBRATE] flags.
        entries = rf.parse_runfile(runfile)
    else:
        gptq_for_experiments = args.gptq

    # Filter inference-eligible entries based on flag + fast/recalib state.
    inference_entries = []
    cal_error_skipped = []
    done_skipped = []
    for flag, name, cmd_args in entries:
        if rf.needs_inference(flag, fast=fast_mode, recalib=args.recalib):
            inference_entries.append((flag, name, cmd_args))
        elif flag == rf.FLAG_ERROR_CAL:
            cal_error_skipped.append(name)
        else:
            done_skipped.append((flag, name))

    if cal_error_skipped:
        print(f"\nWARNING: {len(cal_error_skipped)} line(s) are [ERROR-CALIBRATE]; "
              f"inference skipped. Re-run with --calibrate (or --recalib) to retry.")
        for name in cal_error_skipped:
            print(f"  - {name}")

    if done_skipped:
        print(f"\nSkipping {len(done_skipped)} already-completed line(s):")
        for flag, name in done_skipped:
            print(f"  - [{flag}] {name}")

    if not inference_entries:
        print("\nNo inference runs to execute.")
        return

    # Parse GPUs
    gpu_str = args.gpus
    parts = gpu_str.split(',')
    if len(parts) == 1:
        s = int(parts[0])
        gpus = list(range(s, s + len(inference_entries)))
    else:
        gpus = [int(p) for p in parts]

    # Parse skip set (1-indexed over inference_entries)
    skip_set = set()
    if args.skip:
        skip_set = {int(x) for x in args.skip.split(',')}

    script = os.path.join(SCRIPT_DIR, 'Script', 'dart_gptq_wxaykvz.sh')

    print(f"\n=== {len(inference_entries)} inference run(s) from {runfile} ===\n")

    # Build all commands (no -g; scheduler assigns at launch time)
    cmds = []
    for i, (flag, name, cmd_args) in enumerate(inference_entries):
        idx = i + 1
        cmd = [script] + shlex.split(cmd_args)
        if args.overwrite:
            cmd.append('--overwrite')
        if gptq_for_experiments:
            cmd.append('--gptq')
        if args.very_fast:
            cmd.append('--very-fast')
        elif args.fast:
            cmd.append('-F')
        flag_note = f" [{flag}→rerun]" if flag else ""
        gpu_note = "wait" if getattr(args, 'wait', False) else f"GPU {gpus[i % len(gpus)]}"
        if idx in skip_set:
            cmds.append((name, cmd, True))
            print(f"  [skipped] {name}{flag_note}: {' '.join(cmd)}")
        else:
            cmds.append((name, cmd, False))
            print(f"  [{gpu_note}] {name}{flag_note}: {' '.join(cmd)}")

    if args.dry:
        return

    # Clean old log files
    os.makedirs(RESULTS_DIR, exist_ok=True)
    active_all = [(name, cmd) for name, cmd, skipped in cmds if not skipped]
    for name, _ in active_all:
        log_path = os.path.join(RESULTS_DIR, f'{name}.log')
        if os.path.exists(log_path):
            os.remove(log_path)

    def on_complete(name, rc):
        new_flag = rf.inference_completion_flag(fast_mode) if rc == 0 else rf.FLAG_ERROR
        rf.mark_runfile(runfile, name, new_flag)

    total_failed = _run_jobs(
        active_all, gpus, args,
        log_fn=lambda name: os.path.join(RESULTS_DIR, f'{name}.log'),
        on_complete=on_complete,
    )

    print()
    if total_failed == 0:
        print(f"=== All {len(active_all)} runs completed successfully ===")
    else:
        print(f"=== {total_failed} run(s) failed out of {len(active_all)} ===")


def main():
    args = parse_args()

    if args.runfile:
        run_from_file(args)
        return

    if not args.model:
        print("Error: -m/--model is required (unless using --runfile)")
        sys.exit(1)

    cfg.resolve_v_bits(args)
    quant_args = cfg.build_quant_args(args)

    experiments = cfg.EXPERIMENTS
    num_exp = len(experiments)

    # Parse GPU specification
    gpu_str = args.gpus
    parts = gpu_str.split(',')
    if len(parts) == 1:
        # Single number S -> S, S+1, ..., S+num_exp-1
        s = int(parts[0])
        gpus = list(range(s, s + num_exp))
    else:
        gpus = [int(p) for p in parts]

    # --- Optional calibration ---
    if args.calibrate:
        cal_gpu = args.calibrate_gpu if args.calibrate_gpu is not None else gpus[0]
        if args.dry:
            script = os.path.join(SCRIPT_DIR, '..', 'calibrater', 'calibrate_model.sh')
            model_full = cfg.resolve_model(args.model)
            print(f"bash {script} -m {model_full} -g {cal_gpu}")
        else:
            run_calibration(args.model, cal_gpu)

    # Parse skip set (1-indexed)
    skip_set = set()
    if args.skip:
        skip_set = {int(x) for x in args.skip.split(',')}

    # --- Build and optionally print commands ---
    print(f"=== {len(experiments)} experiments ===\n")

    cmds = []
    for i, (name, mode, extra_flags) in enumerate(experiments):
        idx = i + 1  # 1-indexed
        cmd = build_experiment_cmd(mode, quant_args, extra_flags, args)
        gpu_note = "wait" if getattr(args, 'wait', False) else f"GPU {gpus[i % len(gpus)]}"
        if idx in skip_set:
            cmds.append((name, cmd, True))
            print(f"  [skipped] {name}: {' '.join(cmd)}")
        else:
            cmds.append((name, cmd, False))
            print(f"  [{gpu_note}] {name}: {' '.join(cmd)}")

    if args.dry:
        return

    # --- Clean old result files ---
    for name, _, skipped in cmds:
        if skipped:
            continue
        for ext in ('log', 'out', 'err'):
            path = os.path.join(RESULTS_DIR, f'{name}_results.{ext}')
            if os.path.exists(path):
                os.remove(path)

    # --- Launch experiments ---
    print()
    active_cmds = [(name, cmd) for name, cmd, skipped in cmds if not skipped]
    num_run = len(active_cmds)

    failed = _run_jobs(
        active_cmds, gpus, args,
        log_fn=lambda name: os.path.join(RESULTS_DIR, f'{name}_results.log'),
    )

    num_skipped = len(skip_set)
    print()
    if failed == 0:
        msg = f"=== All {num_run} experiments completed successfully ==="
        if num_skipped:
            msg += f" ({num_skipped} skipped)"
        print(msg)
    else:
        print(f"=== {failed} experiment(s) failed ===")
        sys.exit(1)


if __name__ == '__main__':
    main()
