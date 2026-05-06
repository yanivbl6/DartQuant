#!/usr/bin/env python3
# Wrapped by skill: .claude/skills/dart-run-experiments — update SKILL.md if this script's CLI / runfile semantics change.
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
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import experiment_config as cfg
import runfile_flags as rf
from parallel_runner import run_parallel_batches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'cached_results')


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
    parser.add_argument('--sequential', action='store_true',
                        help='Run experiments one at a time, each with --wait for a clear GPU')
    parser.add_argument('--max_used_mb', type=int, default=200,
                        help='Max used memory (MiB) for --sequential GPU waiting (default: 200)')
    parser.add_argument('--refresh', type=int, default=10,
                        help='Status refresh interval in seconds (default: 10)')
    parser.add_argument('--dry', action='store_true',
                        help='Print commands without running them')

    parser.add_argument('--runfile', type=str, default=None,
                        help='Path to a runfile with custom per-run commands. '
                             'When set, -m and quant args are ignored; runs are '
                             'read from the file instead. See module docstring.')

    return parser.parse_args()


def build_experiment_cmd(mode, gpu, quant_args, extra_flags, args, use_wait=False):
    """Build the dart_gptq_wxaykvz.sh command for one experiment."""
    script = os.path.join(SCRIPT_DIR, 'Script', 'dart_gptq_wxaykvz.sh')
    if use_wait:
        cmd = [script, mode, '--wait', '--max_used_mb', str(args.max_used_mb), '-m', args.model]
    else:
        cmd = [script, mode, '-g', str(gpu), '-m', args.model]

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


def run_calibration(model, gpu_id, refresh=10):
    """Run R1/R2 calibration via calibrate_model.sh through the shared runner."""
    from parallel_runner import calibration_fail_line

    script = os.path.join(SCRIPT_DIR, '..', 'calibrater', 'calibrate_model.sh')
    model_full = cfg.resolve_model(model)
    model_name = cfg.model_name_from_path(model_full)
    cmd = ['bash', script, '-m', model_full, '-g', str(gpu_id)]
    label = f"cal_r1r2_{model_name}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f'{label}.log')
    print(f"=== Running calibration: {' '.join(cmd)} ===")
    print(f"  log: {log_path}")

    failed = run_parallel_batches(
        [(label, cmd)],
        gpus=[gpu_id],
        refresh=refresh,
        log_fn=lambda name: os.path.join(RESULTS_DIR, f'{name}.log'),
        fail_message_fn=calibration_fail_line,
    )
    if failed:
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
        ensure_r1r2(dart_models, gpus, dry=args.dry, cwd=CALIBRATER_DIR,
                    log_dir=RESULTS_DIR, refresh=args.refresh)

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

    results = run_batch(jobs, gpus, dry=args.dry, cwd=CALIBRATER_DIR,
                        log_dir=RESULTS_DIR, refresh=args.refresh)

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

    # if done_skipped:
    #     print(f"\nSkipping {len(done_skipped)} already-completed line(s):")
    #     for flag, name in done_skipped:
    #         print(f"  - [{flag}] {name}")

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

    # Build all commands
    cmds = []
    for i, (flag, name, cmd_args) in enumerate(inference_entries):
        idx = i + 1
        gpu = gpus[i % len(gpus)]
        cmd = [script] + shlex.split(cmd_args) + ['-g', str(gpu)]
        if args.overwrite:
            cmd.append('--overwrite')
        if gptq_for_experiments:
            cmd.append('--gptq')
        if args.very_fast:
            cmd.append('--very-fast')
        elif args.fast:
            cmd.append('-F')
        flag_note = f" [{flag}→rerun]" if flag else ""
        if idx in skip_set:
            cmds.append((name, cmd, True))
            print(f"  [skipped] {name}{flag_note}: {' '.join(cmd)}")
        else:
            cmds.append((name, cmd, False))
            print(f"  [GPU {gpu}] {name}{flag_note}: {' '.join(cmd)}")

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

    total_failed = run_parallel_batches(
        active_all, gpus, args.refresh,
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

    cfg.apply_set_preset(args)
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
            run_calibration(args.model, cal_gpu, refresh=args.refresh)

    # Parse skip set (1-indexed)
    skip_set = set()
    if args.skip:
        skip_set = {int(x) for x in args.skip.split(',')}

    # --- Build and optionally print commands ---
    print(f"=== {len(experiments)} experiments ===\n")

    cmds = []
    for i, (name, mode, extra_flags) in enumerate(experiments):
        idx = i + 1  # 1-indexed
        gpu = gpus[i % len(gpus)]
        cmd = build_experiment_cmd(mode, gpu, quant_args, extra_flags, args,
                                   use_wait=args.sequential)
        if idx in skip_set:
            cmds.append((name, cmd, True))
            print(f"  [skipped] {name}: {' '.join(cmd)}")
        elif args.sequential:
            cmds.append((name, cmd, False))
            print(f"  [wait] {name}: {' '.join(cmd)}")
        else:
            cmds.append((name, cmd, False))
            print(f"  [GPU {gpu}] {name}: {' '.join(cmd)}")

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

    if args.sequential:
        # Run one at a time; between runs wait then poll for a clear GPU
        sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'utils'))
        from gpu_wait import wait_for_gpu

        print(f"=== Running {num_run} experiments sequentially ===\n")
        failed = 0
        for i, (name, cmd) in enumerate(active_cmds):
            print(f"  [{i+1}/{num_run}] Launching {name} ...")
            log_path = os.path.join(RESULTS_DIR, f'{name}_results.log')
            log_f = open(log_path, 'w')
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            proc.wait()
            log_f.close()
            if proc.returncode == 0:
                print(f"  [done] {name}")
            else:
                print(f"  [FAIL] {name} (see {log_path})")
                failed += 1

            # After each run (except the last), wait then poll for a clear GPU
            if i < len(active_cmds) - 1:
                print(f"  Cooling down {args.refresh}s before next experiment ...")
                time.sleep(args.refresh)
                wait_for_gpu(max_used_mb=args.max_used_mb, poll_interval=30)
    else:
        failed = run_parallel_batches(
            active_cmds, gpus, args.refresh,
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
