#!/usr/bin/env python3
"""
Run static-activation-scale calibration for baseline, quarot, and dart
in parallel across GPUs.

Dart requires pre-trained R1/R2 rotations. If they are missing, the script
trains them first (on the first available GPU) before launching the three
calibrations in parallel.

Usage:
    python multi_calibration.py -m 1b --sym --kv_ex 8 --proj_ex 15 -k 8 -v 8 -g 2
    python multi_calibration.py -m 3b --sym -g 3,6,8

Runfile mode — calibrate from the same runfile used by run_experiments.py:
    python multi_calibration.py --runfile ../fake_quant/runs.txt -g 2,3,4
    python multi_calibration.py --runfile ../fake_quant/runs.txt -g 2 --gptq --dry

Only lines with --static-act are calibrated; others are silently skipped.
Each calibrated line is marked [CAL] on success or [ERROR-CALIBRATE] on
failure. Use --recalib to force re-calibration of already-calibrated lines.
"""

import argparse
import shlex
import subprocess
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fake_quant'))
import experiment_config as cfg
import runfile_flags as rf


def stream_output(proc, prefix):
    """Read process stdout/stderr line-by-line and print with a prefix tag."""
    for stream in (proc.stdout, proc.stderr):
        if stream is None:
            continue
        for line in stream:
            print(f"[{prefix}] {line}", end='', flush=True)


def run_job(cmd, env, label, cwd=None):
    """Run a subprocess, stream its output with a label prefix, return the exit code."""
    print(f"\n{'='*60}")
    print(f"[{label}] Starting: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)

    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(f"[{label}] {line}", end='', flush=True)

    proc.wait()

    if proc.returncode != 0:
        print(f"\n[{label}] FAILED (exit code {proc.returncode})")
    else:
        print(f"\n[{label}] Done.")

    return proc.returncode


def make_env(gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['HF_DATASETS_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    return env


def _suffix_weights_stats(args_list, mode):
    """Append _<mode> before the extension of --weights_stats path in an arg list."""
    result = list(args_list)
    for i, arg in enumerate(result):
        if arg == '--weights_stats' and i + 1 < len(result):
            path = result[i + 1]
            base, sep, ext = path.rpartition('.')
            if sep:
                result[i + 1] = f"{base}_{mode}.{ext}"
            else:
                result[i + 1] = f"{path}_{mode}"
            break
    return result


def build_calibrate_cmd(mode, model_short, quant_args, extra_args):
    """Build the calibrate_act_scales.py command for the given mode."""
    cmd = [
        sys.executable, 'calibrate_act_scales.py',
        '--mode', mode,
        '-m', model_short,
    ]
    if mode == 'dart':
        cmd += ['--r1', '--r2']
    cmd += _suffix_weights_stats(quant_args, mode) + extra_args
    return cmd


def get_calibrate_help():
    """Get the help text from calibrate_act_scales.py."""
    result = subprocess.run(
        [sys.executable, 'calibrate_act_scales.py', '-h'],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.stdout


def parse_args():
    # If -h/--help is in args, show both our help and calibrate's help
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        calibrate_help = get_calibrate_help()
        extra_help = (
            "\n\nExtra arguments (e.g. --nsamples, --seqlen) are forwarded "
            "to calibrate_act_scales.py.\n"
            "Below is its help output:\n"
            + "=" * 60 + "\n"
            + calibrate_help
        )
    else:
        extra_help = ""

    parser = argparse.ArgumentParser(
        description='Run baseline/quarot/dart calibrations in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_calibration.py -m 1b --sym --kv_ex 8 --proj_ex 15 -k 8 -v 8 -g 2
  python multi_calibration.py -m 3b --sym -g 3,6,8
""" + extra_help)

    cfg.add_model_arg(parser)
    parser._option_string_actions['-m'].required = False
    parser.add_argument('-g', '--gpus', type=str, default='1',
                        help='GPU IDs: a single number S (expands to S,S+1,S+2) '
                             'or a comma-separated list like "3,6,8" (default: "1")')
    cfg.add_quant_args(parser)
    parser.add_argument('--dry', action='store_true',
                        help='Print commands without running them')
    parser.add_argument('--runfile', type=str, default=None,
                        help='Path to a runfile (same format as run_experiments.py). '
                             'Only --static-act lines are calibrated. '
                             'When set, -m and quant args are taken from each line.')
    parser.add_argument('--gptq', action='store_true',
                        help='Delete cached GPTQ checkpoint and re-quantize (forwarded to calibrate)')
    parser.add_argument('--recalib', action='store_true',
                        help='Force re-calibration regardless of existing [CAL]/[FAST]/[DONE]/[ERROR] flags')

    # Calibration-specific extras (--nsamples, --seqlen, etc.) remain in extra_args
    return parser.parse_known_args()


def parse_gpus(gpu_str, n_needed=3):
    """Parse GPU specification into a list of GPU IDs."""
    parts = gpu_str.split(',')
    if len(parts) == 1:
        s = int(parts[0])
        return list(range(s, s + n_needed))
    return [int(p) for p in parts]


def run_batch(jobs, gpus, dry=False, cwd=None):
    """Run a list of (label, cmd) jobs in batches of len(gpus).

    Each batch runs in parallel (one job per GPU), then waits for all to finish
    before starting the next batch. Returns dict of label -> exit code.
    """
    all_results = {}

    for batch_start in range(0, len(jobs), len(gpus)):
        batch = jobs[batch_start:batch_start + len(gpus)]

        if dry:
            for i, (label, cmd) in enumerate(batch):
                gpu = gpus[i % len(gpus)]
                print(f"  [GPU {gpu}] {label}: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
            continue

        threads = []
        results = {}

        def worker(label, cmd, gpu_id):
            results[label] = run_job(cmd, make_env(gpu_id), label, cwd=cwd)

        for i, (label, cmd) in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            print(f"  [GPU {gpu}] {label}: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
            t = threading.Thread(target=worker, args=(label, cmd, gpu), daemon=True)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_results.update(results)

    return all_results


def _parse_runfile_line_args(tokens):
    """Parse runfile tokens through experiment_config's argparser.

    Returns (parsed_args, extra_args) where parsed_args has model/quant fields
    and extra_args are unknown flags forwarded to calibrate_act_scales.py.
    """
    parser = argparse.ArgumentParser(add_help=False)
    cfg.add_model_arg(parser)
    cfg.add_quant_args(parser)
    return parser.parse_known_args(tokens)


# Flags from the shell script that are irrelevant to calibration
_IGNORE_FLAGS = {'--static-act', '--fast', '-F', '--very-fast', '--overwrite',
                 '--stochastic_quant', '--ig_compare'}
# Flags with a value argument that should be stripped for calibration
_IGNORE_FLAGS_WITH_VALUE = {'--r4_stats', '--r4_stats_batches', '--semi_int_gemm'}


def _filter_runtime_flags(tokens):
    """Remove runtime-only flags (with and without values) from token list."""
    filtered = []
    skip_next = False
    for t in tokens:
        if skip_next:
            skip_next = False
            continue
        if t in _IGNORE_FLAGS:
            continue
        if t in _IGNORE_FLAGS_WITH_VALUE:
            skip_next = True
            continue
        filtered.append(t)
    return filtered


def parse_runfile_for_calibration(path, recalib=False):
    """Parse a runfile and return groups of --static-act lines that need calibration.

    Groups lines by (mode, quant_tag). A group is returned only if at least
    one contributing line needs calibration according to
    ``runfile_flags.needs_calibration(flag, recalib)``.

    Returns list of dicts:
        {'mode': str, 'model_short': str, 'quant_args': [...],
         'extra_args': [...], 'quant_tag': str, 'names': [str, ...]}

    ``names`` contains every runfile line name that maps to this group,
    including ones already calibrated — the caller uses it to mark all
    contributing lines when a calibration completes (or fails).
    """
    entries = rf.parse_runfile(path)
    groups = {}  # key (mode, quant_tag) -> group dict
    order = []  # preserve insertion order for deterministic output

    for flag, name, cmd_args in entries:
        tokens = shlex.split(cmd_args)

        # Only calibrate static-act runs
        if '--static-act' not in tokens:
            continue

        mode = tokens[0]
        rest = tokens[1:]

        filtered = _filter_runtime_flags(rest)
        line_args, extra = _parse_runfile_line_args(filtered)
        cfg.resolve_v_bits(line_args)
        line_args.model = cfg.resolve_model(line_args.model)

        quant_tag = cfg.build_quant_tag(line_args)
        key = (mode, quant_tag)

        if key not in groups:
            groups[key] = {
                'mode': mode,
                'model_short': line_args.model,
                'quant_args': cfg.build_quant_args(line_args),
                'extra_args': extra,
                'quant_tag': quant_tag,
                'names': [],
                'flags': [],
            }
            order.append(key)
        groups[key]['names'].append(name)
        groups[key]['flags'].append(flag)

    # Keep only groups where at least one member needs calibration.
    result = []
    for key in order:
        g = groups[key]
        if any(rf.needs_calibration(flag, recalib=recalib) for flag in g['flags']):
            # Drop the transient 'flags' field before returning.
            g.pop('flags', None)
            result.append(g)
    return result


def ensure_r1r2(model_names, gpus, dry=False, cwd=None):
    """Train R1/R2 for any model that needs it."""
    for model_name in model_names:
        if cfg.r1r2_exist(model_name):
            print(f"R1/R2 already exist for {model_name}. Skipping training.")
            continue
        model_full = cfg.resolve_model(model_name)
        r1r2_cmd = ['bash', 'calibrate_model.sh', '-m', model_full, '-g', str(gpus[0])]
        print(f"R1/R2 not found for {model_name}. Training on GPU {gpus[0]} first...")
        if dry:
            print(f"  CUDA_VISIBLE_DEVICES={gpus[0]} {' '.join(r1r2_cmd)}")
        else:
            ret = run_job(r1r2_cmd, make_env(gpus[0]), f'R1/R2 train ({model_name})', cwd=cwd)
            if ret != 0:
                print(f"R1/R2 training failed for {model_name}. Aborting.")
                sys.exit(1)


def run_from_runfile(args, extra_args):
    """Run calibrations from a runfile.

    Marks each contributing runfile line with [CAL] on success or
    [ERROR-CALIBRATE] on failure. With --recalib, calibrates every
    static-act group regardless of existing flags.
    """
    groups = parse_runfile_for_calibration(args.runfile, recalib=args.recalib)
    if not groups:
        print("No calibration groups need running (all static-act lines already calibrated).")
        return

    gpus = parse_gpus(args.gpus, n_needed=len(groups))

    print(f"=== {len(groups)} calibration group(s) from {args.runfile} ===\n")

    # Phase 1: Train R1/R2 if any dart groups need it
    dart_models = set()
    for g in groups:
        if g['mode'] == 'dart':
            model_full = cfg.resolve_model(g['model_short'])
            dart_models.add(cfg.model_name_from_path(model_full))
    if dart_models:
        ensure_r1r2(dart_models, gpus, dry=args.dry)

    # Phase 2: Build calibration commands
    jobs = []
    label_to_names = {}
    for g in groups:
        all_extra = list(g['extra_args']) + list(extra_args)
        if args.gptq:
            all_extra.append('--gptq')
        cmd = build_calibrate_cmd(g['mode'], g['model_short'], g['quant_args'], all_extra)
        label = f"{g['mode']}:{g['quant_tag']}"
        jobs.append((label, cmd))
        label_to_names[label] = list(g['names'])

    # Phase 3: Run in batches
    results = run_batch(jobs, gpus, dry=args.dry)

    if args.dry:
        return

    # Phase 4: Mark runfile per group outcome
    for label, rc in results.items():
        new_flag = rf.FLAG_CAL if rc == 0 else rf.FLAG_ERROR_CAL
        for name in label_to_names.get(label, []):
            rf.mark_runfile(args.runfile, name, new_flag)

    # Summary
    print(f"\n{'='*60}")
    print("Calibration summary:")
    print(f"{'='*60}")
    for label, _ in jobs:
        rc = results.get(label)
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        names = label_to_names.get(label, [])
        print(f"  {label:40s} : {status}  ({len(names)} line(s))")


def main():
    args, extra_args = parse_args()

    if args.runfile:
        run_from_runfile(args, extra_args)
        return

    if not args.model:
        print("Error: -m/--model is required (unless using --runfile)")
        sys.exit(1)

    cfg.resolve_v_bits(args)
    quant_args = cfg.build_quant_args(args)

    gpus = parse_gpus(args.gpus, n_needed=3)
    if len(gpus) < 3:
        print(f"Error: need at least 3 GPUs for standard mode, got {len(gpus)}")
        sys.exit(1)

    model_short = args.model

    # Resolve model name for R1/R2 existence check
    model_full = cfg.resolve_model(model_short)
    model_name = cfg.model_name_from_path(model_full)

    modes = ['dart', 'quarot', 'baseline']

    # --- Phase 1: Train R1/R2 if needed ---
    ensure_r1r2({model_name}, gpus, dry=args.dry)

    # --- Phase 2: Run all three calibrations in parallel ---
    gptq_extra = ['--gptq'] if args.gptq else []
    jobs = []
    for mode in modes:
        cmd = build_calibrate_cmd(mode, model_short, quant_args, extra_args + gptq_extra)
        jobs.append((mode, cmd))

    print()
    results = run_batch(jobs, gpus, dry=args.dry)

    if args.dry:
        return

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Calibration summary:")
    print(f"{'='*60}")
    for mode in modes:
        status = "OK" if results.get(mode) == 0 else f"FAILED (exit {results.get(mode, '?')})"
        print(f"  {mode:10s} : {status}")


if __name__ == '__main__':
    main()
