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

The first GPU is used for dart (and for R1/R2 training if needed), the second
for quarot, the third for baseline.
"""

import argparse
import subprocess
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import experiment_config as cfg


def stream_output(proc, prefix):
    """Read process stdout/stderr line-by-line and print with a prefix tag."""
    for stream in (proc.stdout, proc.stderr):
        if stream is None:
            continue
        for line in stream:
            print(f"[{prefix}] {line}", end='', flush=True)


def run_job(cmd, env, label):
    """Run a subprocess, stream its output with a label prefix, return the exit code."""
    print(f"\n{'='*60}")
    print(f"[{label}] Starting: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)

    proc = subprocess.Popen(
        cmd,
        env=env,
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


def build_calibrate_cmd(mode, model_short, quant_args, extra_args):
    """Build the calibrate_act_scales.py command for the given mode."""
    cmd = [
        sys.executable, 'calibrate_act_scales.py',
        '--mode', mode,
        '-m', model_short,
    ]
    if mode == 'dart':
        cmd += ['--r1', '--r2']
    cmd += quant_args + extra_args
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
    parser.add_argument('-g', '--gpus', type=str, default='1',
                        help='GPU IDs: a single number S (expands to S,S+1,S+2) '
                             'or a comma-separated list like "3,6,8" (default: "1")')
    cfg.add_quant_args(parser)
    parser.add_argument('--dry', action='store_true',
                        help='Print commands without running them')

    # Calibration-specific extras (--nsamples, --seqlen, etc.) remain in extra_args
    return parser.parse_known_args()


def main():
    args, extra_args = parse_args()
    cfg.resolve_v_bits(args)
    quant_args = cfg.build_quant_args(args)

    # Parse GPU specification
    gpu_str = args.gpus
    parts = gpu_str.split(',')
    if len(parts) == 1:
        # Single number S -> S, S+1, S+2
        s = int(parts[0])
        gpus = [s, s + 1, s + 2]
    elif len(parts) == 3:
        gpus = [int(p) for p in parts]
    else:
        print(f"Error: --gpus expects a single number or exactly 3 comma-separated IDs, got '{gpu_str}'")
        sys.exit(1)

    model_short = args.model

    # Resolve model name for R1/R2 existence check
    model_full = cfg.resolve_model(model_short)
    model_name = cfg.model_name_from_path(model_full)

    modes = ['dart', 'quarot', 'baseline']

    # --- Phase 1: Train R1/R2 if needed (blocks dart GPU) ---
    r1r2_cmd = ['bash', 'calibrate_model.sh', '-m', model_full, '-g', str(gpus[0])]
    if not cfg.r1r2_exist(model_name):
        print(f"R1/R2 not found for {model_name}. Training on GPU {gpus[0]} first...")
        if args.dry:
            print(f"  CUDA_VISIBLE_DEVICES={gpus[0]} {' '.join(r1r2_cmd)}")
        else:
            ret = run_job(r1r2_cmd, make_env(gpus[0]), 'R1/R2 train')
            if ret != 0:
                print("R1/R2 training failed. Aborting.")
                sys.exit(1)
    else:
        print(f"R1/R2 already exist for {model_name}. Skipping training.")

    # --- Phase 2: Run all three calibrations in parallel ---
    # Assign GPUs round-robin: mode[i] -> gpus[i % len(gpus)]
    print()
    for i, mode in enumerate(modes):
        gpu = gpus[i % len(gpus)]
        cmd = build_calibrate_cmd(mode, model_short, quant_args, extra_args)
        print(f"  [GPU {gpu}] {mode}: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")

    if args.dry:
        return

    threads = []
    results = {}

    def worker(mode, gpu_id):
        cmd = build_calibrate_cmd(mode, model_short, quant_args, extra_args)
        results[mode] = run_job(cmd, make_env(gpu_id), mode)

    for i, mode in enumerate(modes):
        gpu = gpus[i % len(gpus)]
        t = threading.Thread(target=worker, args=(mode, gpu), daemon=True)
        threads.append((mode, t))

    for _, t in threads:
        t.start()
    for _, t in threads:
        t.join()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Calibration summary:")
    print(f"{'='*60}")
    for mode in modes:
        status = "OK" if results.get(mode) == 0 else f"FAILED (exit {results.get(mode, '?')})"
        print(f"  {mode:10s} : {status}")


if __name__ == '__main__':
    main()
