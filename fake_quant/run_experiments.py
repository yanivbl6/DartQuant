#!/usr/bin/env python3
"""
Run the 7 standard DartQuant experiments in parallel.

Replaces Script/run_experiments_after_calibrate.sh.  Shares argument
definitions with calibrater/multi_calibration.py via experiment_config.

Usage:
    python run_experiments.py -m 1b -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15
    python run_experiments.py -m 1b --calibrate
    python run_experiments.py -m 3b -w 4 -a 8 -k 4 --sym --fast
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import experiment_config as cfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    cfg.add_model_arg(parser)
    cfg.add_quant_args(parser)

    parser.add_argument('-g', '--gpus', type=str, default='1',
                        help='GPU IDs: a single number S (expands to S,S+1,...,S+6) '
                             'or a comma-separated list like "1,3,5,7,2,4,6" (default: "1")')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run R1/R2 calibration (calibrate_model.sh) before experiments')
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
    parser.add_argument('--dry', action='store_true',
                        help='Print commands without running them')

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
        cmd.append('-F')

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


def main():
    args = parse_args()
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
        for ext in ('out', 'err'):
            path = f'/tmp/{name}_results.{ext}'
            if os.path.exists(path):
                os.remove(path)

    # --- Launch experiments ---
    print()
    active_cmds = [(name, cmd) for name, cmd, skipped in cmds if not skipped]
    num_run = len(active_cmds)

    if args.sequential:
        # Run one at a time; between runs wait 30s then poll for a clear GPU
        sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'utils'))
        from gpu_wait import wait_for_gpu

        print(f"=== Running {num_run} experiments sequentially ===\n")
        failed = 0
        for i, (name, cmd) in enumerate(active_cmds):
            print(f"  [{i+1}/{num_run}] Launching {name} ...")
            out_f = open(f'/tmp/{name}_results.out', 'w')
            err_f = open(f'/tmp/{name}_results.err', 'w')
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
            proc.wait()
            out_f.close()
            err_f.close()
            if proc.returncode == 0:
                print(f"  [done] {name}")
            else:
                print(f"  [FAIL] {name} (see /tmp/{name}_results.err)")
                failed += 1

            # After each run (except the last), wait 30s then poll for a clear GPU
            if i < len(active_cmds) - 1:
                print(f"  Cooling down 30s before next experiment ...")
                time.sleep(30)
                wait_for_gpu(max_used_mb=args.max_used_mb, poll_interval=30)
    else:
        # Launch all in parallel
        procs = []
        for name, cmd in active_cmds:
            out_f = open(f'/tmp/{name}_results.out', 'w')
            err_f = open(f'/tmp/{name}_results.err', 'w')
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
            procs.append((name, proc, out_f, err_f))

        print(f"=== Waiting for all experiments ===\n")

        failed = 0
        for name, proc, out_f, err_f in procs:
            proc.wait()
            out_f.close()
            err_f.close()
            if proc.returncode == 0:
                print(f"  [done] {name}")
            else:
                print(f"  [FAIL] {name} (see /tmp/{name}_results.err)")
                failed += 1

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
