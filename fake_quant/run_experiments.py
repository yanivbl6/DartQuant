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

    parser.add_argument('-g', '--gpus', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7],
                        help='GPU IDs for experiments (default: 1 2 3 4 5 6 7)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run R1/R2 calibration (calibrate_model.sh) before experiments')
    parser.add_argument('--calibrate_gpu', type=int, default=None,
                        help='GPU for calibration (default: first of --gpus)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Ignore cached results, re-run all')
    parser.add_argument('--gptq', action='store_true',
                        help='Delete cached GPTQ checkpoint and re-quantize')
    parser.add_argument('-F', '--fast', action='store_true',
                        help='Fast mode (fewer eval tasks/datasets)')
    parser.add_argument('--dry', action='store_true',
                        help='Print commands without running them')

    return parser.parse_args()


def build_experiment_cmd(mode, gpu, quant_args, extra_flags, args):
    """Build the dart_gptq_wxaykvz.sh command for one experiment."""
    script = os.path.join(SCRIPT_DIR, 'Script', 'dart_gptq_wxaykvz.sh')
    cmd = [script, mode, '-g', str(gpu), '-m', args.model]

    if mode != 'full':
        cmd += quant_args
        cmd += extra_flags

    if args.overwrite:
        cmd.append('--overwrite')
    if args.gptq:
        cmd.append('--gptq')
    if args.fast:
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

    # --- Optional calibration ---
    if args.calibrate:
        cal_gpu = args.calibrate_gpu if args.calibrate_gpu is not None else args.gpus[0]
        if args.dry:
            script = os.path.join(SCRIPT_DIR, '..', 'calibrater', 'calibrate_model.sh')
            model_full = cfg.resolve_model(args.model)
            print(f"bash {script} -m {model_full} -g {cal_gpu}")
        else:
            run_calibration(args.model, cal_gpu)

    experiments = cfg.EXPERIMENTS
    gpus = args.gpus

    # --- Build and optionally print commands ---
    print(f"=== {len(experiments)} experiments ===\n")

    cmds = []
    for i, (name, mode, extra_flags) in enumerate(experiments):
        gpu = gpus[i % len(gpus)]
        cmd = build_experiment_cmd(mode, gpu, quant_args, extra_flags, args)
        cmds.append((name, cmd))
        print(f"  [GPU {gpu}] {name}: {' '.join(cmd)}")

    if args.dry:
        return

    # --- Clean old result files ---
    for name, _, _ in experiments:
        for ext in ('out', 'err'):
            path = f'/tmp/{name}_results.{ext}'
            if os.path.exists(path):
                os.remove(path)

    # --- Launch experiments in parallel ---
    print()
    procs = []
    for name, cmd in cmds:
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

    print()
    if failed == 0:
        print(f"=== All {len(experiments)} experiments completed successfully ===")
    else:
        print(f"=== {failed} experiment(s) failed ===")
        sys.exit(1)


if __name__ == '__main__':
    main()
