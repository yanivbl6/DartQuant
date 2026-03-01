#!/usr/bin/env python3
"""
Run static-activation-scale calibration for baseline, quarot, and dart
in parallel across GPUs.

Dart requires pre-trained R1/R2 rotations. If they are missing, the script
trains them first (on the first available GPU) before launching the three
calibrations in parallel.

Usage:
    python multi_calibration.py -m 1b --sym --kv_ex 8 --proj_ex 15 -k 8 --v_bits 8 -g 2 3 4
    python multi_calibration.py -m 3b --sym -g 0 1 2

The first GPU is used for dart (and for R1/R2 training if needed), the second
for quarot, the third for baseline.  With fewer GPUs the jobs are queued.
"""

import argparse
import subprocess
import sys
import os
import threading
import time


MODEL_BASE = "/data/users/sashas/LLMC/Models/meta-llama"
MODEL_MAP = {
    '1b': f'{MODEL_BASE}/Llama-3.2-1B-Instruct',
    '3b': f'{MODEL_BASE}/Llama-3.2-3B-Instruct',
    '7b': f'{MODEL_BASE}/Llama-2-7b-hf',
}

R1_PATHS = {
    'Llama-2-7b-hf':         '../data/trained_rotation/wikitext2_128samples/r1/sgd.0.0015.0.9.10.64.0.1.1',
    'Llama-3.2-1B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1',
    'Llama-3.2-3B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1',
}
R2_PATHS = {
    'Llama-2-7b-hf':         '../data/trained_rotation/wikitext2_128samples/r2/sgd.0.001.0.9.10.64.2',
    'Llama-3.2-1B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r2/sgd.0.001.0.9.10.64.2',
    'Llama-3.2-3B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r2/sgd.0.001.0.9.10.64.2',
}


def resolve_r_path(base_path):
    """Append the .pt file inside the directory, matching calibrate_act_scales.py logic."""
    if base_path and '.pt' not in base_path and '.bin' not in base_path:
        return base_path + '/' + base_path.split('/')[-1] + '.pt'
    return base_path


def r1r2_exist(model_name):
    """Check whether pre-trained R1 and R2 .pt files exist for the model."""
    r1_base = R1_PATHS.get(model_name)
    r2_base = R2_PATHS.get(model_name)
    if not r1_base or not r2_base:
        return False
    return os.path.isfile(resolve_r_path(r1_base)) and os.path.isfile(resolve_r_path(r2_base))


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


def build_calibrate_cmd(mode, model_short, extra_args):
    """Build the calibrate_act_scales.py command for the given mode."""
    cmd = [
        sys.executable, 'calibrate_act_scales.py',
        '--mode', mode,
        '-m', model_short,
    ]
    if mode == 'dart':
        cmd += ['--r1', '--r2']
    cmd += extra_args
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run baseline/quarot/dart calibrations in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_calibration.py -m 1b --sym --kv_ex 8 --proj_ex 15 -k 8 --v_bits 8 -g 2 3 4
  python multi_calibration.py -m 3b --sym -g 0 1 2
""")
    parser.add_argument('-m', '--model', required=True,
                        help='Model shorthand (1b, 3b, 7b) or full path')
    parser.add_argument('-g', '--gpus', type=int, nargs='+', required=True,
                        help='GPU IDs to use (at least 1; up to 3 for full parallelism)')
    # All remaining args are forwarded to calibrate_act_scales.py
    return parser.parse_known_args()


def main():
    args, extra_args = parse_args()
    gpus = args.gpus
    model_short = args.model

    # Resolve model name for R1/R2 existence check
    model_full = MODEL_MAP.get(model_short, model_short)
    model_name = os.path.basename(model_full.rstrip('/'))

    modes = ['dart', 'quarot', 'baseline']

    # --- Phase 1: Train R1/R2 if needed (blocks dart GPU) ---
    if not r1r2_exist(model_name):
        print(f"R1/R2 not found for {model_name}. Training on GPU {gpus[0]} first...")
        ret = run_job(
            ['bash', 'calibrate_model.sh', '-m', model_full, '-g', str(gpus[0])],
            make_env(gpus[0]),
            'R1/R2 train',
        )
        if ret != 0:
            print("R1/R2 training failed. Aborting.")
            sys.exit(1)
    else:
        print(f"R1/R2 already exist for {model_name}. Skipping training.")

    # --- Phase 2: Run all three calibrations in parallel ---
    # Assign GPUs round-robin: mode[i] -> gpus[i % len(gpus)]
    threads = []
    results = {}

    def worker(mode, gpu_id):
        cmd = build_calibrate_cmd(mode, model_short, extra_args)
        results[mode] = run_job(cmd, make_env(gpu_id), mode)

    for i, mode in enumerate(modes):
        gpu = gpus[i % len(gpus)]
        t = threading.Thread(target=worker, args=(mode, gpu), daemon=True)
        threads.append((mode, t))

    # If we have fewer GPUs than modes, stagger launches so jobs sharing
    # a GPU don't compete.  With >= 3 GPUs, all start immediately.
    if len(gpus) >= len(modes):
        # All parallel
        for _, t in threads:
            t.start()
        for _, t in threads:
            t.join()
    else:
        # Launch in waves grouped by GPU
        from collections import defaultdict
        waves = defaultdict(list)
        for i, (mode, t) in enumerate(threads):
            waves[i % len(gpus)].append((mode, t))

        # Start first wave (one job per GPU)
        active = []
        pending = []
        for gpu_slot, jobs in waves.items():
            mode, t = jobs[0]
            t.start()
            active.append((gpu_slot, mode, t))
            pending.extend(jobs[1:])

        # As jobs finish, launch the next pending job for that GPU slot
        while active or pending:
            still_active = []
            for gpu_slot, mode, t in active:
                t.join(timeout=1.0)
                if t.is_alive():
                    still_active.append((gpu_slot, mode, t))
                else:
                    # Slot freed, launch next pending job for any slot
                    if pending:
                        next_mode, next_t = pending.pop(0)
                        next_t.start()
                        still_active.append((gpu_slot, next_mode, next_t))
            active = still_active
            if active:
                time.sleep(0.5)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Calibration summary:")
    print(f"{'='*60}")
    for mode in modes:
        status = "OK" if results.get(mode) == 0 else f"FAILED (exit {results.get(mode, '?')})"
        print(f"  {mode:10s} : {status}")


if __name__ == '__main__':
    main()
