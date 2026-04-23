"""GPU wait utility — poll nvidia-smi until a GPU with low used memory is available.

Usage (shell):
    GPU_ID=$(python utils/gpu_wait.py)                    # wait for clear GPU, print ID
    GPU_ID=$(python utils/gpu_wait.py --max_used_mb 500)  # custom threshold
    python utils/gpu_wait.py --count                      # print count of clear GPUs

Usage (Python):
    from gpu_wait import wait_for_gpu, count_available_gpus
    gpu_id = wait_for_gpu(max_used_mb=200)
    n = count_available_gpus(max_used_mb=200)
"""

import subprocess
import sys
import time


def get_gpu_memory():
    """Query nvidia-smi for per-GPU memory stats.

    Returns:
        dict: {gpu_index: {'used': int, 'free': int, 'total': int}} in MiB.
    """
    out = subprocess.check_output(
        ['nvidia-smi',
         '--query-gpu=index,memory.used,memory.free,memory.total',
         '--format=csv,noheader,nounits'],
        text=True)
    result = {}
    for line in out.strip().split('\n'):
        parts = line.split(',')
        idx = int(parts[0].strip())
        result[idx] = {
            'used': int(parts[1].strip()),
            'free': int(parts[2].strip()),
            'total': int(parts[3].strip()),
        }
    return result


def count_available_gpus(max_used_mb=200):
    """Count GPUs with used memory at or below *max_used_mb* MiB."""
    mem = get_gpu_memory()
    return sum(1 for info in mem.values() if info['used'] <= max_used_mb)


def wait_for_gpu(max_used_mb=200, poll_interval=20):
    """Block until a GPU with used memory <= *max_used_mb* MiB is found.

    Among qualifying GPUs, selects the one with the least used memory.
    Status messages go to stderr so stdout stays clean for shell capture.
    While waiting, refreshes a single status line in place (carriage return)
    instead of spamming a new line per poll.

    Returns:
        int: GPU index.
    """
    start = time.time()
    waited = False
    while True:
        mem = get_gpu_memory()
        candidates = {idx: info for idx, info in mem.items()
                      if info['used'] <= max_used_mb}
        if candidates:
            best = min(candidates, key=lambda i: candidates[i]['used'])
            if waited:
                sys.stderr.write('\n')  # finish the in-place line
            print(f"GPU {best} available ({candidates[best]['used']} MiB used, "
                  f"{candidates[best]['free']} MiB free)", file=sys.stderr)
            return best
        best_used = min(info['used'] for info in mem.values())
        elapsed = int(time.time() - start)
        sys.stderr.write(
            f"\rwaiting {elapsed}s for GPU (lowest used: {best_used} MiB, "
            f"need <= {max_used_mb} MiB) ...\033[K")
        sys.stderr.flush()
        waited = True
        time.sleep(poll_interval)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Wait for a clear GPU')
    p.add_argument('--max_used_mb', type=int, default=200,
                   help='Max used memory (MiB) for a GPU to be considered clear (default: 200)')
    p.add_argument('--poll_interval', type=int, default=20,
                   help='Seconds between polls (default: 20)')
    p.add_argument('--count', action='store_true',
                   help='Print count of available GPUs and exit (no waiting)')
    args = p.parse_args()

    if args.count:
        print(count_available_gpus(args.max_used_mb))
    else:
        gpu_id = wait_for_gpu(args.max_used_mb, args.poll_interval)
        print(gpu_id)
