"""Shared parallel-runner UI used by run_experiments.py and multi_calibration.py.

Each job's stdout/stderr is redirected to its own log file. A status block at
the bottom of the terminal refreshes every `refresh` seconds, showing one line
per job with the tail of its log (`[running]` / `[done]` / `[FAIL]`).

Refresh is done with ANSI cursor moves (`\\033[{n}A` to move up, `\\033[K` to
clear to end of line) — no curses, no rich, no extra deps.
"""

import os
import re
import shutil
import subprocess
import sys
import time


def read_last_line(path):
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


def default_fail_line(name, rc, log_path):
    return f"  [FAIL]    {name}: see log at {os.path.relpath(log_path)} (exit {rc})"


def calibration_fail_line(name, rc, log_path):
    return (f"  [FAIL]    {name}: calibration failed: "
            f"see backtrace in {os.path.relpath(log_path)} (exit {rc})")


def _truncate_to_width(line, width):
    """Truncate a status line to fit `width` columns (preserves the tail)."""
    if len(line) <= width:
        return line
    if width <= 3:
        return line[:width]
    return "..." + line[-(width - 3):]


def print_status(procs, first_call=False, fail_message_fn=None):
    """Print or refresh the status block for all in-flight processes.

    `procs` is a list of (name, Popen, log_path).
    `fail_message_fn(name, rc, log_path) -> str` overrides the FAIL line.
    """
    if fail_message_fn is None:
        fail_message_fn = default_fail_line

    term_width = shutil.get_terminal_size((120, 24)).columns
    n = len(procs)
    if not first_call:
        # Move cursor up to overwrite previous status block
        sys.stdout.write(f"\033[{n}A")

    for name, proc, log_path in procs:
        rc = proc.poll()
        if rc is None:
            tail = read_last_line(log_path)
            if tail:
                prefix = f"  [running] {name}: "
                max_tail = term_width - len(prefix)
                if max_tail > 3 and len(tail) > max_tail:
                    tail = "..." + tail[-(max_tail - 3):]
                line = prefix + tail
            else:
                line = f"  [running] {name}: starting..."
        elif rc == 0:
            tail = read_last_line(log_path)
            if tail:
                prefix = f"  [done]    {name}: "
                max_tail = term_width - len(prefix)
                if max_tail > 3 and len(tail) > max_tail:
                    tail = "..." + tail[-(max_tail - 3):]
                line = prefix + tail
            else:
                line = f"  [done]    {name}"
        else:
            line = fail_message_fn(name, rc, log_path)
            line = _truncate_to_width(line, term_width)
        # Clear rest of line in case previous line was longer
        sys.stdout.write(f"\033[K{line}\n")
    sys.stdout.flush()


def run_parallel_batches(
    active_cmds,
    gpus,
    refresh,
    log_fn,
    on_complete=None,
    cwd=None,
    fail_message_fn=None,
):
    """Run a list of jobs in batches of len(gpus) with a refreshing status block.

    Args:
        active_cmds: list of (name, cmd) or (name, cmd, env). When env is
            provided, it is passed to subprocess.Popen.
        gpus: list of GPU IDs (only len(gpus) is used here — determines
            batch size).
        refresh: status refresh interval in seconds.
        log_fn: callable(name) -> log file path.
        on_complete: optional callable(name, returncode) called once per job
            as soon as it finishes.
        cwd: optional working directory passed to every Popen.
        fail_message_fn: optional callable(name, rc, log_path) -> str to
            customize the FAIL status line.

    Returns the total number of failed jobs.
    """
    n_gpus = len(gpus)
    n_batches = (len(active_cmds) + n_gpus - 1) // n_gpus
    total_failed = 0

    for batch_idx in range(n_batches):
        batch = active_cmds[batch_idx * n_gpus : (batch_idx + 1) * n_gpus]

        if n_batches > 1:
            print(f"\n=== Batch {batch_idx + 1}/{n_batches}: "
                  f"{len(batch)} runs (refreshing every {refresh}s) ===\n")
        else:
            print(f"\n=== Waiting for {len(batch)} runs "
                  f"(refreshing every {refresh}s) ===\n")

        procs = []
        log_files = []
        for entry in batch:
            if len(entry) == 2:
                name, cmd = entry
                env = None
            else:
                name, cmd, env = entry
            log_path = log_fn(name)
            log_f = open(log_path, 'w')
            popen_kwargs = {'stdout': log_f, 'stderr': subprocess.STDOUT}
            if env is not None:
                popen_kwargs['env'] = env
            if cwd is not None:
                popen_kwargs['cwd'] = cwd
            proc = subprocess.Popen(cmd, **popen_kwargs)
            procs.append((name, proc, log_path))
            log_files.append(log_f)

        print_status(procs, first_call=True, fail_message_fn=fail_message_fn)

        completed = set()

        while any(proc.poll() is None for _, proc, _ in procs):
            time.sleep(refresh)
            print_status(procs, fail_message_fn=fail_message_fn)
            if on_complete:
                for name, proc, _ in procs:
                    if name not in completed and proc.poll() is not None:
                        on_complete(name, proc.returncode)
                        completed.add(name)

        print_status(procs, fail_message_fn=fail_message_fn)

        # Notify any remaining
        if on_complete:
            for name, proc, _ in procs:
                if name not in completed:
                    on_complete(name, proc.returncode)

        for lf in log_files:
            lf.close()

        total_failed += sum(1 for _, proc, _ in procs if proc.returncode != 0)

    return total_failed
