"""Shared runfile flag parsing/writing for run_experiments and multi_calibration.

Runfile format:
    name: MODE [args for dart_gptq_wxaykvz.sh, WITHOUT -g]
    [FLAG] name: MODE args...       # flagged entries
    # comment lines begin with # or ;

Flag vocabulary:
    [CAL]              calibration done, inference never ran
    [FAST]             fast inference done (implies calibration done)
    [DONE]             thorough inference done
    [ERROR]            inference failed (calibration succeeded) — retryable
    [ERROR-CALIBRATE]  calibration failed — needs re-calibration

Marking a line uses name-prefix matching so it's safe under concurrent
user edits (lines added/removed while runs are in progress).
"""

import threading


FLAG_DONE = 'DONE'
FLAG_FAST = 'FAST'
FLAG_CAL = 'CAL'
FLAG_ERROR = 'ERROR'
FLAG_ERROR_CAL = 'ERROR-CALIBRATE'

ALL_FLAGS = (FLAG_DONE, FLAG_FAST, FLAG_CAL, FLAG_ERROR, FLAG_ERROR_CAL)

# Flags that imply calibration has completed successfully at some point.
_CAL_DONE_FLAGS = (FLAG_CAL, FLAG_FAST, FLAG_DONE, FLAG_ERROR)

_mark_lock = threading.Lock()


def _strip_flag(stripped_line):
    """If stripped_line starts with a known [FLAG] token, return (flag, rest).
    Otherwise return (None, stripped_line)."""
    if not stripped_line.startswith('['):
        return None, stripped_line
    close = stripped_line.find(']')
    if close == -1:
        return None, stripped_line
    token = stripped_line[1:close]
    if token not in ALL_FLAGS:
        return None, stripped_line
    rest = stripped_line[close + 1:].lstrip()
    return token, rest


def parse_runfile(path):
    """Return list of (flag_or_None, name, cmd_args) for every run line.

    Skips blank lines and comments (# or ;). Emits a warning and skips
    lines that have no ':' separator.
    """
    entries = []
    with open(path) as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            flag, payload = _strip_flag(line)
            colon = payload.find(':')
            if colon == -1:
                print(f"Warning: skipping malformed line {i+1}: {line}")
                continue
            name = payload[:colon].strip()
            cmd_args = payload[colon+1:].strip()
            entries.append((flag, name, cmd_args))
    return entries


def mark_runfile(path, name, new_flag):
    """Rewrite the line whose run-name matches `name`.

    Strips any existing [FLAG] prefix and applies `new_flag`. Pass
    new_flag=None to clear the flag entirely.

    Thread-safe. Name-prefix matching makes this safe under concurrent
    user edits to the runfile.
    """
    with _mark_lock:
        with open(path) as f:
            lines = f.readlines()

        target = name + ':'
        for i, line in enumerate(lines):
            had_newline = line.endswith('\n')
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith(';'):
                continue
            # Preserve the line's leading whitespace (shouldn't matter, but tidy).
            leading = line[:len(line) - len(line.lstrip())]
            _, payload = _strip_flag(stripped)
            if not payload.startswith(target):
                continue
            prefix = f'[{new_flag}] ' if new_flag else ''
            new_line = leading + prefix + payload
            if had_newline:
                new_line += '\n'
            lines[i] = new_line
            break

        with open(path, 'w') as f:
            f.writelines(lines)


def needs_calibration(flag, recalib=False):
    """True iff this line's calibration must run."""
    if recalib:
        return True
    if flag is None or flag == FLAG_ERROR_CAL:
        return True
    return False


def needs_inference(flag, fast=False, recalib=False):
    """True iff this line's inference must run under the current mode."""
    if recalib:
        return True
    if flag == FLAG_DONE:
        return False
    if flag == FLAG_FAST:
        return not fast
    if flag == FLAG_ERROR_CAL:
        return False
    # None, CAL, ERROR
    return True


def inference_completion_flag(fast):
    """FAST if fast else DONE."""
    return FLAG_FAST if fast else FLAG_DONE
