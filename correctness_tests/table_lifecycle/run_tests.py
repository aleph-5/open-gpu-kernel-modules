#!/usr/bin/env python3
"""
run_tests.py  —  per-suite test runner

Builds all test binaries via make, then runs each one (as root / via sudo)
and reports PASS / FAIL.

Usage:
    sudo python3 run_tests.py              # build + run all
    sudo python3 run_tests.py --no-build   # skip make, run existing binaries
    sudo python3 run_tests.py tc01 tc03    # run specific tests by prefix
    sudo python3 run_tests.py -v           # verbose: always print output
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_NAME = os.path.basename(SCRIPT_DIR)

ANSI_GREEN  = "\033[32m"
ANSI_RED    = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_CYAN   = "\033[36m"

_IS_TTY = sys.stdout.isatty()


def color(text, code):
    return f"{code}{text}{ANSI_RESET}" if _IS_TTY else text


def build():
    cmd = ["make", "-C", SCRIPT_DIR]
    print(f"[{SUITE_NAME}] building...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(color(f"[{SUITE_NAME}] build failed", ANSI_RED))
        sys.exit(1)
    print()


_SKIP_EXTENSIONS = {".cu", ".py", ".sh", ".md", ".txt", ".o"}
_SKIP_NAMES      = {"Makefile", "GNUmakefile", "makefile"}


def find_binaries(prefixes=None):
    bins = []
    for entry in sorted(os.listdir(SCRIPT_DIR)):
        path = os.path.join(SCRIPT_DIR, entry)
        if not os.path.isfile(path):
            continue
        if not os.access(path, os.X_OK):
            continue
        _, ext = os.path.splitext(entry)
        if ext in _SKIP_EXTENSIONS:
            continue
        if entry in _SKIP_NAMES:
            continue
        if prefixes and not any(entry.startswith(p) for p in prefixes):
            continue
        bins.append(path)
    return bins


def run_one(binary, timeout=300):
    cmd = [binary]
    if os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - t0
        return result.returncode, result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return -1, "", f"TIMEOUT after {timeout}s", elapsed


def main():
    parser = argparse.ArgumentParser(
        description=f"{SUITE_NAME} test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tests", nargs="*",
        help="test name prefixes to run (e.g. tc01 tc03); default: all",
    )
    parser.add_argument("--no-build", action="store_true", help="skip make step")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="always print test output even on PASS")
    parser.add_argument("--timeout", type=int, default=300,
                        help="per-test timeout in seconds (default: 300)")
    args = parser.parse_args()

    if not args.no_build:
        build()

    binaries = find_binaries(args.tests or None)
    if not binaries:
        print(color(f"[{SUITE_NAME}] no test binaries found", ANSI_YELLOW))
        sys.exit(0)

    bar = "=" * 62
    print(color(bar, ANSI_BOLD))
    print(color(f"  {SUITE_NAME}  —  {len(binaries)} test(s)", ANSI_BOLD))
    print(color(bar, ANSI_BOLD))
    print()

    passed, failed = 0, 0
    results = []

    for binary in binaries:
        name = os.path.basename(binary)
        print(f"  running {name} ...", end="", flush=True)
        rc, stdout, stderr, elapsed = run_one(binary, timeout=args.timeout)

        if rc == 0:
            status = color("PASS", ANSI_GREEN)
            passed += 1
        elif rc == -1:
            status = color("TIMEOUT", ANSI_YELLOW)
            failed += 1
        else:
            status = color("FAIL", ANSI_RED)
            failed += 1

        print(f"\r  {status}  {name:<44}  ({elapsed:.2f}s)")

        if args.verbose or rc != 0:
            for line in stdout.splitlines():
                print(f"         {line}")
            for line in stderr.splitlines():
                print(f"         {color(line, ANSI_RED)}")

        results.append((name, rc, stdout, stderr, elapsed))

    total = passed + failed
    print()
    print(color(bar, ANSI_BOLD))
    summary_color = ANSI_GREEN if failed == 0 else ANSI_RED
    print(color(f"  Results: {passed}/{total} passed, {failed} failed", summary_color + ANSI_BOLD))
    print(color(bar, ANSI_BOLD))

    if failed > 0:
        print()
        print(color("  Failed tests:", ANSI_RED + ANSI_BOLD))
        for name, rc, _, _, _ in results:
            if rc != 0:
                marker = "TIMEOUT" if rc == -1 else "FAIL"
                print(f"    [{marker}]  {name}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
