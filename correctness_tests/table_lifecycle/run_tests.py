#!/usr/bin/env python3
"""
run_tests.py — table_lifecycle test runner

Builds all tc*.cu files then runs each binary (as root via sudo if needed)
and reports PASS / FAIL per test.

Usage:
    sudo python3 run_tests.py          # build + run all
    sudo python3 run_tests.py --no-build   # skip build, just run
    sudo python3 run_tests.py tc01 tc03    # run specific tests by prefix
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ANSI_GREEN  = "\033[32m"
ANSI_RED    = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"

def color(text, code):
    if sys.stdout.isatty():
        return f"{code}{text}{ANSI_RESET}"
    return text


def build(targets=None):
    """Run make in SCRIPT_DIR, optionally for specific targets."""
    cmd = ["make", "-C", SCRIPT_DIR]
    if targets:
        cmd += targets
    print(f"[runner] building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(color("[runner] build failed", ANSI_RED))
        sys.exit(1)
    print()


def find_binaries(prefixes=None):
    """Return sorted list of test binary paths matching optional prefixes."""
    bins = []
    for entry in sorted(os.listdir(SCRIPT_DIR)):
        if not entry.startswith("tc"):
            continue
        path = os.path.join(SCRIPT_DIR, entry)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            continue
        # skip source files
        if entry.endswith(".cu"):
            continue
        if prefixes:
            if not any(entry.startswith(p) for p in prefixes):
                continue
        bins.append(path)
    return bins


def run_one(binary):
    """Run a single test binary, return (returncode, stdout, stderr, elapsed)."""
    cmd = [binary]
    # if not root, wrap with sudo
    if os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    return result.returncode, result.stdout, result.stderr, elapsed


def main():
    parser = argparse.ArgumentParser(description="table_lifecycle test runner")
    parser.add_argument("tests", nargs="*", help="test prefixes to run (e.g. tc01 tc03)")
    parser.add_argument("--no-build", action="store_true", help="skip make step")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="always print test output even on PASS")
    args = parser.parse_args()

    if not args.no_build:
        build()

    binaries = find_binaries(args.tests if args.tests else None)
    if not binaries:
        print(color("[runner] no test binaries found", ANSI_YELLOW))
        sys.exit(0)

    print(color(f"{'='*60}", ANSI_BOLD))
    print(color(f"  table_lifecycle — {len(binaries)} test(s)", ANSI_BOLD))
    print(color(f"{'='*60}", ANSI_BOLD))
    print()

    passed = 0
    failed = 0
    results = []

    for binary in binaries:
        name = os.path.basename(binary)
        print(f"  running {name} ...", end="", flush=True)
        rc, stdout, stderr, elapsed = run_one(binary)

        if rc == 0:
            status = color("PASS", ANSI_GREEN)
            passed += 1
        else:
            status = color("FAIL", ANSI_RED)
            failed += 1

        print(f"\r  {status}  {name:<40}  ({elapsed:.2f}s)")

        show_output = args.verbose or rc != 0
        if show_output:
            # indent each output line for readability
            for line in stdout.splitlines():
                print(f"         {line}")
            for line in stderr.splitlines():
                print(f"         {color(line, ANSI_RED)}")

        results.append((name, rc, stdout, stderr, elapsed))

    total = passed + failed
    print()
    print(color(f"{'='*60}", ANSI_BOLD))
    summary = f"  Results: {passed}/{total} passed, {failed} failed"
    summary_color = ANSI_GREEN if failed == 0 else ANSI_RED
    print(color(summary, summary_color + ANSI_BOLD))
    print(color(f"{'='*60}", ANSI_BOLD))

    if failed > 0:
        print()
        print(color("  Failed tests:", ANSI_RED + ANSI_BOLD))
        for name, rc, stdout, stderr, _ in results:
            if rc != 0:
                print(f"    {name}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
