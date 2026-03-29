#!/usr/bin/env python3
"""
run_all.py - operation_microbenchmarks runner

Builds all benchmarks then runs each one, rotating existing CSVs to .old
before each run to preserve the previous results.

Usage:
    python3 run_all.py                        # build + run all
    python3 run_all.py --no-build             # skip make, just run
    python3 run_all.py single_fault table     # run specific benchmarks by prefix
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


# (binary, [csv outputs it produces])
BENCHMARKS = [
    ("single_fault_record_cost",  ["single_fault_record_cost.csv"]),
    ("duplicate_fault_skip_cost", ["duplicate_fault_skip_cost.csv"]),
    ("record_contention_cost",    ["record_contention_cost.csv"]),
    ("table_lifecycle_cost",      ["table_init_cost.csv", "table_destroy_cost.csv", "table_reinit_cost.csv"]),
    ("basic_rmw",                 []),
]


def build():
    cmd = ["make", "-C", SCRIPT_DIR]
    print(f"[runner] building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(color("[runner] build failed", ANSI_RED))
        sys.exit(1)
    print()


def rotate_csv(csv_path):
    old_path = csv_path + ".old"
    if os.path.exists(csv_path):
        if os.path.exists(old_path):
            os.remove(old_path)
        os.rename(csv_path, old_path)
        print(f"         rotated {os.path.basename(csv_path)} -> {os.path.basename(old_path)}")


def run_one(binary, csvs, verbose):
    binary_path = os.path.join(SCRIPT_DIR, binary)
    if not os.path.isfile(binary_path) or not os.access(binary_path, os.X_OK):
        return None, "", "", 0.0  # signal skip

    for csv in csvs:
        rotate_csv(os.path.join(SCRIPT_DIR, csv))

    cmd = [binary_path]
    if os.geteuid() != 0:
        cmd = ["sudo"] + cmd

    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    return result.returncode, result.stdout, result.stderr, elapsed


def main():
    parser = argparse.ArgumentParser(description="operation_microbenchmarks runner")
    parser.add_argument("benchmarks", nargs="*",
                        help="benchmark name prefixes to run (e.g. single_fault table)")
    parser.add_argument("--no-build", action="store_true", help="skip make step")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="always print benchmark output even on PASS")
    args = parser.parse_args()

    if not args.no_build:
        build()

    selected = [
        (binary, csvs) for binary, csvs in BENCHMARKS
        if not args.benchmarks or any(binary.startswith(p) for p in args.benchmarks)
    ]

    if not selected:
        print(color("[runner] no matching benchmarks found", ANSI_YELLOW))
        sys.exit(0)

    print(color(f"{'='*60}", ANSI_BOLD))
    print(color(f"  operation_microbenchmarks - {len(selected)} benchmark(s)", ANSI_BOLD))
    print(color(f"{'='*60}", ANSI_BOLD))
    print()

    passed = 0
    failed = 0
    skipped = 0
    results = []

    for binary, csvs in selected:
        print(f"  running {binary} ...", end="", flush=True)
        rc, stdout, stderr, elapsed = run_one(binary, csvs, args.verbose)

        if rc is None:
            print(f"\r  {color('SKIP', ANSI_YELLOW)}  {binary:<40}  (binary not found)")
            skipped += 1
            continue

        if rc == 0:
            status = color("PASS", ANSI_GREEN)
            passed += 1
        else:
            status = color("FAIL", ANSI_RED)
            failed += 1

        print(f"\r  {status}  {binary:<40}  ({elapsed:.2f}s)")

        if args.verbose or rc != 0:
            for line in stdout.splitlines():
                print(f"         {line}")
            for line in stderr.splitlines():
                print(f"         {color(line, ANSI_RED)}")

        results.append((binary, rc, stdout, stderr, elapsed))

    total = passed + failed
    print()
    print(color(f"{'='*60}", ANSI_BOLD))
    summary = f"  Results: {passed}/{total} passed, {failed} failed, {skipped} skipped"
    summary_color = ANSI_GREEN if failed == 0 else ANSI_RED
    print(color(summary, summary_color + ANSI_BOLD))
    print(color(f"{'='*60}", ANSI_BOLD))

    if failed > 0:
        print()
        print(color("  Failed benchmarks:", ANSI_RED + ANSI_BOLD))
        for name, rc, _, _, _ in results:
            if rc != 0:
                print(f"    {name}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
