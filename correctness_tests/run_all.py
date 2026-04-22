#!/usr/bin/env python3
"""
run_all.py  —  top-level runner for all dirty-tracking correctness suites

Discovers every subdirectory that contains a run_tests.py and executes it in
order, collecting per-suite pass/fail counts and printing a final summary.

Usage:
    sudo python3 run_all.py                        # build + run everything
    sudo python3 run_all.py --no-build             # skip make in every suite
    sudo python3 run_all.py --suites ordering scale # run specific suites only
    sudo python3 run_all.py -v                     # verbose (show all output)
    sudo python3 run_all.py --list                 # list discovered suites
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
ANSI_CYAN   = "\033[36m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"

_IS_TTY = sys.stdout.isatty()


def color(text, code):
    return f"{code}{text}{ANSI_RESET}" if _IS_TTY else text


# Canonical suite order: roughly from foundational to advanced.
SUITE_ORDER = [
    "table_lifecycle",
    "ordering",
    "determinism",
    "first_write_wins",
    "cumulative_mode",
    "address_range_filtering",
    "access_type_filtering",
    "snapshot_isolation",
    "backend_correctness",
    "procfs_robustness",
    "lifecycle_races",
    "scale",
    "stats_accuracy",
    "footprint_clearing",
    "generic",
]


def discover_suites():
    """Return ordered list of (suite_name, run_tests_path) for all suites found."""
    found = {}
    for entry in os.listdir(SCRIPT_DIR):
        suite_dir = os.path.join(SCRIPT_DIR, entry)
        runner    = os.path.join(suite_dir, "run_tests.py")
        if os.path.isdir(suite_dir) and os.path.isfile(runner):
            found[entry] = runner

    # Emit in canonical order first, then any unlisted suites alphabetically.
    ordered = []
    for name in SUITE_ORDER:
        if name in found:
            ordered.append((name, found[name]))
    for name in sorted(found):
        if name not in SUITE_ORDER:
            ordered.append((name, found[name]))
    return ordered


def run_suite(suite_name, runner_path, no_build=False, clean=False, verbose=False, timeout=600):
    """
    Invoke run_tests.py for a single suite as a subprocess.
    Returns (passed, failed, elapsed).
    """
    def build_cmd(exe):
        cmd = [exe, runner_path, "--timeout", "300"]
        if no_build:
            cmd.append("--no-build")
        if not clean:
            cmd.append("--no-clean")
        if verbose:
            cmd.append("--verbose")
        return cmd

    if os.geteuid() != 0:
        cmd = ["sudo"] + build_cmd(sys.executable)
    else:
        cmd = build_cmd(sys.executable)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,   # stream output live
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        return result.returncode, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return -1, elapsed


def parse_suite_result(returncode):
    """Map run_tests.py exit code to a human label."""
    if returncode == 0:
        return "PASS"
    if returncode == -1:
        return "TIMEOUT"
    return "FAIL"


def main():
    parser = argparse.ArgumentParser(
        description="Run all dirty-tracking correctness test suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suites", nargs="+", metavar="SUITE",
        help="run only these named suites (e.g. --suites ordering scale)",
    )
    parser.add_argument("--no-build", action="store_true",
                        help="pass --no-build to every suite runner")
    parser.add_argument("-c", "--clean", action="store_true",
                        help="run make clean after each suite (default: keep build artifacts)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="pass --verbose to every suite runner")
    parser.add_argument("--list", action="store_true",
                        help="list discovered suites and exit")
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-suite timeout in seconds (default: 600)")
    args = parser.parse_args()

    all_suites = discover_suites()

    if args.list:
        print("Discovered suites:")
        for name, path in all_suites:
            print(f"  {name:<35}  {path}")
        sys.exit(0)

    if args.suites:
        requested = set(args.suites)
        suites = [(n, p) for n, p in all_suites if n in requested]
        missing = requested - {n for n, _ in suites}
        if missing:
            print(color(f"[run_all] unknown suite(s): {', '.join(sorted(missing))}", ANSI_RED))
            sys.exit(1)
    else:
        suites = all_suites

    if not suites:
        print(color("[run_all] no suites found", ANSI_YELLOW))
        sys.exit(0)

    wide = "=" * 70
    print()
    print(color(wide, ANSI_BOLD))
    print(color(f"  dirty-tracking correctness suite  —  {len(suites)} suite(s)", ANSI_BOLD))
    print(color(wide, ANSI_BOLD))
    print()

    suite_results = []   # (name, returncode, elapsed)

    for suite_name, runner_path in suites:
        header = f"  ── {suite_name} "
        header += "─" * max(0, 66 - len(header))
        print(color(header, ANSI_CYAN + ANSI_BOLD))
        print()

        rc, elapsed = run_suite(
            suite_name, runner_path,
            no_build=args.no_build,
            clean=args.clean,
            verbose=args.verbose,
            timeout=args.timeout,
        )
        suite_results.append((suite_name, rc, elapsed))
        print()

    # ── Final summary ──────────────────────────────────────────────────────
    passed  = sum(1 for _, rc, _ in suite_results if rc == 0)
    failed  = sum(1 for _, rc, _ in suite_results if rc != 0)
    total_t = sum(e for _, _, e in suite_results)

    print(color(wide, ANSI_BOLD))
    print(color("  SUITE SUMMARY", ANSI_BOLD))
    print(color(wide, ANSI_BOLD))
    print()

    name_width = max(len(n) for n, _, _ in suite_results)
    for suite_name, rc, elapsed in suite_results:
        label = parse_suite_result(rc)
        if label == "PASS":
            status = color(f"  PASS", ANSI_GREEN)
        elif label == "TIMEOUT":
            status = color(f"  TIMEOUT", ANSI_YELLOW)
        else:
            status = color(f"  FAIL", ANSI_RED)
        print(f"{status}  {suite_name:<{name_width}}  ({elapsed:.1f}s)")

    print()
    print(color(wide, ANSI_BOLD))
    overall_color = ANSI_GREEN if failed == 0 else ANSI_RED
    print(color(
        f"  Overall: {passed}/{len(suite_results)} suites passed, "
        f"{failed} failed  ({total_t:.1f}s total)",
        overall_color + ANSI_BOLD,
    ))
    print(color(wide, ANSI_BOLD))

    if failed > 0:
        print()
        print(color("  Failed suites:", ANSI_RED + ANSI_BOLD))
        for name, rc, _ in suite_results:
            if rc != 0:
                marker = "TIMEOUT" if rc == -1 else "FAIL"
                print(f"    [{marker}]  {name}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
