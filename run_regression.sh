#!/bin/bash
# run_regression.sh - Run all correctness test suites.
#
# Usage:
#   sudo ./run_regression.sh              # build + run all suites
#   sudo ./run_regression.sh --no-build   # skip make, just run binaries
#   sudo ./run_regression.sh --verbose    # print output for every test, not just failures

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/correctness_tests"

NO_BUILD=0
VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --no-build) NO_BUILD=1 ;;
        --verbose|-v) VERBOSE=1 ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--no-build] [--verbose]"
            exit 1
            ;;
    esac
done

ANSI_GREEN="\033[32m"
ANSI_RED="\033[31m"
ANSI_BOLD="\033[1m"
ANSI_RESET="\033[0m"

color() { printf "%b%s\n%b" "$2" "$1" "$ANSI_RESET"; }

TOTAL_SUITES=0
FAILED_SUITES=0
FAILED_SUITE_NAMES=()

run_suite_with_runner() {
    local suite_dir="$1"
    local suite_name
    suite_name="$(basename "$suite_dir")"
    TOTAL_SUITES=$((TOTAL_SUITES + 1))

    echo ""
    color "Running suite: $suite_name" "$ANSI_BOLD"

    local args=()
    [ "$NO_BUILD" -eq 1 ] && args+=(--no-build)
    [ "$VERBOSE" -eq 1 ] && args+=(--verbose)

    python3 "$suite_dir/run_tests.py" "${args[@]}"
    local rc=$?

    if [ "$rc" -ne 0 ]; then
        FAILED_SUITES=$((FAILED_SUITES + 1))
        FAILED_SUITE_NAMES+=("$suite_name")
    fi
}

run_generic_suite() {
    local suite_dir="$TESTS_DIR/generic"
    local suite_name="generic"
    TOTAL_SUITES=$((TOTAL_SUITES + 1))

    echo ""
    color "Running suite: $suite_name" "$ANSI_BOLD"

    if [ "$NO_BUILD" -eq 0 ]; then
        echo "[runner] building: make -C $suite_dir"
        make -C "$suite_dir"
        if [ $? -ne 0 ]; then
            color "[runner] build failed for $suite_name\n" "$ANSI_RED"
            FAILED_SUITES=$((FAILED_SUITES + 1))
            FAILED_SUITE_NAMES+=("$suite_name")
            return
        fi
        echo ""
    fi

    local binary="$suite_dir/test_dirty_tracking_suite"
    if [ ! -x "$binary" ]; then
        color "[runner] binary not found: $binary\n" "$ANSI_RED"
        FAILED_SUITES=$((FAILED_SUITES + 1))
        FAILED_SUITE_NAMES+=("$suite_name (binary missing)")
        return
    fi

    local cmd=("$binary")
    [ "$(id -u)" -ne 0 ] && cmd=(sudo "${cmd[@]}")

    printf "  running test_dirty_tracking_suite ..."
    local output rc elapsed
    local t0=$SECONDS
    output=$("${cmd[@]}" 2>&1)
    rc=$?
    elapsed=$((SECONDS - t0))

    if [ "$rc" -eq 0 ]; then
        printf "\r  $(color PASS "$ANSI_GREEN")  %-40s  (%ds)\n" "test_dirty_tracking_suite" "$elapsed"
    else
        printf "\r  $(color FAIL "$ANSI_RED")  %-40s  (%ds)\n" "test_dirty_tracking_suite" "$elapsed"
        FAILED_SUITES=$((FAILED_SUITES + 1))
        FAILED_SUITE_NAMES+=("$suite_name")
    fi

    if [ "$VERBOSE" -eq 1 ] || [ "$rc" -ne 0 ]; then
        while IFS= read -r line; do
            echo "         $line"
        done <<< "$output"
    fi
}

# -- Run all suites ----------------------------------------------------------

printf "\n%b%s%b\n" "$ANSI_BOLD" "$(printf '=%.0s' {1..60})" "$ANSI_RESET"
printf "%b  Dirty Tracking - Regression Test Suite%b\n" "$ANSI_BOLD" "$ANSI_RESET"
printf "%b%s%b\n" "$ANSI_BOLD" "$(printf '=%.0s' {1..60})" "$ANSI_RESET"

# Suites with a run_tests.py runner are discovered automatically.
# Suites without one (e.g. generic) are handled by run_generic_suite.
while IFS= read -r -d '' runner; do
    run_suite_with_runner "$(dirname "$runner")"
done < <(find "$TESTS_DIR" -maxdepth 2 -name run_tests.py -print0 | sort -z)

# generic suite has no run_tests.py - run its binary directly.
run_generic_suite

# -- Summary -----------------------------------------------------------------

PASSED_SUITES=$((TOTAL_SUITES - FAILED_SUITES))
echo ""
printf "%b%s%b\n" "$ANSI_BOLD" "$(printf '=%.0s' {1..60})" "$ANSI_RESET"

if [ "$FAILED_SUITES" -eq 0 ]; then
    printf "%b  All %d suite(s) passed.%b\n" "${ANSI_GREEN}${ANSI_BOLD}" "$TOTAL_SUITES" "$ANSI_RESET"
else
    printf "%b  %d/%d suite(s) passed, %d failed.%b\n" \
        "${ANSI_RED}${ANSI_BOLD}" "$PASSED_SUITES" "$TOTAL_SUITES" "$FAILED_SUITES" "$ANSI_RESET"
    echo ""
    printf "%bFailed suites:%b\n" "${ANSI_RED}${ANSI_BOLD}" "$ANSI_RESET"
    for name in "${FAILED_SUITE_NAMES[@]}"; do
        echo "    $name"
    done
fi

printf "%b%s%b\n" "$ANSI_BOLD" "$(printf '=%.0s' {1..60})" "$ANSI_RESET"

# -- Clean up binaries --------------------------------------------------------

echo ""
echo "[runner] cleaning test binaries..."
while IFS= read -r -d '' makefile; do
    make -C "$(dirname "$makefile")" clean -s 2>/dev/null || true
done < <(find "$TESTS_DIR" -maxdepth 2 -name Makefile -print0)

[ "$FAILED_SUITES" -eq 0 ]
