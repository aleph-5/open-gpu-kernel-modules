#!/usr/bin/env bash
#!/usr/bin/env bash
# run_overhead_benchmark.sh
#
# Measures dirty-tracking pipeline overhead on real workloads.
# Runs each benchmark REPS times with tracking OFF and ON, reports wall time
# and overhead % in a CSV.
#
# Must be run as root (procfs requires it).
# The patched nvidia-uvm module must be loaded.
#
# Usage: sudo bash run_overhead_benchmark.sh [REPS [benchmark ...]]
#   REPS       number of repetitions per benchmark (default: 5)
#   benchmark  names of specific benchmarks to run (default: all)
#
# Examples:
#   sudo bash run_overhead_benchmark.sh          # all benchmarks, 5 reps
#   sudo bash run_overhead_benchmark.sh 10       # all benchmarks, 10 reps
#   sudo bash run_overhead_benchmark.sh 5 gemm sgemm bfs   # 3 benchmarks, 5 reps

set -euo pipefail

REPS=${1:-5}
shift || true   # remaining positional args are benchmark names to run (empty = all)
FILTER=("$@")   # e.g. ("gemm" "sgemm" "bfs")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_CSV="${OUT_CSV:-$SCRIPT_DIR/overhead_results.csv}"
DUMP_DIR="/tmp/tracking_preload_dumps"

DT_START="/proc/driver/nvidia-uvm/dirty_tracking_start"
DT_STOP="/proc/driver/nvidia-uvm/dirty_tracking_stop"
DT_CUTOVER="/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
DT_DUMP="/proc/driver/nvidia-uvm/dirty_tracking_query_dump"


# Tunables. Override like:
#   SLEEP_INTERVALS="0.01 0.05 0.10 0.50 1.00" QUERY_COUNTS="1 5 10" sudo -E bash run_overhead_benchmark.sh
# SLEEP_INTERVAL is now the tracking window length: after dirty tracking starts,
# tracking is stopped after exactly this many seconds. Dirty-page queries are
# evenly spread inside this window.
read -ra SLEEP_INTERVALS <<< "${SLEEP_INTERVALS:-0.01 0.05 0.10 0.50 1.00}"
read -ra QUERY_COUNTS    <<< "${QUERY_COUNTS:-1 5 10}"
START_POLL_INTERVAL="${START_POLL_INTERVAL:-0.05}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run as root (sudo)" >&2
    exit 1
fi

if ! lsmod | grep -q nvidia_uvm; then
    echo "ERROR: nvidia_uvm module not loaded" >&2
    exit 1
fi

if [[ ! -e "$DT_START" ]]; then
    echo "ERROR: $DT_START not found — is the patched module loaded?" >&2
    exit 1
fi

mkdir -p "$DUMP_DIR"

echo "==> Building benchmarks ..."
make -C "$SCRIPT_DIR" -j"$(nproc)" 2>&1 | tail -5 || echo "  (some targets failed to build — missing binaries will be skipped)" >&2

# ---------------------------------------------------------------------------
# Benchmark table: NAME  WORKDIR  COMMAND (relative to WORKDIR)
# ---------------------------------------------------------------------------
# Format: "name|relative_dir|cmd"
# Add/remove rows to taste.  Use default sizes from each benchmark's README.

declare -a BENCHMARKS=(
    # --- int_set_uvm (pure write, 4K stride): uses -mb flag ---
    # fits in GPU (A40 ~45GB): 512MB, 4GB, 8GB, 16GB, 32GB, 40GB
    # oversubscribed:          48GB, 64GB
    "int_set_4k_512m|synthetic_benchmarks|./int_set_uvm.out -mb 512 -stride-4k"
    "int_set_4k_4g|synthetic_benchmarks|./int_set_uvm.out -mb 4096 -stride-4k"
    "int_set_4k_8g|synthetic_benchmarks|./int_set_uvm.out -mb 8192 -stride-4k"
    "int_set_4k_16g|synthetic_benchmarks|./int_set_uvm.out -mb 16384 -stride-4k"
    "int_set_4k_32g|synthetic_benchmarks|./int_set_uvm.out -mb 32768 -stride-4k"
    "int_set_4k_40g|synthetic_benchmarks|./int_set_uvm.out -mb 40960 -stride-4k"
    "int_set_4k_48g|synthetic_benchmarks|./int_set_uvm.out -mb 49152 -stride-4k"
    "int_set_4k_64g|synthetic_benchmarks|./int_set_uvm.out -mb 65536 -stride-4k"

    # --- int_set_uvm (sequential, no stride): same size sweep ---
    "int_set_seq_512m|synthetic_benchmarks|./int_set_uvm.out -mb 512"
    "int_set_seq_4g|synthetic_benchmarks|./int_set_uvm.out -mb 4096"
    "int_set_seq_8g|synthetic_benchmarks|./int_set_uvm.out -mb 8192"
    "int_set_seq_16g|synthetic_benchmarks|./int_set_uvm.out -mb 16384"
    "int_set_seq_32g|synthetic_benchmarks|./int_set_uvm.out -mb 32768"
    "int_set_seq_40g|synthetic_benchmarks|./int_set_uvm.out -mb 40960"
    "int_set_seq_48g|synthetic_benchmarks|./int_set_uvm.out -mb 49152"
    "int_set_seq_64g|synthetic_benchmarks|./int_set_uvm.out -mb 65536"

    # --- polybench GEMM: first arg is working-set size in bytes ---
    "gemm_512m|polybench/GEMM|./gemm.exe 536870912"
    "gemm_4g|polybench/GEMM|./gemm.exe 4294967296"
    "gemm_8g|polybench/GEMM|./gemm.exe 8589934592"
    "gemm_16g|polybench/GEMM|./gemm.exe 17179869184"
    "gemm_32g|polybench/GEMM|./gemm.exe 34359738368"
    "gemm_40g|polybench/GEMM|./gemm.exe 42949672960"
    "gemm_48g|polybench/GEMM|./gemm.exe 51539607552"
    "gemm_64g|polybench/GEMM|./gemm.exe 68719476736"

    # --- polybench 2MM ---
    "2mm_512m|polybench/2MM|./2mm.exe 536870912"
    "2mm_4g|polybench/2MM|./2mm.exe 4294967296"
    "2mm_8g|polybench/2MM|./2mm.exe 8589934592"
    "2mm_16g|polybench/2MM|./2mm.exe 17179869184"
    "2mm_32g|polybench/2MM|./2mm.exe 34359738368"
    "2mm_40g|polybench/2MM|./2mm.exe 42949672960"
    # "2mm_48g|polybench/2MM|./2mm.exe 51539607552"
    # "2mm_64g|polybench/2MM|./2mm.exe 68719476736"

    # --- polybench BICG (read-heavy; overhead should be low — good control) ---
    "bicg_512m|polybench/BICG|./bicg.exe 536870912"
    "bicg_4g|polybench/BICG|./bicg.exe 4294967296"
    "bicg_8g|polybench/BICG|./bicg.exe 8589934592"
    "bicg_16g|polybench/BICG|./bicg.exe 17179869184"
    "bicg_32g|polybench/BICG|./bicg.exe 34359738368"
    "bicg_40g|polybench/BICG|./bicg.exe 42949672960"
    "bicg_48g|polybench/BICG|./bicg.exe 51539607552"
    "bicg_64g|polybench/BICG|./bicg.exe 68719476736"

    # --- polybench MVT (with read-mostly hint) ---
    "mvt_512m|polybench/MVT|./mvt.exe 536870912"
    "mvt_4g|polybench/MVT|./mvt.exe 4294967296"
    "mvt_8g|polybench/MVT|./mvt.exe 8589934592"
    "mvt_16g|polybench/MVT|./mvt.exe 17179869184"
    "mvt_32g|polybench/MVT|./mvt.exe 34359738368"
    "mvt_40g|polybench/MVT|./mvt.exe 42949672960"
    "mvt_48g|polybench/MVT|./mvt.exe 51539607552"
    "mvt_64g|polybench/MVT|./mvt.exe 68719476736"

    # --- needle (rodinia): uses -mb flag ---
    "needle_512m|rodinia/nw|./needle -mb 512"
    "needle_4g|rodinia/nw|./needle -mb 4096"
    "needle_8g|rodinia/nw|./needle -mb 8192"
    "needle_16g|rodinia/nw|./needle -mb 16384"
    "needle_32g|rodinia/nw|./needle -mb 32768"
    "needle_40g|rodinia/nw|./needle -mb 40960"
    "needle_48g|rodinia/nw|./needle -mb 49152"
    "needle_64g|rodinia/nw|./needle -mb 65536"
)

# Run benchmarks grouped by increasing memory size, so slow large-memory cases
# do not delay results for smaller memory sizes. Names are expected to end in
# one of these suffixes, e.g. gemm_4g or int_set_seq_512m.
read -ra MEMORY_ORDER <<< "${MEMORY_ORDER:-512m 4g 8g 16g 32g 40g 48g 64g}"

ordered_benchmarks() {
    local size entry bname

    for size in "${MEMORY_ORDER[@]}"; do
        for entry in "${BENCHMARKS[@]}"; do
            IFS='|' read -r bname _ <<< "$entry"
            [[ "$bname" == *_"$size" ]] && echo "$entry"
        done
    done
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Run a command and return its wall time in seconds (decimal).
measure_wall() {
    local dir="$1"; shift
    local cmd=("$@")
    local t0 t1
    t0=$(date +%s%N)
    (cd "$SCRIPT_DIR/$dir" && "${cmd[@]}" >/dev/null 2>&1)
    t1=$(date +%s%N)
    # nanoseconds -> seconds with 3 decimal places
    echo "scale=3; ($t1 - $t0) / 1000000000" | bc
}

# Convert a seconds value like 0.05 into integer nanoseconds.
seconds_to_ns() {
    awk -v s="$1" 'BEGIN { printf "%.0f", s * 1000000000 }'
}

# Sleep until an absolute CLOCK_REALTIME timestamp in nanoseconds.
sleep_until_ns() {
    local target_ns="$1"
    local now_ns remaining_ns remaining_s

    now_ns=$(date +%s%N)
    if (( now_ns < target_ns )); then
        remaining_ns=$((target_ns - now_ns))
        remaining_s=$(awk -v ns="$remaining_ns" 'BEGIN { printf "%.9f", ns / 1000000000 }')
        sleep "$remaining_s"
    fi
}

# After tracking has started, query query_count times evenly inside the
# sleep_interval window, and stop tracking exactly at the end of the window.
tracking_window() {
    local bench_pid="$1"
    local sleep_interval="$2"
    local query_count="$3"
    local row_name="$4"
    local rep="$5"
    local start_ns="$6"

    local interval_ns stop_ns stopper_pid q target_ns now_ns dump_file pages
    interval_ns=$(seconds_to_ns "$sleep_interval")
    stop_ns=$((start_ns + interval_ns))

    # Stop is timed independently of dump/query latency.
    (
        sleep_until_ns "$stop_ns"
        echo "$bench_pid" > "$DT_STOP" 2>/dev/null || true
    ) &
    stopper_pid=$!

    # Query times are strictly inside the interval:
    #   k * sleep_interval / (query_count + 1), k = 1..query_count.
    for ((q=1; q<=query_count; q++)); do
        target_ns=$((start_ns + (interval_ns * q) / (query_count + 1)))
        sleep_until_ns "$target_ns"

        now_ns=$(date +%s%N)
        (( now_ns >= stop_ns )) && break

        # Stop querying if the benchmark already finished.
        kill -0 "$bench_pid" 2>/dev/null || break

        dump_file="$DUMP_DIR/${row_name}_rep${rep}_query${q}_pid_${bench_pid}.txt"
        echo "$bench_pid" > "$DT_CUTOVER" 2>/dev/null || true
        cat "$DT_DUMP" > "$dump_file" 2>/dev/null || true
        pages=$(grep -c '^0x' "$dump_file" 2>/dev/null || true)
        echo "  [tracking] ${row_name} rep=${rep} query=${q}/${query_count}: dirty pages: ${pages}  (dump -> ${dump_file})" >&2
    done

    wait "$stopper_pid" 2>/dev/null || true
}

measure_wall_tracked() {
    local sleep_interval="$1"; shift
    local query_count="$1"; shift
    local row_name="$1"; shift
    local rep="$1"; shift
    local dir="$1"; shift
    local cmd=("$@")
    local t0 t1 bench_pid started=0 tracking_start_ns tracker_pid=""

    t0=$(date +%s%N)
    (cd "$SCRIPT_DIR/$dir" && exec "${cmd[@]}" >/dev/null 2>&1) &
    bench_pid=$!

    # Poll until the CUDA va_space exists and tracking starts successfully.
    # The write returns -EFAULT if called before cudaMallocManaged, so we
    # retry until it succeeds or the process exits.
    for ((poll=0; poll<200; poll++)); do
        if echo "$bench_pid cumulative" > "$DT_START" 2>/dev/null; then
            tracking_start_ns=$(date +%s%N)
            started=1
            tracking_window "$bench_pid" "$sleep_interval" "$query_count" "$row_name" "$rep" "$tracking_start_ns" &
            tracker_pid=$!
            break
        fi
        # Stop polling if the benchmark already finished.
        kill -0 "$bench_pid" 2>/dev/null || break
        sleep "$START_POLL_INTERVAL"
    done

    if [[ $started -eq 0 ]]; then
        echo "  [warn] tracking never started for pid $bench_pid" >&2
    fi

    wait "$bench_pid"
    t1=$(date +%s%N)

    # Ensure the stop/query helper is done before the next repetition, but do
    # not include this cleanup wait in the benchmark wall-time measurement.
    if [[ -n "$tracker_pid" ]]; then
        wait "$tracker_pid" 2>/dev/null || true
    fi

    echo "scale=3; ($t1 - $t0) / 1000000000" | bc
}

# Convert values like 0.05 into safe CSV row-name tokens like 0p05.
tagify() {
    local s="$1"
    s="${s//./p}"
    echo "$s"
}

# Compute average and stddev of a space-separated list of numbers.
stats() {
    python3 - "$@" <<'EOF'
import sys, math
vals = list(map(float, sys.argv[1:]))
n = len(vals)
avg = sum(vals) / n
std = math.sqrt(sum((v - avg)**2 for v in vals) / n)
print(f"{avg:.3f} {std:.3f}")
EOF
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

echo "row_name,benchmark,sleep_interval_s,queries_per_epoch,reps,off_avg_s,off_std_s,on_avg_s,on_std_s,overhead_pct" | tee "$OUT_CSV"

while IFS= read -r entry; do
	echo $entry
    IFS='|' read -r name dir cmd_str <<< "$entry"
    read -ra cmd <<< "$cmd_str"

    # If a filter list was given, skip benchmarks not in it
    if [[ ${#FILTER[@]} -gt 0 ]]; then
        match=0
        for f in "${FILTER[@]}"; do
            [[ "$name" == "$f"* ]] && match=1 && break
        done
        [[ $match -eq 0 ]] && continue
    fi

    # Check the binary exists
    bin="$SCRIPT_DIR/$dir/${cmd[0]}"
    if [[ ! -x "$bin" ]]; then
        echo "  SKIP $name — binary not found: $bin" >&2
        continue
    fi

    echo -n "  $name (OFF baseline) ... " >&2
    off_times=()
    for ((i=0; i<REPS; i++)); do
        t=$(measure_wall "$dir" "${cmd[@]}")
        off_times+=("$t")
        echo -n "." >&2
    done
    echo " done" >&2
    read -r off_avg off_std <<< "$(stats "${off_times[@]}")"

    for sleep_interval in "${SLEEP_INTERVALS[@]}"; do
        sleep_tag=$(tagify "$sleep_interval")

        for query_count in "${QUERY_COUNTS[@]}"; do
            row_name="${name}__sleep_${sleep_tag}__queries_${query_count}"

            echo -n "  $row_name (ON) ... " >&2
            on_times=()
            for ((i=0; i<REPS; i++)); do
                t=$(measure_wall_tracked "$sleep_interval" "$query_count" "$row_name" "$i" "$dir" "${cmd[@]}")
                on_times+=("$t")
                echo -n "." >&2
            done
            echo " done" >&2

            read -r on_avg on_std <<< "$(stats "${on_times[@]}")"

            overhead=$(python3 -c "
off=$off_avg; on=$on_avg
pct = 100.0 * (on - off) / off if off > 0 else 0.0
print(f'{pct:+.1f}')
")

            row="$row_name,$name,$sleep_interval,$query_count,$REPS,$off_avg,$off_std,$on_avg,$on_std,$overhead"
            echo "$row" | tee -a "$OUT_CSV"
        done
    done
done < <(ordered_benchmarks)

echo ""
echo "Results saved to $OUT_CSV"

