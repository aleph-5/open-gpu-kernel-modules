#!/usr/bin/env bash
# Usage: sudo bash run_stats_benchmark.sh <mb> [stride-4k]
# Example: sudo bash run_stats_benchmark.sh 512 stride-4k
# Runs int_set_uvm with dirty tracking + stats enabled, polls until UVM context is ready.

set -e

MB=${1:-512}
STRIDE=${2:-}

BENCH_DIR="$(dirname "$0")/synthetic_benchmarks"
TRACKING_START=/proc/driver/nvidia-uvm/dirty_tracking_start
TRACKING_STOP=/proc/driver/nvidia-uvm/dirty_tracking_stop
STATS_TOGGLE=/proc/driver/nvidia-uvm/dirty_ds_stats_toggle
STATS_OUT=/proc/driver/nvidia-uvm/dirty_ds_stats

CMD="$BENCH_DIR/int_set_uvm.out -mb $MB"
[ -n "$STRIDE" ] && CMD="$CMD -$STRIDE"

echo "enable" > "$STATS_TOGGLE"

$CMD & PID=$!
echo "Launched PID $PID"

until echo "$PID cumulative" > "$TRACKING_START" 2>/dev/null; do
    sleep 0.01
done
echo "Tracking started on PID $PID"

wait $PID

echo "$PID" > "$TRACKING_STOP" || true
echo "disable" > "$STATS_TOGGLE" || true

echo ""
echo "=== Stats ==="
cat "$STATS_OUT"

