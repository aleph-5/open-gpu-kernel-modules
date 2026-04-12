#!/usr/bin/env bash
# Kill all processes holding nvidia-uvm open so the module can be unloaded.

set -euo pipefail

SIGNAL="${1:--9}"   # default SIGTERM; pass -9 for SIGKILL

# Gather PIDs from every source that can hold nvidia-uvm open.
pids=()

# 1. Processes with /dev/nvidia-uvm* open (most common)
for dev in /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    if [[ -e "$dev" ]]; then
        mapfile -t tmp < <(fuser "$dev" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true)
        pids+=("${tmp[@]}")
    fi
done

# 2. Module refcount holders visible via /proc/*/maps (catches mmapped regions)
if [[ -d /proc ]]; then
    mapfile -t tmp < <(
        grep -rl 'nvidia-uvm' /proc/[0-9]*/maps 2>/dev/null \
        | sed 's|/proc/||;s|/maps||' || true
    )
    pids+=("${tmp[@]}")
fi

# De-duplicate and remove own PID
self=$$
mapfile -t pids < <(printf '%s\n' "${pids[@]}" | sort -u | grep -v "^${self}$" || true)

if [[ ${#pids[@]} -eq 0 ]]; then
    echo "No processes found using nvidia-uvm."
    exit 0
fi

echo "Sending signal $SIGNAL to PIDs: ${pids[*]}"
for pid in "${pids[@]}"; do
    # Print process name for visibility
    name=$(cat /proc/"$pid"/comm 2>/dev/null || echo "<unknown>")
    echo "sudo pkill $SIGNAL $pid  ($name)"
    sudo pkill "$SIGNAL" "$pid" 2>/dev/null || echo "  (already gone: $pid)"
done
