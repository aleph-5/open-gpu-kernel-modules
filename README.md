# GPU Dirty Page Tracking in NVIDIA UVM

This repository is a fork of [NVIDIA's open-source GPU kernel modules](https://github.com/NVIDIA/open-gpu-kernel-modules) with modifications to the NVIDIA UVM kernel module that add per-process dirty page tracking for GPU-managed memory.

This work is part of the CS614 course project.

**Team:** Aditi Khandelia, Arush Upadhyaya, Kushagra Srivastava, Sankalp Mittal, Vidhi Jain

---

## Motivation

Iterative pre-copy live migration works by repeatedly transferring dirty memory pages from a running process to a destination host, until the remaining working set is small enough to complete with a brief final pause. For CPU memory, the Linux kernel exposes this information through soft-dirty bits in page-table entries, readable via `/proc/PID/pagemap`.

No equivalent mechanism exists for GPU-managed memory. NVIDIA's Unified Virtual Memory (UVM) subsystem, which underpins `cudaMallocManaged` and handles demand-paging between host and device, provides no interface to query which pages a process has modified. Any migration or checkpointing system must therefore conservatively copy the entire GPU memory footprint on every iteration, making iterative pre-copy migration of large GPU workloads impractical.

This project implements per-process dirty page tracking directly within the open-source NVIDIA UVM kernel module. UVM already intercepts every GPU page fault to manage demand-paging; the write-fault servicing path is augmented to record each written page in an in-kernel data structure without introducing additional hardware overhead. A `procfs`-based interface allows a privileged migration daemon to start and stop tracking epochs and retrieve the complete set of pages written by a given process, with page-level granularity and a per-page nanosecond-resolution timestamp.

---

## Repository Structure

```
open-gpu-kernel-modules/
|
|-- kernel-open/nvidia-uvm/     # NVIDIA UVM driver source (modified)
|
|-- correctness_tests/          # Per-suite CUDA correctness tests
|   |-- generic/                # Basic end-to-end tracking checks
|   |-- table_lifecycle/        # Epoch start/stop/restart lifecycle
|   |-- ordering/               # Timestamp and epoch isolation
|   |-- address_range_filtering/# Range-based query filtering
|   |-- latency/                # Fault-to-visibility and timestamp coherence
|   |-- concurrency/            # Concurrent write stress tests
|   |-- stress/                 # High write-pressure page coverage
|   `-- pid_attribution_isolation/ # Per-process isolation checks
|
|-- performance_tests/          # Performance benchmarks
|   |-- stress_tests/           # Page miss rate under high write load
|   |-- operation_microbenchmarks/ # Per-fault and table operation costs
|   |-- scaling/                # Overhead scaling across page counts
|   `-- results.csv             # Benchmark output data
|
|-- midsem-review/              # Mid-semester project report (LaTeX)
|-- compile.sh                  # Script to build and reload the kernel module
`-- Makefile                    # Top-level build entry point
```

Each correctness test suite contains individual `tc*.cu` test files and a `run_tests.py` script that builds and runs all tests in the suite, reporting per-test PASS/FAIL results.

---

## How It Works

A tracking epoch begins when a privileged process writes to `dirty_pids_start_track`. This initialises a fresh per-process table and invalidates all current GPU page-table entries by unmapping them, forcing every subsequent GPU access to generate a fresh page fault. Each write fault is intercepted in the driver's fault-servicing path, which records the faulting page along with a nanosecond-resolution timestamp before re-establishing the mapping and replaying the faulting instruction.

The tracking state uses a two-level structure:

- `struct dirty_page_info` -- a per-page record holding the page number, the `ktime_get_ns()` timestamp of the first observed write fault, and the PID of the faulting process.
- `struct uvm_dirty_page_table` -- a per-process container holding an `xarray` mapping page numbers to `dirty_page_info` pointers.
- A global `xarray` (`pid_to_page_table`) maps each tracked process's `tgid` to its `uvm_dirty_page_table`.

Entries are recorded under a first-write-wins policy: an existing entry is never overwritten, so the stored timestamp always reflects the earliest observed write to that page within the current epoch.

---

## procfs Interface

User-space interacts with the tracker through five entries under `/proc/driver/nvidia-uvm/`:

| Entry | Access | Description |
|---|---|---|
| `dirty_pids_start_track` | write | Initialises (or re-initialises) the tracking table for `current->tgid` and triggers GPU-mapping invalidation to begin a new epoch. |
| `dirty_pids_stop_track` | write | Destroys the table for `current->tgid`, freeing all recorded entries. |
| `dirty_pid_to_query` | write | Sets the PID whose table is consulted on subsequent `dirty_pages` reads. |
| `dirty_pages` | read | Emits one line per recorded page in the format `0xaddr timestamp_ns pid`. Reads are non-destructive. |
| `dirty_range` | write | Accepts a pair of hex addresses `0xstart 0xend` (end exclusive) to restrict subsequent `dirty_pages` reads to a specific virtual address window. |

---

## Modified Driver Files

All modifications are under `kernel-open/nvidia-uvm/`:

- `uvm_common.h` / `uvm_common.c` -- Primary implementation. Defines the data structures, implements table lifecycle (`init`/`destroy`), fault recording, page lookup, and the `procfs` handler (`uvm_dirty_procfs_init`).
- `uvm_gpu_replayable_faults.c` -- Hooks into the GPU write-fault replay path to call `uvm_dirty_page_table_record` for `UVM_FAULT_ACCESS_TYPE_WRITE` faults.
- `uvm_va_block.c` -- Invokes the dirty tracking record function when a write fault is serviced on a VA block.
- `uvm_va_space.c` / `uvm_va_space.h` -- Implements `uvm_dirty_invalidate_all_gpu_mappings`, which walks all live VA spaces and blocks to unmap GPU PTEs at epoch start.
- `uvm.c` -- Top-level integration; calls `uvm_dirty_procfs_init` and `uvm_va_space_dirty_init` during module initialisation.

---

## Building and Loading the Driver

**Requirements:** Linux kernel 6.8 (custom build), NVIDIA driver 580.65.06, root access.

Build and reload the modified UVM kernel module using the provided script:

```bash
bash compile.sh
```

This runs `make modules`, installs the module, and reloads it via `modprobe`. The script removes the existing `nvidia_uvm` module and any dependent modules (`nvidia_drm`, `nvidia_modeset`) before loading the updated one.

To build manually:

```bash
sudo make modules -j$(nproc)
sudo make modules_install -j$(nproc)
sudo modprobe -r nvidia_uvm
sudo modprobe nvidia_uvm
```

---

## Running the Tests

All tests must be run as root because they write to `/proc/driver/nvidia-uvm/`.

### Correctness Tests

Each suite has a `run_tests.py` script that compiles and runs all tests in the suite:

```bash
cd correctness_tests/<suite_name>
sudo python3 run_tests.py
```

To skip the build step and run only specific tests:

```bash
sudo python3 run_tests.py --no-build tc01 tc03
```

To run all correctness suites at once, repeat the above for each directory under `correctness_tests/`.

### Performance Tests

```bash
cd performance_tests/stress_tests
make
sudo ./tc01_stress_testing [N]          # N = number of pages, default 1000

cd performance_tests/operation_microbenchmarks
make
sudo ./<benchmark_binary>

cd performance_tests/scaling
make
sudo ./<benchmark_binary>
```

Performance overhead results are written to `performance_tests/results.csv`.

---

## Test Results

### Correctness

Current result: **32/32 tests pass** across all suites.

| Suite | Tests | Result |
|---|---|---|
| `generic` | 12 | 12/12 |
| `table_lifecycle` | 5 | 5/5 |
| `ordering` | 8 | 8/8 |
| `address_range_filtering` | 5 | 5/5 |
| `latency` | 2 | 2/2 |

### Performance

The stress test confirms no dirty pages are missed under high write pressure. The overhead benchmark shows a significant performance cost: write-only workloads incur up to approximately 1000% overhead compared to a baseline without tracking active. This is caused by the forced page-fault round-trip for every write access after mapping invalidation. Reducing this overhead is the primary focus of the next phase of work.

---

## Development Environment

- Linux kernel 6.8 (custom build)
- NVIDIA UVM driver 580.65.06 (loaded as a kernel module)
- CUDA toolkit (for compiling tests)
- All tests and module operations require root privileges

---

## Current Status and Planned Work

The following features are implemented and tested:

- Per-process dirty page tracking with nanosecond-resolution timestamps
- Multi-process isolation
- Non-destructive epoch querying
- Address-range filtering via `dirty_range`

Planned extensions:

- **CPU-side dirty tracking** via integration with the `mmu_notifier` invalidation path, so that CPU writes are recorded in the same per-process table.
- **Per-page write counters** to support richer statistics such as write frequency and page age, useful for informed migration and eviction decisions.
- **Performance optimisation** including batching of `xarray` inserts and evaluation of bitmap-based representations for high-density dirty sets.
- **Memory footprint profiling** of the two-level `xarray` structure under large dirty sets.
- **Concurrency stress testing** under multi-process workloads and simultaneous range-filter updates.
- **A command-line interface** wrapping the `procfs` API for human-readable interaction.

---

## Base Repository

This is a fork of [NVIDIA/open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules). The original licensing information from the upstream repository is preserved in `COPYING`.
