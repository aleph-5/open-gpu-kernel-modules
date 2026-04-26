# CS614 Artifact Evaluation — GPU Dirty Page Tracking in NVIDIA UVM

**Project:** Low-Overhead Dirty Access Tracking in GPU-UVM
**Team:** Aditi Khandelia, Arush Upadhyaya, Kushagra Srivastava, Sankalp Mittal, Vidhi Jain

**Course:** CS614

This document is the artifact-evaluation roadmap for the project.

---

## 1. Artifact Directory Structure

```
open-gpu-kernel-modules/
|
|-- kernel-open/nvidia-uvm/        # Modified NVIDIA UVM driver
|
|-- correctness_tests/             # 16 CUDA correctness suites + run_all.py
|   |-- access_type_filtering/     # writes vs. reads, atomics, prefetch
|   |-- address_range_filtering/   # range filter, sub-page rounding
|   |-- backend_correctness/       # invariants of dump output
|   |-- cumulative_mode/           # cumulative-vs-delta semantics
|   |-- determinism/               # cutover border, pause/resume
|   |-- first_write_wins/          # FWW under repetition and races
|   |-- footprint_clearing/        # PTE permissions restored on stop
|   |-- generic/                   # broad smoke / integration suite
|   |-- lifecycle_races/           # concurrent start/stop/dump races
|   |-- ordering/                  # timestamp monotonicity, epoch isolation
|   |-- procfs_robustness/         # malformed input / error paths
|   |-- scale/                     # 100K pages, paginated reads
|   |-- snapshot_isolation/        # atomic cutover snapshot semantics
|   |-- stats_accuracy/            # uvm_dirty_ds_stats counters
|   |-- table_lifecycle/           # create / destroy / reinit cycles
|   |-- tracking_hints/            # backend-selection hint interface
|   |-- run_all.py                 # runs every suite in canonical order
|   `-- suite_runner.py            # shared per-suite runner
|
|-- performance_tests/             # Performance benchmarks
|   |-- common/                    # shared procfs helper headers
|   |-- general_performance/       # tc01/tc02 wall-clock overhead
|   |-- operation_microbenchmarks/ # per-op latency (insert / lookup / lifecycle)
|   |-- stress_tests/              # high-write-pressure stress
|   `-- plots/                     # plot_*.py scripts
|
|-- cuda-oversubscribed-benchmarks-main/  # benchmark suite
|   |-- synthetic_benchmarks/      # int_set_uvm, vector_add, random_acc, ...
|   |-- nvidia-samples/            # sgemm
|   |-- polybench/                 # PolyBench/GPU UVM ports
|   |-- rodinia/                   # Rodinia UVM ports
|   |-- helpers/                   # Shared benchmark helpers
|   |-- dev/                       # Dev / scratch
|   |-- perf_logs/                 # Logs from runs
|   |-- run_overhead_benchmark.sh  # Driver script (REPS, benchmark filter)
|   |-- run_overhead_benchmark_windowed.sh  # Permuted benchmarks runner
|   |-- overhead_results.csv       # Latest run output
|   |-- overhead_results_final-vector.csv   # Reference results (vector backend)
|   |-- Makefile                   # Top-level benchmark build
|   `-- README.md                  # Benchmark suite docs
|
|-- compile.sh                     # Build + reload nvidia_uvm in one step
|-- Makefile                       # Top-level kernel-module build
|-- README.md                      # (This File)
```

---

## 2. Setup Instructions

### 2.1 Hardware

| Resource    | Required                                                     |
|-------------|--------------------------------------------------------------|
| CPU         | x86-64, >= 8 cores recommended for kernel build              |
| Memory      | >= 16 GB RAM (>= 64 GB recommended for oversubscription tests) |
| Storage     | ~25 GB free on the system partition (kernel + module build, CUDA toolkit, benchmark data) |
| GPU         | NVIDIA GPU with UVM support (tested on **NVIDIA A40, 45 GB VRAM**) |
| Privileges  | **root** required (writes to `/proc/driver/nvidia-uvm/`)     |


### 2.2 Software

| Component          | Version (tested)                       |
|--------------------|----------------------------------------|
| Operating System   | Ubuntu 22.04 LTS                       |
| Linux Kernel       | **6.8** (No modifications)             |
| NVIDIA Driver      | **580.65.06** (this tree, modified)    |
| CUDA Toolkit       | >= 12.x (for `nvcc`)                   |
| Python             | 3.8+ (for the `run_tests.py` scripts)  |
| Standard utilities | `make`, `gcc`, `bash`, `modprobe`      |

Verify the driver and toolkit are present:

```bash
nvidia-smi
nvcc --version
```

### 2.3 Nvidia Driver Module Build

The repository **is** the modified driver. Build and load with the bundled script:

```bash
sudo ./compile.sh
```

This runs `make modules`, installs it, and reloads via `modprobe`. It removes the in-memory `nvidia_uvm`, `nvidia_drm`, and `nvidia_modeset` first.

Manual equivalent:

```bash
sudo make modules -j$(nproc)
sudo make modules_install -j$(nproc)
sudo modprobe -r nvidia_uvm
sudo modprobe nvidia_uvm
```

To verify that the modules are loaded

```bash
ls /proc/driver/nvidia-uvm/dirty_*
```

### 2.4 Switching Backends (optional, for cross-backend evaluation)

Seven dirty-set backends are implemented (Bitmap, XArray, Vector sorted, Vector unsorted, Chunked, Nested Bitmap, Linked List). Each is selected by setting the `.ops` pointer in `kernel-open/nvidia-uvm/uvm_common.c` and rebuilding via `compile.sh`. Run-time selection between sequential / random hint backends is exposed via `dirty_tracking_hint` PROCFS interface.

---

## 3. Features Supported

The implementation is feature-complete for **GPU write-fault tracking with first-write-wins semantics, nanosecond timestamps, atomic cutover snapshots, pause/resume, cumulative or delta mode, address-range filtering, and a stats subsystem.** The detailed coverage matrix is the same one used in the final report; reproduced below in tabular form.

| # | Feature | Test suite (script) | Key parameters | Objective | Expected outcome |
|---|---------|---------------------|----------------|-----------|------------------|
| 1 | Access-type discrimination (write/atomic recorded, read/prefetch not) | [`correctness_tests/access_type_filtering/run_tests.py`](correctness_tests/access_type_filtering/run_tests.py) | 4 tests, single-stream | Read-only and prefetch must not produce dirty entries; atomic RMW must | 4/4 PASS |
| 2 | Address-range filtering and isolation | [`correctness_tests/address_range_filtering/run_tests.py`](correctness_tests/address_range_filtering/run_tests.py) | 5 tests; multi-thread; sub-page accesses; inverted/zero-length ranges | Range filter restricts dump; sub-page accesses still produce page-aligned entries | 5/5 PASS |
| 3 | Backend invariants (alignment, no duplicates, monotone TS) | [`correctness_tests/backend_correctness/run_tests.py`](correctness_tests/backend_correctness/run_tests.py) | 9 tests; sparse + dense write patterns | All entries page-aligned, unique, ordered; cudaFree does not erase records | 9/9 PASS |
| 4 | Cumulative-mode accumulation across epochs | [`correctness_tests/cumulative_mode/run_tests.py`](correctness_tests/cumulative_mode/run_tests.py) | 4 tests; cumulative vs. delta toggle | Cumulative dump = union of all epochs; FWW preserved across cutover | 4/4 PASS |
| 5 | Determinism: cutover border + pause/resume | [`correctness_tests/determinism/run_tests.py`](correctness_tests/determinism/run_tests.py) | 2 tests; tightly-interleaved write phases | Pages written before cutover only in snap1; paused writes never recorded | 2/2 PASS |
| 6 | First-write-wins (FWW) policy | [`correctness_tests/first_write_wins/run_tests.py`](correctness_tests/first_write_wins/run_tests.py) | 3 tests; up to 64 concurrent CUDA streams on the same page | One entry per page per epoch; original timestamp preserved | 3/3 PASS |
| 7 | Footprint clearing on stop | [`correctness_tests/footprint_clearing/run_tests.py`](correctness_tests/footprint_clearing/run_tests.py) | 3 tests; uses write-latency timing as oracle | Pre-tracking RW pages restored to RW; previously unmapped pages stay unmapped | 3/3 PASS |
| 8 | Generic integration smoke suite | [`correctness_tests/generic/run_tests.py`](correctness_tests/generic/run_tests.py) | 12 tests in one binary | Core procfs interface end-to-end | 12/12 PASS |
| 9 | Lifecycle races (concurrent start/stop/dump) | [`correctness_tests/lifecycle_races/run_tests.py`](correctness_tests/lifecycle_races/run_tests.py) | 7 tests; multi-process; rapid cycles; killed owner | No crash, deadlock, or leak; second concurrent start gets `EBUSY` | 7/7 PASS |
| 10 | Temporal ordering | [`correctness_tests/ordering/run_tests.py`](correctness_tests/ordering/run_tests.py) | 8 tests; multi-stream flood (1M threads) | Timestamps monotone non-decreasing; phase-A max ≤ phase-B min | 8/8 PASS |
| 11 | Procfs robustness / error paths | [`correctness_tests/procfs_robustness/run_tests.py`](correctness_tests/procfs_robustness/run_tests.py) | 8 tests; malformed input; double start/stop; wrong PID | Well-defined errors (EBUSY/EINVAL); no driver-state corruption | 8/8 PASS |
| 12 | Scale | [`correctness_tests/scale/run_tests.py`](correctness_tests/scale/run_tests.py) | 4 tests; up to 100K pages; small-chunk reads | No truncation; correct paginated dump; bitmap capacity edge handled | 4/4 PASS |
| 13 | Snapshot isolation | [`correctness_tests/snapshot_isolation/run_tests.py`](correctness_tests/snapshot_isolation/run_tests.py) | 4 tests; concurrent writers during dump | Snapshot uncorrupted; drained delta dump returns 0 entries | 4/4 PASS |
| 14 | Stats accuracy | [`correctness_tests/stats_accuracy/run_tests.py`](correctness_tests/stats_accuracy/run_tests.py) | 4 tests; toggle on/off | Insert counter == #distinct pages; counters reset on stop+start | 4/4 PASS |
| 15 | Table lifecycle | [`correctness_tests/table_lifecycle/run_tests.py`](correctness_tests/table_lifecycle/run_tests.py) | 5 tests; multi-reinit stress | Reinit clears entries; multi-allocation tracked in one session | 5/5 PASS |
| 16 | Backend-selection hint interface | [`correctness_tests/tracking_hints/run_tests.py`](correctness_tests/tracking_hints/run_tests.py) | 3 tests; switch hint between sessions | Output identical under SEQ vs. RAND hint; switch is safe between sessions | 3/3 PASS |
| 17 | Performance overhead (general) | [`performance_tests/general_performance/`](performance_tests/general_performance/) | tc01 sequential, tc02 random; tracking ON vs. OFF | Wall-time overhead in `results.csv` | See 7.2 |
| 18 | Per-operation microbenchmarks | [`performance_tests/operation_microbenchmarks/run_tests.py`](performance_tests/operation_microbenchmarks/run_tests.py) | tc01–tc05; per-op nanosecond cost | Insert / lookup / lifecycle per-call ns latency in CSVs | Numerical CSVs |
| 19 | Stress | [`performance_tests/stress_tests/tc01_stress_testing.cu`](performance_tests/stress_tests/tc01_stress_testing.cu) | N pages (default 1000) | No dropped page under high write pressure | All pages reported |
| 20 | Cross-benchmark / cross-regime overhead | [`cuda-oversubscribed-benchmarks-main/`](cuda-oversubscribed-benchmarks-main/) | 7 benchmarks x 8 sizes (512MB → 64GB) x 5 reps | Wall-time overhead vs. baseline | Reproduces Table in 7.2 |

### 3.1 Aggregate test result

**74 / 74 correctness tests pass** on the reference setup. Per-suite runtimes are 1.95 s – 3.05 s. No crashes, deadlocks, or assertion failures observed in any suite (lifecycle_races, snapshot_isolation, and procfs_robustness are the dedicated stressors and all pass).

---

## 4. Assumptions and Unsupported Features

### Assumptions

- **Single-owner per session.** Only one PID may hold an active tracking session at a time; a second `start` returns `EBUSY`. This is enforced by `uvm_dirty_lifecycle_lock` and an owner-PID check.
- **Root only.** All procfs entries under `/proc/driver/nvidia-uvm/dirty_*` require write/read by a privileged process.
- **GPU-managed memory only.** The system tracks pages owned by UVM (`cudaMallocManaged`). Plain `cudaMalloc` (device-only) and host-pinned allocations are out of scope.
- **Linux kernel 6.8 + driver 580.65.06.** Other kernel versions / driver versions are not validated.

### Unsupported / out of scope

- **CPU-side dirty tracking.** Only GPU write faults are intercepted; CPU writes to managed memory are not currently recorded (planned via `mmu_notifier`).
- **Per-page write counters / write frequency.** Only first-write timestamp is stored.
- **Automatic backend selection.** The `dirty_tracking_hint` interface allows manual selection between sequential and random backends. Pattern driven automatic selection is future work.
- **Multi-GPU partitioning.** All tests are single-GPU.

---

## 5. The procfs Interface (quick reference)

All entries live in `/proc/driver/nvidia-uvm/`.

| Entry | Direction | Semantics |
|---|---|---|
| `dirty_tracking_start` | write | `<PID> {delta\|cumulative}` — initialise tables, downgrade GPU PTEs to READ_ONLY, begin recording |
| `dirty_tracking_pause` | write | `<PID>` — drain in-flight faults, disable recording (state preserved) |
| `dirty_tracking_resume` | write | `<PID>` — re-downgrade PTEs and re-arm recording |
| `dirty_tracking_stop` | write | `<PID>` — restore RW permissions, destroy tables, flush stats |
| `dirty_tracking_query_cutover` | write | `<PID>` — atomic swap live → snapshot, install fresh live table |
| `dirty_tracking_query_dump` | read | one line per dirty page: `0x<addr> <ns_timestamp>` |
| `dirty_tracking_hint` | write | select backend (`WRITE_SEQ` / `WRITE_RAND`) for next session |
| `dirty_ds_stats_toggle` | write | `enable` / `disable` per-op stats accumulation |
| `dirty_ds_stats` | read | per-op counts, average ns latency, lock-wait time |

---

## 6. Getting Started

This section reproduces a Hello-world-sized run.

### 6.1 Build and load

```bash
bash compile.sh
ls /proc/driver/nvidia-uvm/dirty_*    # five+ entries should appear
```

### 6.2 Run the correctness test suite to confirm correctness

```bash
cd correctness_tests
sudo python3 run_all.py
```

Expected: `16/16 suites PASS` (74 individual tests total).

### 6.3 Drive the procfs interface by hand (~ 5 min)

In one terminal, launch any CUDA workload that uses `cudaMallocManaged` and writes to it (e.g. one of the [performance_tests/general_performance](performance_tests/general_performance/) binaries). Capture its PID. In another terminal run:

```bash
PID=<pid_of_workload>
echo "$PID delta" | sudo tee /proc/driver/nvidia-uvm/dirty_tracking_start
```
Wait for the process to run for sometime and then run this
```bash
echo "$PID" | sudo tee /proc/driver/nvidia-uvm/dirty_tracking_query_cutover
```
This will record and store all the writes recorded upto the execution of this write and can be accesed by
```bash
sudo cat /proc/driver/nvidia-uvm/dirty_tracking_query_dump > epoch1.txt
```
You can stop the tracking using
```bash
echo "$PID" | sudo tee /proc/driver/nvidia-uvm/dirty_tracking_stop
```

### 6.4 Supply your own input

Any program that allocates with `cudaMallocManaged` and writes from a CUDA kernel is a valid input. Wrap the launch + the procfs `start`/`cutover`/`dump`/`stop` sequence as in 6.3 (or look at any `tc*.cu` under `correctness_tests/` for self-contained examples that exercise the procfs interface from inside the test binary).

You can also provide hints to the system before startig the tracking via the `dirty_tracking_hint` PROCFS, if your workload has most writes in random order, do the following
```bash
echo "WRITE_RAND" | sudo tee /proc/driver/nvidia-uvm/dirty_tracking_hint
```
and if they are mostly sequential then do
```bash
echo "WRITE_SEQ" | sudo tee /proc/driver/nvidia-uvm/dirty_tracking_hint
```

## 7. Detailed Evaluation

### 7.1 Correctness experiments

| # | Purpose | Command | Estimated runtime | Expected result | Where to find the actual result |
|---|---------|---------|-------------------|-----------------|----------------------------------|
| 1 | Run every correctness suite in canonical order | `sudo python3 correctness_tests/run_all.py` | < 10 min | 74/74 PASS | Stdout per-suite PASS/FAIL summary |
| 2 | Run a single suite (e.g. ordering) | `cd correctness_tests/ordering && sudo python3 run_tests.py` | ~ 16 s | 8/8 PASS | Stdout |
| 3 | Run a single test in a suite | `sudo python3 run_tests.py tc01` |~ 2 s | PASS | Stdout |
| 4 | Re-run already-built binaries | `sudo python3 run_tests.py --no-build` | < 5 s/suite | PASS | Stdout |
| 5 | Verbose mode (show stdout of passing tests) | `sudo python3 run_tests.py --verbose` | ~ same as 2 | PASS | Stdout |

### 7.2 Performance experiments

#### P1 — General overhead (sequential vs. random write)

```bash
cd performance_tests/general_performance
make
sudo ./tc01_performance_overhead > results_seq.csv   # sequential write workload
sudo ./tc02_random_write_overhead > results_rand.csv # random write workload
```

- **Estimated runtime:** ~5–10 sec (build) + ~3 min per benchmark binary.
- **Expected result:** Two CSV files with wall-time overhead; tracking ON increases runtime proportionally to write-fault density.
- **Access:** `results_seq.csv` and `results_rand.csv` in `performance_tests/general_performance/`.

#### P2 — Per-operation microbenchmarks

```bash
cd performance_tests/operation_microbenchmarks
sudo python3 run_tests.py
```

- **Estimated runtime:** ~2–5 min total.
- **Expected result:** Per-operation ns-resolution costs:
  - `tc01_basic_rmw` — basic read-modify-write fault path
  - `tc02_duplicate_fault_skip_cost.csv` — cost of FWW skip
  - `tc03_record_contention_cost.csv` — multi-stream insert contention
  - `tc04_single_fault_record_cost.csv` — single-fault record cost
  - `tc05_table_{init,destroy,reinit}_cost.csv` — lifecycle costs
- **Access:** CSVs alongside the test sources; plot with `performance_tests/plots/plot_*.py`.

#### P3 — Stress (page-miss-rate at high write pressure)

```bash
cd performance_tests/stress_tests
make
sudo ./tc01_stress_testing 100000      # N = pages, default 1000
```

- **Estimated runtime:** ~2–3 min (build + run at N=100000).
- **Expected result:** Every written page reported; stdout reports any miss.
- **Access:** Stdout.

#### P4 — Cross-benchmark overhead (the headline numbers)

```bash
cd cuda-oversubscribed-benchmarks-main
make -j$(nproc)                                  # Builds all benchmarks
sudo bash run_overhead_benchmark.sh 5            # REPS=5 (default, all benchmarks)
# OR:
sudo bash run_overhead_benchmark.sh 10 int_set_4k gemm  # REPS=10, specific benchmarks
```

- **Estimated runtime:** ~30–90 min for the full sweep (7 benchmarks x 8 sizes x 5 reps); use the specific-benchmark form for a quick spot-check (~5–10 min).
- **Expected result:** `overhead_results.csv` with wall-time tracking OFF vs. ON, overhead percentage, and dirty page count per benchmark x size combination.
- **Access:** `cuda-oversubscribed-benchmarks-main/overhead_results.csv`

#### P5 — Per-operation latency by backend

To test the latency of different data structures:

1. Edit `kernel-open/nvidia-uvm/uvm_common.c` to set `.ops` to the chosen backend.
2. `sudo ./compile.sh`.
3. `echo enable | sudo tee /proc/driver/nvidia-uvm/dirty_ds_stats_toggle`.
4. Run the workload (e.g. `int_set_uvm` 4K-stride at 16GB).
5. `cat /proc/driver/nvidia-uvm/dirty_ds_stats` — average insert / lookup / for-each / lock-wait nanoseconds.

- **Estimated runtime:** ~15–20 min per backend (rebuild + workload run).
- **Expected result:** Per-backend ns averages for insert, lookup, for-each, and lock-wait; values vary by backend (e.g., bitmap insert is O(1), sorted-vector lookup is O(log n)).
- **Access:** `cat /proc/driver/nvidia-uvm/dirty_ds_stats` after step 5 above.

### 7.3 Findings (crash / deadlock / assertion failures observed during evaluation)

- **None.** Every correctness suite passes on the reference setup. The dedicated stress suites (`lifecycle_races`, `snapshot_isolation`, `procfs_robustness`, `scale`, `stress_tests`) all pass without driver-side warnings or oopses.

---

## 8. Pointers to Further Material

- Correctness Tests: [`correctness_tests/`](./correctness_tests/)
- Performance Tests [`performance_tests/`](./performance_tests/)
- Benchmarks : [`cuda-oversubscribed-benchmarks-main/`](./cuda-oversubscribed-benchmarks-main/)
