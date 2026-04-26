# CUDA UVM Benchmarks

Benchmark suites for evaluating CUDA Unified Virtual Memory (UVM) behaviour
under oversubscription. Obtained from Pranjal Singh (prsingh@cse.iitk.ac.in)
with modifications.

## Benchmark Suites

### Polybench (`polybench/`)

Linear-algebra kernels (2MM, 3MM, 2DCONV, 3DCONV, ATAX, BICG, MVT, CORR,
GRAMSCHM, …) ported to UVM by the UVMBench authors. Working-set size is
configurable via CLI. See `polybench/README.md`.

### Rodinia (`rodinia/`)

- `nw/`: Needleman-Wunsch protein sequence alignment.

### NVIDIA Samples (`nvidia-samples/`)

- `sgemm/`: Single-precision GEMM using cuBLAS, used as a compute-heavy
  reference workload.

### Synthetic Benchmarks (`synthetic_benchmarks/`)

Microbenchmarks targeting specific UVM access patterns:

| File | Description |
|------|-------------|
| `int_set_uvm.cu` | Sequential integer write (UVM); stride configurable (4 B / 4 KiB / 2 MiB) |
| `int_set_nonUvm.cu` | Same write pattern without UVM, for baseline comparison |
| `random_acc.cu` | Random page accesses (from Tyler Allen's `uvm-eval`) |
| `gpu_read.cu` | GPU read of a UVM array; fraction of pages written is tunable |
| `cpu_init.cu` | CPU-side array initialisation; isolates CPU-init cost under UVM |
| `vector_add.cu` / `vector_mul.cu` | Simple vector arithmetic over UVM arrays |
| `read_write_same_page.cu` | Repeatedly reads and writes the same page to stress the fault path |

> **Note:** synthetic benchmarks must be compiled separately.
> ```bash
> cd synthetic_benchmarks && make
> ```
> The top-level `make` faces an import error

## Usage

- `make` works in each benchmark directory and at the repo root (except
  synthetic benchmarks — see above).
- Each binary accepts a `-h` flag listing its options.
- `README.md` and `performance.md` in each subdirectory document expected
  behaviour and suggested memory hints.

### Overhead Benchmark (`run_overhead_benchmark.sh`)

Measures dirty-tracking pipeline overhead across all workloads. Must be run
as root with the patched `nvidia-uvm` module loaded:

```bash
sudo bash run_overhead_benchmark.sh [REPS [benchmark ...]]
```

Results are written to `overhead_results.csv`.

> **Error logs during the run are expected and not a problem.** The script
> polls `/proc/driver/nvidia-uvm/dirty_tracking_*` at a fixed interval; the
> kernel may log warnings on each poll while a benchmark is running. These
> are a side-effect of polling and do not affect the measured results.

### Helpers and Macros

- Allocation macros in `dev/cuda-macros-v1.h` (symlinked into
  `synthetic_benchmarks/`) enable memory-hint flags from the CLI.
- See `dev/README.md`, `perf_logs/README.md`, and `helpers/README.md` for
  additional tooling.

