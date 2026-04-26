# CUDA UVM Benchmarks

This repository has multiple benchmark suites, from multiple sources and 
with some modifications, obtained from Pranjal Singh (prsingh@cse.iitk.ac.in)

### Polybench

The official version of Polybench does not use UVM. However, UVMBench's 
authors ported it to UVM.  
We have made some modifications to it (see `./polybench/README.md`).

### Other Benchmarks

- `rodinia/nw`: Needleman-Wunsch Protein Alignment
- Synthetic benchmarks - memset (`int_set_uvm.cu`), random access
  (`random_acc.cu`)

## Usage

- `make` works in each directory, as well as the root.
- RAPIDS benchmarks need a virtual environment. See `rapids/README.md`.
- Each binary/directory has a `README.md` and `-h` flag.
- For some benchmarks, `README.md` and `performance.md` discuss the behaviour
  and appropriate memory hints.

#### Helpers and Macros: Porting Workloads

- Using allocation macros from `dev/cuda-macros-v1.h` also enables memory hints
  from the CLI.
- Reference (to clean up and port) benchmarks: `polybench/2DCONV`,
  `polybench/MVT`. Do `git diff @~10 @ polybench/2DCONV`.
- Also see `dev/README.md`, `perf_logs/README.md`, all the `-h` flags.
