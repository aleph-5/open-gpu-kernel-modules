# UVM Dirty Tracking Tests

## Build

```bash
cd tests/
make                        # builds both binaries
make test_dirty_tracking_suite   # build suite only
```

## Run

```bash
sudo ./test_dirty_tracking_suite
```

---

## Note

- We need to define the parametere locations here once we have created them

```c
#define TRACKED_PID_PARAM        "/sys/module/nvidia_uvm/parameters/TODO_tracked_pid"
#define TRACKED_ADDR_RANGE_PARAM "/sys/module/nvidia_uvm/parameters/TODO_tracked_addr_range"
```

Until those files exist on the filesystem, GROUP B tests should be skipped

---

## Addr-range param format

The kernel param should accept a single string in the form:

```
0xSTART-0xEND
```

where `START` is inclusive and `END` is exclusive. Example:

```bash
echo "0x7f3400000000-0x7f3400010000" \
  | sudo tee /sys/module/nvidia_uvm/parameters/TODO_tracked_addr_range
```

**Future work** — use a fixed-size sorted array of
`(start, end)` intervals in the kernel:

```c
#define UVM_MAX_TRACKED_RANGES  16

struct uvm_addr_range {
    NvU64 start;   /* inclusive, page-aligned */
    NvU64 end;     /* exclusive, page-aligned */
};

static struct uvm_addr_range uvm_tracked_ranges[UVM_MAX_TRACKED_RANGES];
static int                   uvm_num_tracked_ranges = 0;
static spinlock_t            uvm_tracked_ranges_lock;
```

- v1 (naive): only `ranges[0]` is used; the param handler sets
  `ranges[0] = {start, end}`.
- Future multi-range: parse comma-separated `"0xS1-0xE1,0xS2-0xE2"`, insert
  into sorted array, merge overlaps.
- Future removal: add a companion param `tracked_addr_range_del` accepting the
  same `"0xSTART-0xEND"` format; handler finds and splits/removes matching
  intervals.

---

## Test cases

### GROUP A — works with current implementation

These tests use only `uvm_dirty_tracking` (module param) and
`/proc/driver/nvidia-uvm/dirty_pages` (procfs).

---

#### T01 · `writes_recorded`

| Field | Detail |
|-------|--------|
| **What** | Enable tracking, GPU-write all `NUM_PAGES` pages, read procfs. |
| **Expected** | All `NUM_PAGES` page-aligned addresses appear in `dirty_pages`. |
| **Why** | `uvm_dirty_page_table_record()` is called in the write-fault path in `uvm_gpu_replayable_faults.c`. |

---

#### T02 · `reads_not_recorded`

| Field | Detail |
|-------|--------|
| **What** | Enable tracking, run a GPU read-only kernel (sink is `cudaMalloc`, not managed), read procfs. |
| **Expected** | None of the managed page addresses appear in `dirty_pages`. |
| **Why** | Reads produce READ_ONLY mappings via `compute_new_permission`; `uvm_dirty_page_table_record()` is never called for them. |

---

#### T03 · `reset_clears_table`

| Field | Detail |
|-------|--------|
| **What** | Write first half of pages → reset table (write `"1"` again to param) → write second half → read procfs. |
| **Expected** | Only second-half addresses present; first-half absent. |
| **Why** | Writing `"1"` when already `1` triggers `destroy + init` of the xarray in `uvm_dirty_tracking_set()`. |

---

#### T04 · `mixed_read_write`

| Field | Detail |
|-------|--------|
| **What** | Read first half of pages (read kernel + device-only sink), write second half (write kernel), check procfs. |
| **Expected** | Only second-half addresses in `dirty_pages`; first-half absent. |
| **Why** | Validates that the access-type distinction in the fault path correctly separates reads from writes. |

---

#### T05 · `migration_write_retracked`

| Field | Detail |
|-------|--------|
| **What** | Write all pages → reset table → migrate to CPU (evicts GPU mappings) → write again → check procfs. |
| **Expected** | All `NUM_PAGES` addresses appear after the second write pass. |
| **Why** | After CPU migration, GPU mappings are removed. The next GPU write re-faults each page, re-triggering the record call. |

---

#### T06 · `tracking_disabled`

| Field | Detail |
|-------|--------|
| **What** | Disable tracking (`"0"` to param), write all pages, read procfs. |
| **Expected** | `read_procfs()` returns `-2` (`"dirty tracking not active"` header). |
| **Why** | When `uvm_dirty_tracking == 0`, `page_table_pointer == NULL`; the procfs handler prints the not-active comment. |

---

### GROUP B — speculative

#### T07 · `pid_filter_self`

| Field | Detail |
|-------|--------|
| **What** | Set `tracked_pid = getpid()`, write all pages, check procfs. |
| **Expected** | All `NUM_PAGES` pages appear — own PID must be matched. |
| **Why** | The PID filter should accept faults from the process that set the filter. |

---

#### T08 · `pid_filter_other`

| Field | Detail |
|-------|--------|
| **What** | Set `tracked_pid = 1` (init/systemd, never matches), write all pages, check procfs. |
| **Expected** | Zero managed pages in `dirty_pages`. |
| **Why** | Faults from a different PID must be silently skipped by the record call. |

---

#### T09 · `range_inside`

| Field | Detail |
|-------|--------|
| **What** | Set `addr_range` to exactly cover the managed allocation, write all pages. |
| **Expected** | All `NUM_PAGES` pages appear. |
| **Why** | Writes entirely inside the range must be accepted. |

---

#### T10 · `range_outside`

| Field | Detail |
|-------|--------|
| **What** | Set `addr_range` to `[base + N*PAGE, base + 2N*PAGE)` (no overlap with allocation), write all managed pages. |
| **Expected** | Zero managed pages in `dirty_pages`. |
| **Why** | Writes outside the range must be ignored. |

---

#### T11 · `range_partial_coverage`

| Field | Detail |
|-------|--------|
| **What** | Set `addr_range` to cover only the first half, write ALL pages. |
| **Expected** | First-half pages present, second-half absent. |
| **Why** | Sub-allocation granularity: filter at the page level, not the block level. |

---

#### T12 · `range_boundary_pages`

| Field | Detail |
|-------|--------|
| **What** | Set `addr_range = [base+1*PAGE, base+(N-1)*PAGE)`, write all pages. Check p=0 absent, p=1..N-2 present, p=N-1 absent. |
| **Expected** | Boundary semantics: start inclusive, end exclusive. |
| **Why** | Off-by-one at both boundaries is the most common filter bug. |

---

## Interpreting results

```
=== Results: 6/12 passed, 0 failed, 6 skipped ===
```

Once address range and pid functionality is implemented: all 12 should be PASS.
