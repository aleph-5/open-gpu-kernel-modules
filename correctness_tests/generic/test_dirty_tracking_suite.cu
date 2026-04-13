/*
 * test_dirty_tracking_suite.cu - generic dirty tracking suite
 *
 * Validates core dirty tracking semantics against the procfs interface:
 *   dirty_tracking_start  "<pid> <delta|cumulative>"
 *   dirty_tracking_stop   "<pid>"
 *   dirty_tracking_query_cutover  "<pid>"
 *   dirty_tracking_query_dump     (read)
 *
 * Key semantics tested:
 *   - Writes are tracked; reads are not.
 *   - Cutover atomically snapshots the current epoch; dump consumes it.
 *   - Cutover does NOT re-invalidate PTEs: writing the same page again
 *     after cutover does not produce a new entry.
 *   - stop() fully resets state and re-invalidates PTEs; a fresh start()
 *     re-enables tracking from a clean slate.
 *   - invalidate (cudaFree) does NOT remove entries from the tracker.
 *   - Cumulative mode accumulates entries across epochs; delta mode does not.
 */

#include <cuda_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

#define NUM_PAGES     8
#define PAGE_SIZE     4096
#define INTS_PER_PAGE (PAGE_SIZE / sizeof(int))
#define NUM_INTS      (NUM_PAGES * INTS_PER_PAGE)
#define MAX_ENTRIES   4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef enum { PASS = 0, FAIL = 1, SKIP = 2 } result_t;

typedef struct { unsigned long addr, ts; } entry_t;

typedef struct {
    result_t result;
    char expected[256];
    char actual[256];
} outcome_t;

static int g_pass = 0, g_fail = 0, g_skip = 0;

/* ---- GPU kernels ---- */

__global__ void gpu_write(int *data, int n, int tag)
{
    for (int i = 0; i < n; ++i) data[i] = tag * 10000 + i;
}

__global__ void gpu_write_range(int *base, int start_page, int end_page, int tag)
{
    for (int p = start_page; p < end_page; ++p)
        for (int i = 0; i < INTS_PER_PAGE; ++i)
            base[p * INTS_PER_PAGE + i] = tag * 10000 + p * 100 + i;
}

__global__ void gpu_read(const int *data, int n, volatile int *sink)
{
    int acc = 0;
    for (int i = 0; i < n; ++i) acc += data[i];
    *sink = acc;
}

/* ---- procfs helpers ---- */

static void procfs_write(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    ssize_t w = write(fd, val, strlen(val));
    if (w < 0) { perror("write"); close(fd); exit(1); }
    close(fd);
}

static void start_tracking(const char *mode)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d %s", getpid(), mode);
    procfs_write(PROCFS_START, buf);
}

static void stop_tracking(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    procfs_write(PROCFS_STOP, buf);
}

static void do_cutover(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    procfs_write(PROCFS_CUTOVER, buf);
}

/* Returns entry count on success, -errno if dump unavailable. */
static int read_dump(entry_t *out, int max)
{
    FILE *f = fopen(PROCFS_DUMP, "r");
    if (!f) return -errno;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (n < max && sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2)
            n++;
    }
    fclose(f);
    return n;
}

/* ---- entry counting helpers ---- */

static int entries_in_range(entry_t *e, int n, unsigned long base, int npages)
{
    int found = 0;
    for (int p = 0; p < npages; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

/* ---- test reporting ---- */

static void report(const char *id, const char *name, outcome_t *o)
{
    const char *s = o->result == PASS ? "PASS" : o->result == SKIP ? "SKIP" : "FAIL";
    printf("  [%s] %s - %s\n", s, id, name);
    if (o->expected[0]) printf("         expected: %s\n", o->expected);
    if (o->actual[0])   printf("         actual:   %s\n", o->actual);
    if      (o->result == PASS) g_pass++;
    else if (o->result == FAIL) g_fail++;
    else                        g_skip++;
}

/* ================================================================
 * T01: GPU writes are recorded
 * ================================================================ */
static void t01_writes_recorded(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected), "all %d pages in dump after write", NUM_PAGES);

    start_tracking("delta");
    gpu_write<<<1, 1>>>(managed, NUM_INTS, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    int found = n >= 0 ? entries_in_range(e, n, base, NUM_PAGES) : 0;

    if (n < 0 || found != NUM_PAGES)
        snprintf(o.actual, sizeof(o.actual), "n=%d found=%d/%d", n, found, NUM_PAGES),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "all %d pages present", found);

    stop_tracking();
    report("T01", "writes_recorded", &o);
}

/* ================================================================
 * T02: GPU reads are NOT recorded
 * ================================================================ */
static void t02_reads_not_recorded(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected), "0 pages in dump after read-only access");

    int *sink;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));

    start_tracking("delta");
    gpu_read<<<1, 1>>>(managed, NUM_INTS, (volatile int *)sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    int spurious = n >= 0 ? entries_in_range(e, n, base, NUM_PAGES) : NUM_PAGES;

    if (n < 0 || spurious != 0)
        snprintf(o.actual, sizeof(o.actual), "n=%d spurious=%d", n, spurious),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "0 pages tracked (total=%d)", n);

    stop_tracking();
    CUDA_CHECK(cudaFree(sink));
    report("T02", "reads_not_recorded", &o);
}

/* ================================================================
 * T03: stop+start resets tracking; only post-restart writes appear
 * ================================================================ */
static void t03_stop_start_resets(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    const int half = NUM_PAGES / 2;
    snprintf(o.expected, sizeof(o.expected),
             "pages 0..%d absent, pages %d..%d present after stop+start",
             half - 1, half, NUM_PAGES - 1);

    /* Epoch 1: write first half. */
    start_tracking("delta");
    gpu_write_range<<<1, 1>>>(managed, 0, half, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* stop re-invalidates PTEs; start gives a fresh table. */
    stop_tracking();
    start_tracking("delta");

    /* Epoch 2: write second half only. */
    gpu_write_range<<<1, 1>>>(managed, half, NUM_PAGES, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    int ghost = n >= 0 ? entries_in_range(e, n, base, half) : half;
    int present = n >= 0 ? entries_in_range(e, n, base + (unsigned long)half * PAGE_SIZE,
                                             NUM_PAGES - half) : 0;

    if (n < 0 || ghost != 0 || present != (NUM_PAGES - half))
        snprintf(o.actual, sizeof(o.actual), "n=%d ghost=%d present=%d/%d",
                 n, ghost, present, NUM_PAGES - half),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "reset correct (total=%d)", n);

    stop_tracking();
    report("T03", "stop_start_resets", &o);
}

/* ================================================================
 * T04: mixed read+write — only written pages appear
 * ================================================================ */
static void t04_mixed_read_write(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    const int half = NUM_PAGES / 2;
    snprintf(o.expected, sizeof(o.expected), "read half clean, write half tracked");

    int *sink;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));

    start_tracking("delta");
    gpu_read<<<1, 1>>>(managed, half * INTS_PER_PAGE, (volatile int *)sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_write_range<<<1, 1>>>(managed, half, NUM_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    int spurious = n >= 0 ? entries_in_range(e, n, base, half) : half;
    int present  = n >= 0 ? entries_in_range(e, n, base + (unsigned long)half * PAGE_SIZE,
                                              NUM_PAGES - half) : 0;

    if (n < 0 || spurious != 0 || present != (NUM_PAGES - half))
        snprintf(o.actual, sizeof(o.actual), "n=%d spurious=%d write_present=%d/%d",
                 n, spurious, present, NUM_PAGES - half),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "read/write split correct (total=%d)", n);

    stop_tracking();
    CUDA_CHECK(cudaFree(sink));
    report("T04", "mixed_read_write", &o);
}

/* ================================================================
 * T05: dump returns error when no tracking session is active
 * ================================================================ */
static void t05_dump_when_inactive(void)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected), "dump returns error with no active session");

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);

    if (n > 0)
        snprintf(o.actual, sizeof(o.actual), "got %d entries (expected error)", n),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "error returned as expected (n=%d)", n);

    report("T05", "dump_when_inactive", &o);
}

/* ================================================================
 * T06: first-write-wins — writing same pages twice yields no duplicates
 * ================================================================ */
static void t06_first_write_wins_no_duplicates(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected),
             "exactly %d entries after writing same pages twice in one epoch", NUM_PAGES);

    start_tracking("delta");
    gpu_write<<<1, 1>>>(managed, NUM_INTS, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    /* Second write to same pages: no re-invalidation, no new faults, no new entries. */
    gpu_write<<<1, 1>>>(managed, NUM_INTS, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    int found = n >= 0 ? entries_in_range(e, n, base, NUM_PAGES) : -1;

    if (n < 0 || found != NUM_PAGES)
        snprintf(o.actual, sizeof(o.actual), "n=%d found=%d (want %d)", n, found, NUM_PAGES),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "no duplicates (%d entries)", n);

    stop_tracking();
    report("T06", "first_write_wins_no_duplicates", &o);
}

/* ================================================================
 * T07: cutover does NOT re-invalidate PTEs — same page written after
 *      cutover must NOT appear in the next epoch's dump
 * ================================================================ */
static void t07_cutover_no_reinvalidate(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected),
             "epoch1: %d pages; epoch2: 0 pages (no re-invalidation at cutover)", NUM_PAGES);

    start_tracking("delta");
    gpu_write<<<1, 1>>>(managed, NUM_INTS, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e1[MAX_ENTRIES];
    int n1 = read_dump(e1, MAX_ENTRIES);
    int found1 = n1 >= 0 ? entries_in_range(e1, n1, base, NUM_PAGES) : 0;

    /* PTEs still valid after cutover — second write will not fault. */
    gpu_write<<<1, 1>>>(managed, NUM_INTS, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e2[MAX_ENTRIES];
    int n2 = read_dump(e2, MAX_ENTRIES);
    /* n2 < 0 (no snapshot / empty) or 0 entries both mean "nothing tracked". */
    int found2 = n2 > 0 ? entries_in_range(e2, n2, base, NUM_PAGES) : 0;

    if (found1 != NUM_PAGES || found2 != 0)
        snprintf(o.actual, sizeof(o.actual), "epoch1=%d/%d epoch2=%d (want 0)",
                 found1, NUM_PAGES, found2),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "epoch1=%d epoch2=0 correct", found1);

    stop_tracking();
    report("T07", "cutover_no_reinvalidate", &o);
}

/* ================================================================
 * T08: fresh allocation after cutover is tracked in the next epoch
 * ================================================================ */
static void t08_fresh_alloc_after_cutover(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected),
             "epoch1: alloc_a pages present; epoch2: alloc_b pages present, alloc_a absent");

    /* Epoch 1: write the fixture allocation. */
    start_tracking("delta");
    gpu_write<<<1, 1>>>(managed, NUM_INTS, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e1[MAX_ENTRIES];
    int n1 = read_dump(e1, MAX_ENTRIES);
    int a_epoch1 = n1 >= 0 ? entries_in_range(e1, n1, base, NUM_PAGES) : 0;

    /* Epoch 2: fresh allocation — its PTEs have never been set up. */
    int *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_b, NUM_PAGES * PAGE_SIZE));
    unsigned long base_b = (unsigned long)alloc_b;

    gpu_write<<<1, 1>>>(alloc_b, NUM_INTS, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e2[MAX_ENTRIES];
    int n2 = read_dump(e2, MAX_ENTRIES);
    int a_epoch2 = n2 >= 0 ? entries_in_range(e2, n2, base, NUM_PAGES) : -1;
    int b_epoch2 = n2 >= 0 ? entries_in_range(e2, n2, base_b, NUM_PAGES) : 0;

    CUDA_CHECK(cudaFree(alloc_b));

    if (a_epoch1 != NUM_PAGES || a_epoch2 != 0 || b_epoch2 != NUM_PAGES)
        snprintf(o.actual, sizeof(o.actual),
                 "a_epoch1=%d/%d a_epoch2=%d(want 0) b_epoch2=%d/%d",
                 a_epoch1, NUM_PAGES, a_epoch2, b_epoch2, NUM_PAGES),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "epoch isolation and fresh-fault correct");

    stop_tracking();
    report("T08", "fresh_alloc_after_cutover", &o);
}

/* ================================================================
 * T09: empty epoch produces empty (or error) dump
 * ================================================================ */
static void t09_empty_epoch(unsigned long base, int *managed)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected),
             "epoch1: %d pages; epoch2 (no write): 0 pages", NUM_PAGES);

    start_tracking("delta");
    gpu_write<<<1, 1>>>(managed, NUM_INTS, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e1[MAX_ENTRIES];
    int n1 = read_dump(e1, MAX_ENTRIES);
    int found1 = n1 >= 0 ? entries_in_range(e1, n1, base, NUM_PAGES) : 0;

    /* No write in epoch 2. */
    do_cutover();

    entry_t e2[MAX_ENTRIES];
    int n2 = read_dump(e2, MAX_ENTRIES);
    int found2 = n2 > 0 ? entries_in_range(e2, n2, base, NUM_PAGES) : 0;

    if (found1 != NUM_PAGES || found2 != 0)
        snprintf(o.actual, sizeof(o.actual), "epoch1=%d/%d epoch2=%d (want 0)",
                 found1, NUM_PAGES, found2),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "empty epoch correct");

    stop_tracking();
    report("T09", "empty_epoch", &o);
}

/* ================================================================
 * T10: cumulative mode retains entries across epochs
 * ================================================================ */
static void t10_cumulative_retains_entries(void)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected),
             "cumulative dump after epoch2 contains both epoch1 and epoch2 pages");

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, NUM_PAGES * PAGE_SIZE));
    unsigned long base_a = (unsigned long)alloc_a;
    unsigned long base_b = (unsigned long)alloc_b;

    start_tracking("cumulative");

    /* Epoch 1: write alloc_a. */
    gpu_write<<<1, 1>>>(alloc_a, NUM_INTS, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e1[MAX_ENTRIES];
    int n1 = read_dump(e1, MAX_ENTRIES);
    int a_epoch1 = n1 >= 0 ? entries_in_range(e1, n1, base_a, NUM_PAGES) : 0;

    /* Epoch 2: write alloc_b (fresh, will fault). */
    gpu_write<<<1, 1>>>(alloc_b, NUM_INTS, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();

    entry_t e2[MAX_ENTRIES];
    int n2 = read_dump(e2, MAX_ENTRIES);
    int a_epoch2 = n2 >= 0 ? entries_in_range(e2, n2, base_a, NUM_PAGES) : 0;
    int b_epoch2 = n2 >= 0 ? entries_in_range(e2, n2, base_b, NUM_PAGES) : 0;

    CUDA_CHECK(cudaFree(alloc_a));
    CUDA_CHECK(cudaFree(alloc_b));

    if (a_epoch1 != NUM_PAGES || a_epoch2 != NUM_PAGES || b_epoch2 != NUM_PAGES)
        snprintf(o.actual, sizeof(o.actual),
                 "a_epoch1=%d/%d a_epoch2=%d/%d b_epoch2=%d/%d",
                 a_epoch1, NUM_PAGES, a_epoch2, NUM_PAGES, b_epoch2, NUM_PAGES),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "cumulative retention correct");

    stop_tracking();
    report("T10", "cumulative_retains_entries", &o);
}

/* ================================================================
 * T11: invalidate (cudaFree) does NOT remove entries from the tracker
 * ================================================================ */
static void t11_invalidate_persists_entry(void)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected),
             "freed alloc's pages still present in dump after cudaFree");

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, NUM_PAGES * PAGE_SIZE));
    unsigned long base_a = (unsigned long)alloc_a;
    unsigned long base_b = (unsigned long)alloc_b;

    start_tracking("delta");
    gpu_write<<<1, 1>>>(alloc_a, NUM_INTS, 1);
    gpu_write<<<1, 1>>>(alloc_b, NUM_INTS, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Free alloc_a — invalidate must NOT remove its entries. */
    CUDA_CHECK(cudaFree(alloc_a));
    alloc_a = NULL;

    do_cutover();
    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);

    int a_found = n >= 0 ? entries_in_range(e, n, base_a, NUM_PAGES) : 0;
    int b_found = n >= 0 ? entries_in_range(e, n, base_b, NUM_PAGES) : 0;

    CUDA_CHECK(cudaFree(alloc_b));

    if (a_found != NUM_PAGES || b_found != NUM_PAGES)
        snprintf(o.actual, sizeof(o.actual),
                 "n=%d a_found=%d/%d b_found=%d/%d (a freed before dump)",
                 n, a_found, NUM_PAGES, b_found, NUM_PAGES),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "both allocs present after free (%d entries)", n);

    stop_tracking();
    report("T11", "invalidate_persists_entry", &o);
}

/* ================================================================
 * T12: delta mode — epoch isolation across three consecutive epochs
 *      write A / empty / write B — each dump reflects only that epoch
 * ================================================================ */
static void t12_delta_epoch_isolation(void)
{
    outcome_t o = { PASS, "", "" };
    snprintf(o.expected, sizeof(o.expected),
             "epoch1: A only; epoch2: empty; epoch3: B only");

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, NUM_PAGES * PAGE_SIZE));
    unsigned long base_a = (unsigned long)alloc_a;
    unsigned long base_b = (unsigned long)alloc_b;

    start_tracking("delta");

    /* Epoch 1: write A. */
    gpu_write<<<1, 1>>>(alloc_a, NUM_INTS, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();
    entry_t e1[MAX_ENTRIES];
    int n1 = read_dump(e1, MAX_ENTRIES);
    int a1 = n1 >= 0 ? entries_in_range(e1, n1, base_a, NUM_PAGES) : 0;
    int b1 = n1 >= 0 ? entries_in_range(e1, n1, base_b, NUM_PAGES) : -1;

    /* Epoch 2: no write. */
    do_cutover();
    entry_t e2[MAX_ENTRIES];
    int n2 = read_dump(e2, MAX_ENTRIES);
    int total2 = n2 > 0 ? n2 : 0;

    /* Epoch 3: write B (fresh alloc, always faults). */
    gpu_write<<<1, 1>>>(alloc_b, NUM_INTS, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    do_cutover();
    entry_t e3[MAX_ENTRIES];
    int n3 = read_dump(e3, MAX_ENTRIES);
    int a3 = n3 >= 0 ? entries_in_range(e3, n3, base_a, NUM_PAGES) : -1;
    int b3 = n3 >= 0 ? entries_in_range(e3, n3, base_b, NUM_PAGES) : 0;

    CUDA_CHECK(cudaFree(alloc_a));
    CUDA_CHECK(cudaFree(alloc_b));

    if (a1 != NUM_PAGES || b1 != 0 || total2 != 0 || a3 != 0 || b3 != NUM_PAGES)
        snprintf(o.actual, sizeof(o.actual),
                 "e1(a=%d b=%d) e2(total=%d) e3(a=%d b=%d) want e1(a=%d b=0) e2(0) e3(a=0 b=%d)",
                 a1, b1, total2, a3, b3, NUM_PAGES, NUM_PAGES),
        o.result = FAIL;
    else
        snprintf(o.actual, sizeof(o.actual), "three-epoch isolation correct");

    stop_tracking();
    report("T12", "delta_epoch_isolation", &o);
}

/* ================================================================ */

int main(void)
{
    printf("=== UVM Dirty Tracking Test Suite ===\n\n");

    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }

    const char *required[] = { PROCFS_START, PROCFS_STOP, PROCFS_CUTOVER, PROCFS_DUMP };
    for (int i = 0; i < 4; i++) {
        struct stat st;
        if (stat(required[i], &st) != 0) {
            fprintf(stderr, "ERROR: procfs entry missing: %s\n", required[i]);
            return 1;
        }
    }

    /* Shared fixture: pre-faulted allocation used by most tests. */
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long base = (unsigned long)managed;

    printf("  pid=%d managed=0x%lx (%d pages)\n\n", getpid(), base, NUM_PAGES);

    printf("--- GROUP A: core write/read semantics ---\n");
    t01_writes_recorded(base, managed);
    t02_reads_not_recorded(base, managed);
    t03_stop_start_resets(base, managed);
    t04_mixed_read_write(base, managed);
    t05_dump_when_inactive();

    printf("\n--- GROUP B: epoch and cutover semantics ---\n");
    t06_first_write_wins_no_duplicates(base, managed);
    t07_cutover_no_reinvalidate(base, managed);
    t08_fresh_alloc_after_cutover(base, managed);
    t09_empty_epoch(base, managed);

    printf("\n--- GROUP C: mode and lifecycle semantics ---\n");
    t10_cumulative_retains_entries();
    t11_invalidate_persists_entry();
    t12_delta_epoch_isolation();

    CUDA_CHECK(cudaFree(managed));

    int total = g_pass + g_fail + g_skip;
    printf("\n=== Results: %d/%d passed, %d failed, %d skipped ===\n",
           g_pass, total, g_fail, g_skip);

    return g_fail == 0 ? 0 : 1;
}
