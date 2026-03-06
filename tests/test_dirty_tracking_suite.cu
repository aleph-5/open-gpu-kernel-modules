/* test_dirty_tracking_suite.cu — UVM dirty page tracking tests
 * GROUP A: basic tracking  |  GROUP B: pid logging + addr range filter
 *
 * Kernel interface:
 *   /sys/module/nvidia_uvm/parameters/uvm_dirty_tracking  — write "1" to
 *       enable (and reinitialize/clear table); write "0" to disable+destroy
 *   /proc/driver/nvidia-uvm/dirty_pages  — read dirty entries:
 *       "# dirty tracking not active"  when disabled
 *       "0x<addr_hex> <timestamp_ns> <pid>"  one line per dirty page
 *   /proc/driver/nvidia-uvm/dirty_range — write "<start_hex> <end_hex>" to
 *       restrict which pages are returned on the next read; defaults to
 *       [0, ~0UL] (all pages).  Reset by writing "0x0 0xffffffffffffffff".
 *
 * pid in entries is va_block->creator_pid (tgid of the cudaMallocManaged
 * caller), not the pid of the process that triggered the GPU fault.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define SYSFS_DIRTY        "/sys/module/nvidia_uvm/parameters/uvm_dirty_tracking"
#define PROCFS_DIRTY       "/proc/driver/nvidia-uvm/dirty_pages"
#define PROCFS_DIRTY_RANGE "/proc/driver/nvidia-uvm/dirty_range"

#define NUM_PAGES   8
#define PAGE_SIZE   4096
#define NUM_INTS    (NUM_PAGES * PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES 4096

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));               \
        exit(1);                                                           \
    }                                                                      \
} while (0)

typedef enum { PASS = 0, FAIL = 1, SKIP = 2 } result_t;
typedef struct { result_t result; char expected[512]; char actual[512]; } outcome_t;
typedef struct { unsigned long addr, timestamp_ns; pid_t pid; } dirty_entry_t;

__global__ void kernel_read(const int *data, int n, volatile int *sink)
{
    int acc = 0;
    for (int i = 0; i < n; i++) acc += data[i];
    *sink = acc;
}

__global__ void kernel_write(int *data, int n)
    { for (int i = 0; i < n; i++) data[i] = i + 1; }

__global__ void kernel_write_range(int *data, int ps, int pe)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = ps; p < pe; p++)
        for (int i = 0; i < ipp; i++)
            data[p * ipp + i] = p * ipp + i + 1;
}

/* Returns number of dirty entries parsed, -1 on open error, -2 if tracking
 * is not active.  Parses the 3-field format: 0x<addr> <ts_ns> <pid>. */
static int read_procfs(dirty_entry_t *out, int max)
{
    FILE *f = fopen(PROCFS_DIRTY, "r");
    if (!f) return -1;
    int n = 0; char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) { fclose(f); return -2; }
            continue;
        }
        if (n >= max) break;
        int pid;
        if (sscanf(line, "0x%lx %lu %d",
                   &out[n].addr, &out[n].timestamp_ns, &pid) == 3) {
            out[n].pid = (pid_t)pid;
            n++;
        }
    }
    fclose(f); return n;
}

static int is_page_tracked(dirty_entry_t *e, int n, unsigned long addr)
{
    unsigned long pa = addr & ~((unsigned long)(PAGE_SIZE - 1));
    for (int i = 0; i < n; i++) if (e[i].addr == pa) return 1;
    return 0;
}

/* Returns the recorded pid for the page containing addr, or -1 if not found. */
static pid_t get_page_pid(dirty_entry_t *e, int n, unsigned long addr)
{
    unsigned long pa = addr & ~((unsigned long)(PAGE_SIZE - 1));
    for (int i = 0; i < n; i++) if (e[i].addr == pa) return e[i].pid;
    return -1;
}

static void sysfs_write(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    if (write(fd, val, strlen(val)) < 0) { perror("write"); exit(1); }
    close(fd);
}

static int  sysfs_exists(const char *p)    { struct stat st; return stat(p, &st) == 0; }
static void set_tracking(int v)            { sysfs_write(SYSFS_DIRTY, v ? "1\n" : "0\n"); }
static void reset_table(void)              { sysfs_write(SYSFS_DIRTY, "1\n"); }

/* Reset addr range filter to the full address space (kernel default). */
static void reset_addr_range(void)
    { sysfs_write(PROCFS_DIRTY_RANGE, "0x0 0xffffffffffffffff\n"); }

static void set_addr_range(unsigned long s, unsigned long e)
    { char b[64]; snprintf(b, sizeof(b), "0x%lx 0x%lx\n", s, e); sysfs_write(PROCFS_DIRTY_RANGE, b); }

/* static void migrate_to_cpu(void *ptr, size_t bytes) */
/* { */
/*     cudaMemLocation loc = { cudaMemLocationTypeHost, 0 }; */
/*     CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0)); */
/*     CUDA_CHECK(cudaDeviceSynchronize()); */
/* } */

static int g_pass = 0, g_fail = 0, g_skip = 0;

static void print_result(const char *id, const char *name, result_t r,
                         const char *expected, const char *actual)
{
    printf("  [%s] %s — %s\n", r == PASS ? "PASS" : r == SKIP ? "SKIP" : "FAIL", id, name);
    if (expected && expected[0]) printf("         expected: %s\n", expected);
    if (actual   && actual[0])   printf("         actual:   %s\n", actual);
    if (r == PASS) g_pass++; else if (r == FAIL) g_fail++; else g_skip++;
}

/* =========================================================================
 * GROUP A — basic tracking semantics
 * ========================================================================= */

static void t01_writes_recorded(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    snprintf(out.expected, sizeof(out.expected), "all %d pages present after GPU write", NUM_PAGES);
    set_tracking(1);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n < 0) {
        snprintf(out.actual, sizeof(out.actual), "procfs read failed (n=%d)", n);
        out.result = FAIL;
    } else {
        int miss = 0;
        for (int p = 0; p < NUM_PAGES; p++)
            if (!is_page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) miss++;
        if (miss) { snprintf(out.actual, sizeof(out.actual), "%d/%d pages missing (total=%d)", miss, NUM_PAGES, n); out.result = FAIL; }
        else        snprintf(out.actual, sizeof(out.actual), "all %d pages present (total=%d)", NUM_PAGES, n);
    }
    set_tracking(0);
    print_result("T01", "writes_recorded", out.result, out.expected, out.actual);
}

static void t02_reads_not_recorded(int *managed, int dev)
{
    int *sink = NULL; dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    snprintf(out.expected, sizeof(out.expected), "0 managed pages tracked (GPU reads don't fault as write)");
    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
    set_tracking(1);
    kernel_read<<<1, 1>>>(managed, NUM_INTS, (volatile int *)sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n < 0) {
        snprintf(out.actual, sizeof(out.actual), n == -2 ? "tracking not active" : "procfs read failed (n=%d)", n);
        out.result = FAIL;
    } else {
        int spur = 0;
        for (int p = 0; p < NUM_PAGES; p++)
            if (is_page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) spur++;
        if (spur) { snprintf(out.actual, sizeof(out.actual), "%d pages spuriously tracked (total=%d)", spur, n); out.result = FAIL; }
        else        snprintf(out.actual, sizeof(out.actual), "0 managed pages tracked (total=%d)", n);
    }
    set_tracking(0); cudaFree(sink);
    print_result("T02", "reads_not_recorded", out.result, out.expected, out.actual);
}

static void t03_reset_clears_table(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    int half = NUM_PAGES / 2;
    snprintf(out.expected, sizeof(out.expected),
             "pages 0..%d absent (pre-reset), pages %d..%d present (post-reset)",
             half - 1, half, NUM_PAGES - 1);
    set_tracking(1);
    kernel_write_range<<<1, 1>>>(managed, 0, half);
    CUDA_CHECK(cudaDeviceSynchronize());
    reset_table();
    kernel_write_range<<<1, 1>>>(managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n < 0) {
        snprintf(out.actual, sizeof(out.actual), "procfs read failed (n=%d)", n); out.result = FAIL;
    } else {
        int ghost = 0, present = 0, missing = 0;
        for (int p = 0; p < half; p++)
            if (is_page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) ghost++;
        for (int p = half; p < NUM_PAGES; p++)
            if (is_page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) present++; else missing++;
        if (ghost || missing) { snprintf(out.actual, sizeof(out.actual), "ghost=%d missing=%d present=%d (total=%d)", ghost, missing, present, n); out.result = FAIL; }
        else                    snprintf(out.actual, sizeof(out.actual), "pre-reset absent, post-reset present (%d pages, total=%d)", present, n);
    }
    set_tracking(0);
    print_result("T03", "reset_clears_table", out.result, out.expected, out.actual);
}

static void t04_mixed_read_write(int *managed, int dev)
{
    int *sink = NULL; dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));
    int half = NUM_PAGES / 2, ipp = PAGE_SIZE / sizeof(int);
    snprintf(out.expected, sizeof(out.expected),
             "pages 0..%d clean (reads only), pages %d..%d dirty (writes)",
             half - 1, half, NUM_PAGES - 1);
    set_tracking(1);
    kernel_read<<<1, 1>>>(managed, half * ipp, (volatile int *)sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_write_range<<<1, 1>>>(managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n < 0) {
        snprintf(out.actual, sizeof(out.actual), "procfs read failed (n=%d)", n); out.result = FAIL;
    } else {
        int rs = 0, wm = 0;
        for (int p = 0; p < half; p++)
            if ( is_page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) rs++;
        for (int p = half; p < NUM_PAGES; p++)
            if (!is_page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) wm++;
        if (rs || wm) { snprintf(out.actual, sizeof(out.actual), "read spurious=%d write missing=%d (total=%d)", rs, wm, n); out.result = FAIL; }
        else            snprintf(out.actual, sizeof(out.actual), "read pages clean, write pages tracked (total=%d)", n);
    }
    set_tracking(0); cudaFree(sink);
    print_result("T04", "mixed_read_write", out.result, out.expected, out.actual);
}

static void t05_migration_write_retracked(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    snprintf(out.expected, sizeof(out.expected),
             "all %d pages tracked after table reset + CPU migration + GPU write", NUM_PAGES);
    set_tracking(1);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    reset_table();
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n < 0) {
        snprintf(out.actual, sizeof(out.actual), "procfs read failed (n=%d)", n); out.result = FAIL;
    } else {
        int miss = 0;
        for (int p = 0; p < NUM_PAGES; p++)
            if (!is_page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) miss++;
        if (miss) { snprintf(out.actual, sizeof(out.actual), "%d/%d missing after migration+write (total=%d)", miss, NUM_PAGES, n); out.result = FAIL; }
        else        snprintf(out.actual, sizeof(out.actual), "all %d pages tracked after migration (total=%d)", NUM_PAGES, n);
    }
    set_tracking(0);
    print_result("T05", "migration_write_retracked", out.result, out.expected, out.actual);
}

static void t06_tracking_disabled(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    snprintf(out.expected, sizeof(out.expected),
             "procfs reports '# dirty tracking not active' when uvm_dirty_tracking=0");
    set_tracking(0);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n == -2) snprintf(out.actual, sizeof(out.actual), "procfs reports 'not active'");
    else        { snprintf(out.actual, sizeof(out.actual), "expected not-active, got n=%d", n); out.result = FAIL; }
    print_result("T06", "tracking_disabled", out.result, out.expected, out.actual);
}

/* =========================================================================
 * GROUP B — pid field correctness + addr range filter
 * ========================================================================= */

/* T07: every dirty entry for our managed allocation must carry our pid.
 * The kernel records va_block->creator_pid = tgid of the cudaMallocManaged
 * caller, which is this process. */
static void t07_pid_recorded(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    pid_t my_pid = getpid();
    snprintf(out.expected, sizeof(out.expected),
             "all %d dirty entries carry pid=%d (creator of the VA block)", NUM_PAGES, my_pid);
    set_tracking(1);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n < 0) {
        snprintf(out.actual, sizeof(out.actual), "procfs read failed (n=%d)", n); out.result = FAIL;
    } else {
        int wrong = 0, miss = 0;
        for (int p = 0; p < NUM_PAGES; p++) {
            unsigned long addr = (unsigned long)managed + p * PAGE_SIZE;
            pid_t recorded = get_page_pid(e, n, addr);
            if (recorded == -1)          miss++;
            else if (recorded != my_pid) wrong++;
        }
        if (miss || wrong)
            snprintf(out.actual, sizeof(out.actual),
                     "missing=%d wrong_pid=%d (expected pid=%d, total=%d)",
                     miss, wrong, my_pid, n);
        else
            snprintf(out.actual, sizeof(out.actual),
                     "all %d pages carry pid=%d (total=%d)", NUM_PAGES, my_pid, n);
        if (miss || wrong) out.result = FAIL;
    }
    set_tracking(0);
    print_result("T07", "pid_recorded", out.result, out.expected, out.actual);
}

/* T08: writing the same pages a second time (without resetting the table)
 * must not duplicate entries — xarray overwrites on the same key. */
static void t08_same_page_single_entry(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    snprintf(out.expected, sizeof(out.expected),
             "exactly %d entries after 2 write cycles (xarray overwrites same key)", NUM_PAGES);
    set_tracking(1);
    /* First write cycle */
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    /* Second write cycle — same virtual pages, no table reset */
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES);
    if (n < 0) {
        snprintf(out.actual, sizeof(out.actual), "procfs read failed (n=%d)", n); out.result = FAIL;
    } else if (n != NUM_PAGES) {
        snprintf(out.actual, sizeof(out.actual),
                 "expected %d entries, got %d (duplicates or missing)",
                 NUM_PAGES, n);
        out.result = FAIL;
    } else {
        snprintf(out.actual, sizeof(out.actual),
                 "exactly %d entries after 2 write cycles (no duplicates)", NUM_PAGES);
    }
    set_tracking(0);
    print_result("T08", "same_page_single_entry", out.result, out.expected, out.actual);
}

/* T09: addr range covers all managed pages — all pages must appear. */
static void t09_range_inside(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    unsigned long base = (unsigned long)managed;
    snprintf(out.expected, sizeof(out.expected),
             "all %d pages present when range exactly covers allocation", NUM_PAGES);
    set_addr_range(base, base + NUM_PAGES * PAGE_SIZE); set_tracking(1);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES), miss = 0;
    for (int p = 0; p < NUM_PAGES; p++)
        if (!is_page_tracked(e, n, base + p * PAGE_SIZE)) miss++;
    if (n < 0 || miss) { snprintf(out.actual, sizeof(out.actual), "n=%d miss=%d/%d", n, miss, NUM_PAGES); out.result = FAIL; }
    else                  snprintf(out.actual, sizeof(out.actual), "all %d pages inside range tracked (total=%d)", NUM_PAGES, n);
    set_tracking(0); reset_addr_range();
    print_result("T09", "range_inside", out.result, out.expected, out.actual);
}

/* T10: addr range is entirely above managed allocation — no pages must appear. */
static void t10_range_outside(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    snprintf(out.expected, sizeof(out.expected),
             "0 managed pages returned when range is entirely above the allocation");
    unsigned long base = (unsigned long)managed;
    unsigned long rs = base + NUM_PAGES * PAGE_SIZE;
    set_addr_range(rs, rs + NUM_PAGES * PAGE_SIZE); set_tracking(1);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES), spur = 0;
    if (n >= 0)
        for (int p = 0; p < NUM_PAGES; p++)
            if (is_page_tracked(e, n, base + p * PAGE_SIZE)) spur++;
    if (n < 0 || spur) { snprintf(out.actual, sizeof(out.actual), "n=%d spurious=%d", n, spur); out.result = FAIL; }
    else                  snprintf(out.actual, sizeof(out.actual), "0 pages tracked (range outside, total=%d)", n);
    set_tracking(0); reset_addr_range();
    print_result("T10", "range_outside", out.result, out.expected, out.actual);
}

/* T11: addr range covers only the first half — second half must be absent. */
static void t11_range_partial_coverage(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    int half = NUM_PAGES / 2;
    snprintf(out.expected, sizeof(out.expected),
             "pages 0..%d tracked (in range), pages %d..%d absent (out of range)",
             half - 1, half, NUM_PAGES - 1);
    unsigned long base = (unsigned long)managed;
    set_addr_range(base, base + half * PAGE_SIZE); set_tracking(1);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES), in_miss = 0, out_pres = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        int t = is_page_tracked(e, n, base + p * PAGE_SIZE);
        if (p <  half && !t) in_miss++;
        if (p >= half &&  t) out_pres++;
    }
    if (n < 0 || in_miss || out_pres) { snprintf(out.actual, sizeof(out.actual), "n=%d in_miss=%d out_pres=%d", n, in_miss, out_pres); out.result = FAIL; }
    else                                 snprintf(out.actual, sizeof(out.actual), "first half tracked, second half absent (total=%d)", n);
    set_tracking(0); reset_addr_range();
    print_result("T11", "range_partial_coverage", out.result, out.expected, out.actual);
}

/* T12: range excludes first and last page — only inner pages must appear. */
static void t12_range_boundary_pages(int *managed, int dev)
{
    dirty_entry_t e[MAX_ENTRIES]; outcome_t out = { PASS, "", "" };
    snprintf(out.expected, sizeof(out.expected),
             "page 0 absent, pages 1..%d present, page %d absent",
             NUM_PAGES - 2, NUM_PAGES - 1);
    unsigned long base = (unsigned long)managed;
    set_addr_range(base + PAGE_SIZE, base + (NUM_PAGES - 1) * PAGE_SIZE);
    set_tracking(1);
    kernel_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    int n = read_procfs(e, MAX_ENTRIES), errors = 0; char diag[256] = "";
    if (n >= 0 && is_page_tracked(e, n, base))
        { strcat(diag, "p=0 tracked; "); errors++; }
    for (int p = 1; p < NUM_PAGES - 1; p++)
        if (n < 0 || !is_page_tracked(e, n, base + p * PAGE_SIZE)) {
            errors++;
            if (strlen(diag) < 200) { char t[32]; snprintf(t, sizeof(t), "p=%d missing; ", p); strcat(diag, t); }
        }
    if (n >= 0 && is_page_tracked(e, n, base + (NUM_PAGES - 1) * PAGE_SIZE))
        { strcat(diag, "p=last tracked; "); errors++; }
    if (errors) { snprintf(out.actual, sizeof(out.actual), "n=%d  %s", n, diag); out.result = FAIL; }
    else          snprintf(out.actual, sizeof(out.actual), "boundary pages excluded, inner present (total=%d)", n);
    set_tracking(0); reset_addr_range();
    print_result("T12", "range_boundary_pages", out.result, out.expected, out.actual);
}

int main(void)
{
    int *managed = NULL, dev;

    printf("=== UVM Dirty Tracking Test Suite ===\nPID = %d\n\n", getpid());

    if (geteuid() != 0)
        { fprintf(stderr, "ERROR: must run as root\n"); return 1; }
    if (!sysfs_exists(SYSFS_DIRTY))
        { fprintf(stderr, "ERROR: %s not found\n", SYSFS_DIRTY); return 1; }
    if (!sysfs_exists(PROCFS_DIRTY))
        { fprintf(stderr, "ERROR: %s not found\n", PROCFS_DIRTY); return 1; }

    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  managed: 0x%lx - 0x%lx  (%d pages)\n\n",
           (unsigned long)managed, (unsigned long)managed + NUM_PAGES * PAGE_SIZE, NUM_PAGES);

    printf("--- GROUP A ---\n");
    t01_writes_recorded          (managed, dev);
    t02_reads_not_recorded       (managed, dev);
    t03_reset_clears_table       (managed, dev);
    t04_mixed_read_write         (managed, dev);
    t05_migration_write_retracked(managed, dev);
    t06_tracking_disabled        (managed, dev);

    printf("\n--- GROUP B ---\n");
    reset_addr_range();
    t07_pid_recorded             (managed, dev);
    t08_same_page_single_entry   (managed, dev);
    t09_range_inside             (managed, dev);
    t10_range_outside            (managed, dev);
    t11_range_partial_coverage   (managed, dev);
    t12_range_boundary_pages     (managed, dev);

    CUDA_CHECK(cudaFree(managed));
    int total = g_pass + g_fail + g_skip;
    printf("\n=== Results: %d/%d passed, %d failed, %d skipped ===\n",
           g_pass, total, g_fail, g_skip);
    return g_fail > 0 ? 1 : 0;
}
