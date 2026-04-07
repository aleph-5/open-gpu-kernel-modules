/*
 * test_dirty_tracking_suite.cu - generic dirty tracking suite
 *
 * This suite validates base semantics (write tracking, reset behavior,
 * disabled behavior) and dirty_range filtering behavior against procfs.
 */

#include <cuda_runtime.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PROCFS_START "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_PAGES "/proc/driver/nvidia-uvm/dirty_pages"
#define PROCFS_RANGE "/proc/driver/nvidia-uvm/dirty_range"

#define NUM_PAGES 8
#define PAGE_SIZE 4096
#define INTS_PER_PAGE (PAGE_SIZE / sizeof(int))
#define NUM_INTS (NUM_PAGES * INTS_PER_PAGE)
#define MAX_ENTRIES 4096

#define CUDA_CHECK(c)                                                            \
    do {                                                                         \
        cudaError_t _e = (c);                                                    \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__,                                                     \
                    __LINE__,                                                     \
                    cudaGetErrorString(_e));                                     \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

typedef enum {
    PASS = 0,
    FAIL = 1,
    SKIP = 2,
} result_t;

typedef struct {
    unsigned long addr;
    unsigned long ts;
} entry_t;

typedef struct {
    int *managed;
    unsigned long base;
} fixture_t;

typedef struct {
    result_t result;
    char expected[256];
    char actual[256];
} outcome_t;

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;

__global__ void read_kernel(const int *data, int n, volatile int *sink)
{
    int acc = 0;
    for (int i = 0; i < n; ++i)
        acc += data[i];
    *sink = acc;
}

__global__ void write_kernel(int *data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = i + 1;
}

__global__ void write_range_kernel(int *data, int start_page, int end_page)
{
    for (int p = start_page; p < end_page; ++p) {
        for (int i = 0; i < INTS_PER_PAGE; ++i)
            data[p * INTS_PER_PAGE + i] = p * INTS_PER_PAGE + i + 1;
    }
}

static void print_result(const char *id,
                         const char *name,
                         result_t result,
                         const char *expected,
                         const char *actual)
{
    const char *status = result == PASS ? "PASS" : result == SKIP ? "SKIP" : "FAIL";

    printf("  [%s] %s - %s\n", status, id, name);
    if (expected && expected[0])
        printf("         expected: %s\n", expected);
    if (actual && actual[0])
        printf("         actual:   %s\n", actual);

    if (result == PASS)
        ++g_pass;
    else if (result == FAIL)
        ++g_fail;
    else
        ++g_skip;
}

static int procfs_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static void procfs_write(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    ssize_t written;

    if (fd < 0) {
        perror(path);
        exit(1);
    }

    written = write(fd, val, strlen(val));
    if (written < 0) {
        perror("write");
        close(fd);
        exit(1);
    }

    close(fd);
}

static void start_tracking(void) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    procfs_write(PROCFS_START, buf);
}
static void stop_tracking(void) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    procfs_write(PROCFS_STOP, buf);
}
static void reset_range(void)   { procfs_write(PROCFS_RANGE, "0x0 0xffffffffffffffff\n");	}

static void set_range(unsigned long start, unsigned long end)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "0x%lx 0x%lx\n", start, end);
    procfs_write(PROCFS_RANGE, buf);
}

/* Return: entries parsed, -1 if file open failed, -2 if table not active. */
static int read_pages(entry_t *out, int max)
{
    FILE *f = fopen(PROCFS_PAGES, "r");
    char line[256];
    int n = 0;

    if (!f)
        return -1;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) {
                fclose(f);
                return -2;
            }
            continue;
        }

        if (n >= max)
            break;

        if (sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2)
            ++n;
    }

    fclose(f);
    return n;
}

static int page_tracked(const entry_t *entries, int n, unsigned long addr)
{
    unsigned long page_addr = addr & ~((unsigned long)(PAGE_SIZE - 1));

    for (int i = 0; i < n; ++i) {
        if (entries[i].addr == page_addr)
            return 1;
    }

    return 0;
}

static int count_missing(const fixture_t *fx, const entry_t *entries, int n, int start_page, int end_page)
{
    int missing = 0;

    for (int p = start_page; p < end_page; ++p) {
        unsigned long addr = fx->base + (unsigned long)p * PAGE_SIZE;
        if (!page_tracked(entries, n, addr))
            ++missing;
    }

    return missing;
}

static int count_present(const fixture_t *fx, const entry_t *entries, int n, int start_page, int end_page)
{
    int present = 0;

    for (int p = start_page; p < end_page; ++p) {
        unsigned long addr = fx->base + (unsigned long)p * PAGE_SIZE;
        if (page_tracked(entries, n, addr))
            ++present;
    }

    return present;
}

static void t01_writes_recorded(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };

    snprintf(out.expected, sizeof(out.expected), "all %d pages present after GPU write", NUM_PAGES);

    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int missing = n >= 0 ? count_missing(fx, entries, n, 0, NUM_PAGES) : NUM_PAGES;

    if (n < 0 || missing > 0) {
        snprintf(out.actual,
                 sizeof(out.actual),
                 "n=%d missing=%d/%d",
                 n,
                 missing,
                 NUM_PAGES);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "all pages present (total=%d)", n);
    }

    stop_tracking();
    print_result("T01", "writes_recorded", out.result, out.expected, out.actual);
}

static void t02_reads_not_recorded(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };
    int *sink = NULL;

    snprintf(out.expected, sizeof(out.expected), "0 pages tracked for read-only GPU access");

    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));

    start_tracking();
    read_kernel<<<1, 1>>>(fx->managed, NUM_INTS, (volatile int *)sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int spurious = n >= 0 ? count_present(fx, entries, n, 0, NUM_PAGES) : NUM_PAGES;

    if (n < 0 || spurious > 0) {
        snprintf(out.actual,
                 sizeof(out.actual),
                 "n=%d spurious=%d",
                 n,
                 spurious);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "0 tracked pages (total=%d)", n);
    }

    stop_tracking();
    CUDA_CHECK(cudaFree(sink));

    print_result("T02", "reads_not_recorded", out.result, out.expected, out.actual);
}

static void t03_reset_clears_table(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };
    const int half = NUM_PAGES / 2;

    snprintf(out.expected,
             sizeof(out.expected),
             "pages 0..%d absent, pages %d..%d present after stop+start",
             half - 1,
             half,
             NUM_PAGES - 1);

    start_tracking();
    write_range_kernel<<<1, 1>>>(fx->managed, 0, half);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Stop+start resets the table in current implementation. */
    stop_tracking();
    start_tracking();
    write_range_kernel<<<1, 1>>>(fx->managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int ghost = n >= 0 ? count_present(fx, entries, n, 0, half) : half;
    int missing = n >= 0 ? count_missing(fx, entries, n, half, NUM_PAGES) : (NUM_PAGES - half);

    if (n < 0 || ghost > 0 || missing > 0) {
        snprintf(out.actual,
                 sizeof(out.actual),
                 "n=%d ghost=%d missing=%d",
                 n,
                 ghost,
                 missing);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "table reset behavior correct (total=%d)", n);
    }

    stop_tracking();
    print_result("T03", "reset_clears_table", out.result, out.expected, out.actual);
}

static void t04_mixed_read_write(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };
    int *sink = NULL;
    const int half = NUM_PAGES / 2;

    snprintf(out.expected,
             sizeof(out.expected),
             "read-only half clean, write half tracked");

    CUDA_CHECK(cudaMalloc(&sink, sizeof(int)));

    start_tracking();
    read_kernel<<<1, 1>>>(fx->managed, half * INTS_PER_PAGE, (volatile int *)sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    write_range_kernel<<<1, 1>>>(fx->managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int read_spurious = n >= 0 ? count_present(fx, entries, n, 0, half) : half;
    int write_missing = n >= 0 ? count_missing(fx, entries, n, half, NUM_PAGES) : (NUM_PAGES - half);

    if (n < 0 || read_spurious > 0 || write_missing > 0) {
        snprintf(out.actual,
                 sizeof(out.actual),
                 "n=%d read_spurious=%d write_missing=%d",
                 n,
                 read_spurious,
                 write_missing);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "read/write split respected (total=%d)", n);
    }

    stop_tracking();
    CUDA_CHECK(cudaFree(sink));

    print_result("T04", "mixed_read_write", out.result, out.expected, out.actual);
}

static void t05_restart_and_retrack(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };

    snprintf(out.expected,
             sizeof(out.expected),
             "all pages tracked after stop+start + second write");

    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    stop_tracking();
    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int missing = n >= 0 ? count_missing(fx, entries, n, 0, NUM_PAGES) : NUM_PAGES;

    if (n < 0 || missing > 0) {
        snprintf(out.actual,
                 sizeof(out.actual),
                 "n=%d missing=%d/%d",
                 n,
                 missing,
                 NUM_PAGES);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "all pages retracked (total=%d)", n);
    }

    stop_tracking();
    print_result("T05", "restart_and_retrack", out.result, out.expected, out.actual);
}

static void t06_tracking_disabled(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };

    (void)fx;

    snprintf(out.expected,
             sizeof(out.expected),
             "dirty_pages returns '# dirty tracking not active' when stopped");

    int n = read_pages(entries, MAX_ENTRIES);

    if (n != -2) {
        snprintf(out.actual, sizeof(out.actual), "expected -2, got %d", n);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "returned not-active marker");
    }

    print_result("T06", "tracking_disabled", out.result, out.expected, out.actual);
}

/* T07 intentionally reserved to keep historical numbering stable. */

static void t08_same_page_single_entry(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };

    snprintf(out.expected,
             sizeof(out.expected),
             "exactly %d entries after writing same pages twice",
             NUM_PAGES);

    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    if (n != NUM_PAGES) {
        snprintf(out.actual, sizeof(out.actual), "expected %d, got %d", NUM_PAGES, n);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "no duplicate entries");
    }

    stop_tracking();
    print_result("T08", "same_page_single_entry", out.result, out.expected, out.actual);
}

static void t09_range_inside(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };
    unsigned long end = fx->base + (unsigned long)NUM_PAGES * PAGE_SIZE;

    snprintf(out.expected,
             sizeof(out.expected),
             "all pages visible when range exactly covers allocation");

    set_range(fx->base, end);
    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int missing = n >= 0 ? count_missing(fx, entries, n, 0, NUM_PAGES) : NUM_PAGES;

    if (n < 0 || missing > 0) {
        snprintf(out.actual, sizeof(out.actual), "n=%d missing=%d", n, missing);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "all pages in-range (total=%d)", n);
    }

    stop_tracking();
    reset_range();
    print_result("T09", "range_inside", out.result, out.expected, out.actual);
}

static void t10_range_outside(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };
    unsigned long range_start = fx->base + (unsigned long)NUM_PAGES * PAGE_SIZE;
    unsigned long range_end = range_start + (unsigned long)NUM_PAGES * PAGE_SIZE;

    snprintf(out.expected,
             sizeof(out.expected),
             "0 pages visible when range is outside allocation");

    set_range(range_start, range_end);
    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int spurious = n >= 0 ? count_present(fx, entries, n, 0, NUM_PAGES) : NUM_PAGES;

    if (n < 0 || spurious > 0) {
        snprintf(out.actual, sizeof(out.actual), "n=%d spurious=%d", n, spurious);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "no managed pages in result (total=%d)", n);
    }

    stop_tracking();
    reset_range();
    print_result("T10", "range_outside", out.result, out.expected, out.actual);
}

static void t11_range_partial_coverage(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };
    const int half = NUM_PAGES / 2;
    unsigned long end = fx->base + (unsigned long)half * PAGE_SIZE;

    snprintf(out.expected,
             sizeof(out.expected),
             "first half present, second half absent");

    set_range(fx->base, end);
    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int in_missing = n >= 0 ? count_missing(fx, entries, n, 0, half) : half;
    int out_present = n >= 0 ? count_present(fx, entries, n, half, NUM_PAGES) : (NUM_PAGES - half);

    if (n < 0 || in_missing > 0 || out_present > 0) {
        snprintf(out.actual,
                 sizeof(out.actual),
                 "n=%d in_missing=%d out_present=%d",
                 n,
                 in_missing,
                 out_present);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "partial range filtering correct (total=%d)", n);
    }

    stop_tracking();
    reset_range();
    print_result("T11", "range_partial_coverage", out.result, out.expected, out.actual);
}

static void t12_range_boundary_pages(const fixture_t *fx)
{
    entry_t entries[MAX_ENTRIES];
    outcome_t out = { PASS, "", "" };
    unsigned long start = fx->base + PAGE_SIZE;
    unsigned long end = fx->base + (unsigned long)(NUM_PAGES - 1) * PAGE_SIZE;

    snprintf(out.expected,
             sizeof(out.expected),
             "page 0 and last page absent, inner pages present");

    set_range(start, end);
    start_tracking();
    write_kernel<<<1, 1>>>(fx->managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n = read_pages(entries, MAX_ENTRIES);
    int first_present = n >= 0 ? count_present(fx, entries, n, 0, 1) : 1;
    int middle_missing = n >= 0 ? count_missing(fx, entries, n, 1, NUM_PAGES - 1) : (NUM_PAGES - 2);
    int last_present = n >= 0 ? count_present(fx, entries, n, NUM_PAGES - 1, NUM_PAGES) : 1;

    if (n < 0 || first_present > 0 || middle_missing > 0 || last_present > 0) {
        snprintf(out.actual,
                 sizeof(out.actual),
                 "n=%d first_present=%d middle_missing=%d last_present=%d",
                 n,
                 first_present,
                 middle_missing,
                 last_present);
        out.result = FAIL;
    }
    else {
        snprintf(out.actual, sizeof(out.actual), "boundary filtering correct (total=%d)", n);
    }

    stop_tracking();
    reset_range();
    print_result("T12", "range_boundary_pages", out.result, out.expected, out.actual);
}

int main(void)
{
    fixture_t fx = { 0 };

    printf("=== UVM Dirty Tracking Test Suite ===\n\n");

    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }

    if (!procfs_exists(PROCFS_START) ||
        !procfs_exists(PROCFS_STOP) ||
        !procfs_exists(PROCFS_PAGES) ||
        !procfs_exists(PROCFS_RANGE)) {
        fprintf(stderr, "ERROR: one or more procfs controls are missing\n");
        return 1;
    }

    CUDA_CHECK(cudaMallocManaged(&fx.managed, NUM_PAGES * PAGE_SIZE));
    memset(fx.managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    fx.base = (unsigned long)fx.managed;

    printf("  managed: 0x%lx - 0x%lx (%d pages)\n\n",
           fx.base,
           fx.base + (unsigned long)NUM_PAGES * PAGE_SIZE,
           NUM_PAGES);

    printf("--- GROUP A: basic semantics ---\n");
    t01_writes_recorded(&fx);
    t02_reads_not_recorded(&fx);
    t03_reset_clears_table(&fx);
    t04_mixed_read_write(&fx);
    t05_restart_and_retrack(&fx);
    t06_tracking_disabled(&fx);

    printf("\n--- GROUP B: range filtering ---\n");
    reset_range();
    t08_same_page_single_entry(&fx);
    t09_range_inside(&fx);
    t10_range_outside(&fx);
    t11_range_partial_coverage(&fx);
    t12_range_boundary_pages(&fx);

    CUDA_CHECK(cudaFree(fx.managed));

    int total = g_pass + g_fail + g_skip;
    printf("\n=== Results: %d/%d passed, %d failed, %d skipped ===\n",
           g_pass,
           total,
           g_fail,
           g_skip);

    return g_fail == 0 ? 0 : 1;
}
