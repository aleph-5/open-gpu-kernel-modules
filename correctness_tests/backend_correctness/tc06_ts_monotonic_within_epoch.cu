/*
 * tc06_ts_monotonic_within_epoch.cu
 *
 * Timestamps within a single epoch must be non-decreasing when pages are
 * written sequentially (with synchronization between each write).
 *
 * Each page is written one at a time with a cudaDeviceSynchronize between
 * writes to ensure the fault for page[i] is resolved before page[i+1] begins.
 * The recorded timestamps must satisfy: ts[i] <= ts[i+1] for all i in addr order.
 *
 * This verifies that the clock source used for timestamps is monotonic and
 * that the timestamp is recorded at fault time (not at cutover time).
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

#define NUM_PAGES   32
#define PAGE_SIZE   4096
#define MAX_ENTRIES 4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_page(int *page, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) page[i] = tag * 1000 + i;
}

static int procfs_write_exact(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -errno;
    ssize_t n = write(fd, val, strlen(val));
    int saved = errno;
    close(fd);
    if (n < 0) return -saved;
    return 0;
}

static int start_track_delta(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d delta", getpid());
    return procfs_write_exact(PROCFS_START, buf);
}

static int stop_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_STOP, buf);
}

static int cutover(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_CUTOVER, buf);
}

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
    if (ferror(f)) { int saved = errno; fclose(f); return -saved; }
    fclose(f);
    return n;
}

static int cmp_by_addr(const void *a, const void *b)
{
    const entry_t *ea = (const entry_t *)a;
    const entry_t *eb = (const entry_t *)b;
    if (ea->addr < eb->addr) return -1;
    if (ea->addr > eb->addr) return  1;
    return 0;
}

int main(void)
{
    printf("[tc06] ts_monotonic_within_epoch: %d pages (sequential writes)\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc06] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc06] start failed: %s\n", strerror(-rc)); return 1; }

    /* Write pages one at a time with sync between each. */
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = 0; p < NUM_PAGES; p++) {
        gpu_write_page<<<1, 1>>>(managed + p * ipp, p + 1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc06] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc06] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc06] dump: %d entries (want %d)\n", n, NUM_PAGES);

    /* Sort by address to get sequential order. */
    qsort(e, n, sizeof(entry_t), cmp_by_addr);

    /* Check monotonically non-decreasing timestamps. */
    int regressions = 0;
    for (int i = 1; i < n; i++) {
        if (e[i].ts < e[i-1].ts) {
            printf("[tc06]   ts regression at i=%d: addr=0x%lx ts=%lu < prev_ts=%lu\n",
                   i, e[i].addr, e[i].ts, e[i-1].ts);
            regressions++;
        }
    }

    /* Check all non-zero. */
    int zero_ts = 0;
    for (int i = 0; i < n; i++) if (e[i].ts == 0) zero_ts++;

    printf("[tc06] regressions=%d zero_ts=%d\n", regressions, zero_ts);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != NUM_PAGES || regressions != 0 || zero_ts != 0);
    printf("[tc06] %s\n", failed ? "FAIL" : "PASS");
    if (n != NUM_PAGES)  printf("[tc06]   count: got %d want %d\n", n, NUM_PAGES);
    if (regressions)     printf("[tc06]   %d timestamp regressions (non-monotonic)\n", regressions);
    if (zero_ts)         printf("[tc06]   %d zero timestamps\n", zero_ts);
    return failed;
}
