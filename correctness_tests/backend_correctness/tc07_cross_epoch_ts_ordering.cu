/*
 * tc07_cross_epoch_ts_ordering.cu
 *
 * Timestamps must be strictly non-decreasing across epoch boundaries.
 * The maximum timestamp from epoch N must be <= the minimum timestamp
 * from epoch N+1 (clocks don't go backwards between cutovers).
 *
 * Flow:
 *   start(delta)
 *   epoch 1: write first half → cutover → dump1 (record max_ts_1)
 *   epoch 2: write second half → cutover → dump2 (record min_ts_2)
 *   PASS: min_ts_2 >= max_ts_1 (cross-epoch monotonicity)
 *         all ts in both dumps > 0
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

#define HALF_PAGES  16
#define TOTAL_PAGES (HALF_PAGES * 2)
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

__global__ void gpu_write_range(int *base, int start_page, int end_page, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = start_page; p < end_page; p++)
        for (int i = 0; i < ipp; i++) base[p * ipp + i] = tag * 1000 + p * 100 + i;
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

int main(void)
{
    printf("[tc07] cross_epoch_ts_ordering: %d+%d pages\n", HALF_PAGES, HALF_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc07] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc07] start failed: %s\n", strerror(-rc)); return 1; }

    /* Epoch 1. */
    gpu_write_range<<<1, 32>>>(managed, 0, HALF_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc07] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc07] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }

    unsigned long max_ts_1 = 0, zero_ts_1 = 0;
    for (int i = 0; i < n1; i++) {
        if (snap1[i].ts > max_ts_1) max_ts_1 = snap1[i].ts;
        if (snap1[i].ts == 0) zero_ts_1++;
    }
    printf("[tc07] epoch1: n=%d max_ts=%lu zero_ts=%lu\n", n1, max_ts_1, zero_ts_1);

    /* Epoch 2. */
    gpu_write_range<<<1, 32>>>(managed, HALF_PAGES, TOTAL_PAGES, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc07] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc07] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    unsigned long min_ts_2 = (unsigned long)-1UL, zero_ts_2 = 0;
    for (int i = 0; i < n2; i++) {
        if (snap2[i].ts < min_ts_2) min_ts_2 = snap2[i].ts;
        if (snap2[i].ts == 0) zero_ts_2++;
    }
    if (n2 == 0) min_ts_2 = 0; /* no entries */
    printf("[tc07] epoch2: n=%d min_ts=%lu zero_ts=%lu\n", n2, min_ts_2, zero_ts_2);
    printf("[tc07] cross-epoch: min_ts_2=%lu >= max_ts_1=%lu? %s\n",
           min_ts_2, max_ts_1, min_ts_2 >= max_ts_1 ? "YES" : "NO");

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n1 != HALF_PAGES || n2 != HALF_PAGES ||
                  zero_ts_1 || zero_ts_2 || min_ts_2 < max_ts_1);
    printf("[tc07] %s\n", failed ? "FAIL" : "PASS");
    if (n1 != HALF_PAGES) printf("[tc07]   epoch1: got %d want %d\n", n1, HALF_PAGES);
    if (n2 != HALF_PAGES) printf("[tc07]   epoch2: got %d want %d\n", n2, HALF_PAGES);
    if (zero_ts_1)        printf("[tc07]   epoch1: %lu zero timestamps\n", zero_ts_1);
    if (zero_ts_2)        printf("[tc07]   epoch2: %lu zero timestamps\n", zero_ts_2);
    if (min_ts_2 < max_ts_1)
        printf("[tc07]   cross-epoch regression: min_ts_2=%lu < max_ts_1=%lu\n",
               min_ts_2, max_ts_1);
    return failed;
}
