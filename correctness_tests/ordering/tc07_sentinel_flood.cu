/*
 * tc07_sentinel_flood.cu
 *
 * Sentinel + 1M-thread flood ordering test (delta mode).
 *
 * Step 1: write a single sentinel page with a tiny kernel.
 *         cutover+dump → record ts_sentinel.
 * Step 2: write NUM_FLOOD_PAGES with a massive kernel (~1M threads).
 *         cutover+dump → all flood entries must have ts >= ts_sentinel.
 *
 * In delta mode each cutover creates a new snapshot epoch, so:
 *   - dump1 contains only the sentinel (epoch 1).
 *   - dump2 contains only the flood pages (epoch 2).
 *   Cross-epoch ordering: flood ts >= sentinel ts because the flood kernel
 *   was launched AFTER the sentinel's cudaDeviceSynchronize.
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

#define NUM_FLOOD_PAGES   256
#define FLOOD_BLOCKS      4096
#define FLOOD_THREADS     256
#define PAGE_SIZE         4096
#define INTS_PER_PAGE     (PAGE_SIZE / sizeof(int))
#define TOTAL_PAGES       (1 + NUM_FLOOD_PAGES)
#define MAX_ENTRIES       8192

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void write_sentinel(int *page)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) page[i] = 0xDEAD0000 + i;
}

__global__ void flood_write(int *base, int npages)
{
    int page = blockIdx.x % npages;
    int *p   = base + page * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = page * 777 + i;
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
    long flood_threads = (long)FLOOD_BLOCKS * FLOOD_THREADS;
    printf("[tc07] sentinel_then_flood: 1 sentinel + %d flood pages, %ld flood threads (delta mode)\n",
           NUM_FLOOD_PAGES, flood_threads);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *sentinel_page = managed;
    int *flood_base    = managed + INTS_PER_PAGE;
    unsigned long sentinel_va = (unsigned long)sentinel_page;
    unsigned long flood_va    = (unsigned long)flood_base;
    printf("[tc07] pid=%d sentinel=0x%lx flood_base=0x%lx\n",
           getpid(), sentinel_va, flood_va);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc07] start failed: %s\n", strerror(-rc)); return 1; }

    /* Step 1: sentinel. */
    write_sentinel<<<1, 1>>>(sentinel_page);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc07] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e1[MAX_ENTRIES];
    int n1 = read_dump(e1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc07] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }

    unsigned long ts_sentinel = 0;
    for (int i = 0; i < n1; i++)
        if (e1[i].addr == sentinel_va) { ts_sentinel = e1[i].ts; break; }
    printf("[tc07] sentinel ts=%lu (n1=%d)\n", ts_sentinel, n1);

    /* Step 2: flood. */
    flood_write<<<FLOOD_BLOCKS, FLOOD_THREADS>>>(flood_base, NUM_FLOOD_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc07] flood complete\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc07] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e2 = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e2) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n2 = read_dump(e2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc07] dump2 failed: %s\n", strerror(-n2)); stop_track(); free(e2); return 1; }
    printf("[tc07] post-flood dump: %d entries\n", n2);

    int early_violations = 0, flood_missing = 0;
    for (int p = 0; p < NUM_FLOOD_PAGES; p++) {
        unsigned long pa = flood_va + p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n2; i++) {
            if (e2[i].addr != pa) continue;
            found = 1;
            if (e2[i].ts < ts_sentinel) {
                printf("[tc07]   flood page %d: ts=%lu < ts_sentinel=%lu (violation)\n",
                       p, e2[i].ts, ts_sentinel);
                early_violations++;
            }
        }
        if (!found) flood_missing++;
    }

    printf("[tc07] flood_missing=%d ordering_violations=%d\n",
           flood_missing, early_violations);

    stop_track();
    CUDA_CHECK(cudaFree(managed));
    free(e2);

    int failed = (ts_sentinel == 0 || flood_missing > 0 || early_violations > 0);
    printf("[tc07] %s\n", failed ? "FAIL" : "PASS");
    if (ts_sentinel == 0)    printf("[tc07]   sentinel page not recorded\n");
    if (flood_missing > 0)   printf("[tc07]   %d flood pages not recorded\n", flood_missing);
    if (early_violations > 0) printf("[tc07]   %d flood entries ts < ts_sentinel\n", early_violations);
    return failed;
}
