/*
 * tc02_repeated_growth_and_query.cu
 *
 * Repeat many cycles of: write N new pages → cutover → dump → verify count.
 * Each cycle the tracked set grows by STEP pages. Tests that the backing
 * data structure handles incremental growth without losing entries and that
 * repeated cutover+dump pairs stay consistent.
 *
 * In DELTA mode each epoch is independent: each dump must contain exactly
 * STEP pages (the new pages written in that epoch).
 *
 * Flow (delta mode):
 *   start(delta)
 *   for c in 0..NUM_CYCLES:
 *       write pages [c*STEP .. (c+1)*STEP)
 *       cutover → dump
 *       PASS: dump has exactly STEP entries, all within [c*STEP .. (c+1)*STEP)
 *   stop
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

#define NUM_CYCLES  20
#define STEP        512      /* pages per cycle */
#define TOTAL_PAGES (NUM_CYCLES * STEP)
#define PAGE_SIZE   4096
#define MAX_ENTRIES (STEP + 64)

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
    int p = start_page + blockIdx.x;
    if (p >= end_page) return;
    int ipp = PAGE_SIZE / sizeof(int);
    int *page = base + p * ipp;
    for (int i = 0; i < ipp; i++) page[i] = tag * 1000 + p * 100 + i;
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
    printf("[tc02] repeated_growth_and_query: %d cycles × %d pages\n",
           NUM_CYCLES, STEP);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    size_t alloc_bytes = (size_t)TOTAL_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc02] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); return 1; }

    int total_failed = 0;
    entry_t e[MAX_ENTRIES];

    for (int c = 0; c < NUM_CYCLES; c++) {
        int start_p = c * STEP;
        int end_p   = start_p + STEP;

        gpu_write_range<<<STEP, 1>>>(managed, start_p, end_p, c);
        CUDA_CHECK(cudaDeviceSynchronize());

        rc = cutover();
        if (rc) { fprintf(stderr, "[tc02] cycle %d cutover failed: %s\n", c, strerror(-rc)); stop_track(); break; }

        int n = read_dump(e, MAX_ENTRIES);
        if (n < 0) { fprintf(stderr, "[tc02] cycle %d dump failed: %s\n", c, strerror(-n)); stop_track(); break; }

        /* Count how many of the cycle's pages are in the dump. */
        int in_epoch = 0;
        for (int p = start_p; p < end_p; p++) {
            unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
            for (int i = 0; i < n; i++) if (e[i].addr == pa) { in_epoch++; break; }
        }

        if (n != STEP || in_epoch != STEP) {
            printf("[tc02] cycle %d FAIL: n=%d in_epoch=%d (want %d)\n",
                   c, n, in_epoch, STEP);
            total_failed++;
        }
    }

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    printf("[tc02] cycles_failed=%d/%d\n", total_failed, NUM_CYCLES);
    printf("[tc02] %s\n", total_failed ? "FAIL" : "PASS");
    return total_failed ? 1 : 0;
}
