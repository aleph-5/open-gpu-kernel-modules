/*
 * tc01_all_entries_page_aligned.cu
 *
 * Every entry returned by the dump must have a page-aligned address
 * (addr % PAGE_SIZE == 0). This is a fundamental invariant of the dirty
 * tracking backend: only page granularity is meaningful.
 *
 * Also verifies: no duplicate addresses, all within the allocation range,
 * all timestamps non-zero.
 *
 * Flow:
 *   start(delta) → write NUM_PAGES → cutover → dump
 *   verify: all addrs page-aligned, no dups, all in range, all ts > 0
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

#define NUM_PAGES   128
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

__global__ void gpu_write_pages(int *base, int num_pages)
{
    int p = blockIdx.x;
    if (p >= num_pages) return;
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) base[p * ipp + i] = p * 100 + i;
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
    printf("[tc01] all_entries_page_aligned: %d pages\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    size_t alloc_bytes = (size_t)NUM_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc01] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_pages<<<NUM_PAGES, 1>>>(managed, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc01] dump: %d entries (want %d)\n", n, NUM_PAGES);

    int misaligned = 0, zero_ts = 0, out_of_range = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr % PAGE_SIZE != 0) {
            printf("[tc01]   misaligned: addr=0x%lx\n", e[i].addr);
            misaligned++;
        }
        if (e[i].ts == 0) zero_ts++;
        if (e[i].addr < base || e[i].addr >= base + alloc_bytes) {
            printf("[tc01]   out-of-range: addr=0x%lx\n", e[i].addr);
            out_of_range++;
        }
    }

    /* Duplicate check. */
    int dups = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (e[i].addr == e[j].addr) { dups++; break; }

    printf("[tc01] misaligned=%d zero_ts=%d out_of_range=%d dups=%d\n",
           misaligned, zero_ts, out_of_range, dups);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != NUM_PAGES || misaligned || zero_ts || out_of_range || dups);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (n != NUM_PAGES)  printf("[tc01]   count: got %d want %d\n", n, NUM_PAGES);
    if (misaligned)      printf("[tc01]   %d misaligned addresses\n", misaligned);
    if (zero_ts)         printf("[tc01]   %d zero timestamps\n", zero_ts);
    if (out_of_range)    printf("[tc01]   %d out-of-range addresses\n", out_of_range);
    if (dups)            printf("[tc01]   %d duplicate addresses\n", dups);
    return failed;
}
