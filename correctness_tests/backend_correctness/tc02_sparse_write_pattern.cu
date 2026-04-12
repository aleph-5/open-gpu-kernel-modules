/*
 * tc02_sparse_write_pattern.cu
 *
 * Write only every K-th page in a large allocation. Verifies that the backend
 * correctly tracks a sparse set — only the written pages appear, not all pages
 * in the span.
 *
 * This is important for bitmap backends that might set entire words/chunks at
 * once rather than individual bits.
 *
 * Flow:
 *   allocate TOTAL_PAGES pages
 *   start(delta) → write every STRIDE-th page → cutover → dump
 *   PASS: exactly (TOTAL_PAGES / STRIDE) entries, each at a STRIDE*PAGE_SIZE
 *         boundary offset from base. No unwritten pages present.
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

#define TOTAL_PAGES  256
#define STRIDE       8    /* write every 8th page */
#define WRITTEN      (TOTAL_PAGES / STRIDE)
#define PAGE_SIZE    4096
#define MAX_ENTRIES  4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

/* Each block writes one page at index blockIdx.x * STRIDE. */
__global__ void gpu_write_sparse(int *base, int total_pages, int stride)
{
    int p = blockIdx.x * stride;
    if (p >= total_pages) return;
    int ipp = PAGE_SIZE / sizeof(int);
    int *page = base + p * ipp;
    for (int i = 0; i < ipp; i++) page[i] = p * 100 + i;
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
    printf("[tc02] sparse_write_pattern: %d pages total, stride=%d → %d written\n",
           TOTAL_PAGES, STRIDE, WRITTEN);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    size_t alloc_bytes = (size_t)TOTAL_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc02] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_sparse<<<WRITTEN, 1>>>(managed, TOTAL_PAGES, STRIDE);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc02] dump: %d entries (want %d)\n", n, WRITTEN);

    /* Check that each entry corresponds to a written page (offset divisible by STRIDE). */
    int wrong_pages = 0, out_of_range = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr < base || e[i].addr >= base + alloc_bytes) { out_of_range++; continue; }
        unsigned long page_idx = (e[i].addr - base) / PAGE_SIZE;
        if (page_idx % STRIDE != 0) {
            printf("[tc02]   unexpected page idx=%lu (not a multiple of stride=%d)\n",
                   page_idx, STRIDE);
            wrong_pages++;
        }
    }

    /* Check all expected pages are present. */
    int missing = 0;
    for (int p = 0; p < TOTAL_PAGES; p += STRIDE) {
        unsigned long expected = base + (unsigned long)p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n; i++) if (e[i].addr == expected) { found = 1; break; }
        if (!found) missing++;
    }

    printf("[tc02] wrong_pages=%d out_of_range=%d missing=%d\n",
           wrong_pages, out_of_range, missing);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != WRITTEN || wrong_pages || out_of_range || missing);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (n != WRITTEN)  printf("[tc02]   count: got %d want %d\n", n, WRITTEN);
    if (wrong_pages)   printf("[tc02]   %d entries at non-stride offsets (backend marks extra pages)\n", wrong_pages);
    if (out_of_range)  printf("[tc02]   %d out-of-range entries\n", out_of_range);
    if (missing)       printf("[tc02]   %d expected pages missing from dump\n", missing);
    return failed;
}
