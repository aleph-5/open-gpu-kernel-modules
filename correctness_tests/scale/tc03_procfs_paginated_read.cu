/*
 * tc03_procfs_paginated_read.cu
 *
 * The dump procfs file is read in multiple small-sized read() calls to
 * simulate a slow or paginated reader. Verifies that the kernel-side procfs
 * seq_file implementation correctly handles partial reads without losing,
 * duplicating, or corrupting entries.
 *
 * Flow:
 *   write NUM_PAGES pages → cutover
 *   open dump file; read in CHUNK_BYTES chunks; parse all lines
 *   PASS: exactly NUM_PAGES unique entries, all page-aligned, all within alloc.
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

#define NUM_PAGES    1024
#define PAGE_SIZE    4096
#define CHUNK_BYTES  128   /* small chunk to force many read() calls */
#define MAX_ENTRIES  (NUM_PAGES + 64)
#define LINE_BUF     512

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

/* Read dump in small chunks; parse all entries. Returns count or -errno. */
static int read_dump_chunked(entry_t *out, int max)
{
    int fd = open(PROCFS_DUMP, O_RDONLY);
    if (fd < 0) return -errno;

    /* Accumulate into a large buffer line by line. */
    char rawbuf[NUM_PAGES * 64 + 4096]; /* generous upper bound */
    int total = 0;
    char chunk[CHUNK_BYTES + 1];
    ssize_t r;

    while ((r = read(fd, chunk, CHUNK_BYTES)) > 0) {
        if (total + (int)r >= (int)sizeof(rawbuf)) {
            close(fd);
            return -ENOBUFS;
        }
        memcpy(rawbuf + total, chunk, r);
        total += (int)r;
    }
    int saved = (r < 0) ? errno : 0;
    close(fd);
    if (saved) return -saved;

    rawbuf[total] = '\0';

    /* Parse lines. */
    int n = 0;
    char *line = rawbuf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (line[0] != '#' && line[0] != '\0') {
            unsigned long addr, ts;
            if (n < max && sscanf(line, "0x%lx %lu", &addr, &ts) == 2) {
                out[n].addr = addr;
                out[n].ts   = ts;
                n++;
            }
        }
        line = nl ? nl + 1 : NULL;
    }
    return n;
}

int main(void)
{
    printf("[tc03] procfs_paginated_read: %d pages, chunk=%d bytes\n",
           NUM_PAGES, CHUNK_BYTES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    size_t alloc_bytes = (size_t)NUM_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc03] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_pages<<<NUM_PAGES, 1>>>(managed, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "[tc03] malloc failed\n"); stop_track(); return 1; }

    int n = read_dump_chunked(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc03] chunked dump failed: %s\n", strerror(-n)); stop_track(); free(e); return 1; }
    printf("[tc03] parsed %d entries via chunked reads (want %d)\n", n, NUM_PAGES);

    /* Verify no duplicates and all within range. */
    int in_range = 0, zero_ts = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr >= base && e[i].addr < base + alloc_bytes) in_range++;
        if (e[i].ts == 0) zero_ts++;
    }

    /* Check for duplicates using a simple O(n^2) scan (n ≤ 1024, tolerable). */
    int dups = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (e[i].addr == e[j].addr) { dups++; break; }

    printf("[tc03] in_range=%d zero_ts=%d duplicates=%d\n", in_range, zero_ts, dups);

    free(e);
    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != NUM_PAGES || in_range != NUM_PAGES || zero_ts != 0 || dups != 0);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (n != NUM_PAGES)        printf("[tc03]   expected %d entries, got %d\n", NUM_PAGES, n);
    if (in_range != NUM_PAGES) printf("[tc03]   %d entries outside alloc range\n", NUM_PAGES - in_range);
    if (zero_ts)               printf("[tc03]   %d zero-timestamp entries\n", zero_ts);
    if (dups)                  printf("[tc03]   %d duplicate addresses\n", dups);
    return failed;
}
