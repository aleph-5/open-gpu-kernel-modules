/*
 * tc01_large_page_count.cu
 *
 * Write 100,000 distinct pages in a single epoch and verify all appear in the
 * dump. This exercises the backing data structure (bitmap, xarray, vector, etc.)
 * at scale and verifies there is no truncation, off-by-one, or capacity cap.
 *
 * Allocation: 100K pages × 4KB = 400MB managed memory.
 * Kernel: 100K blocks × 1 thread each, each block writes one page.
 *
 * Flow:
 *   start(delta) → launch 100K-page write kernel → sync → cutover → dump
 *   PASS: exactly 100K entries in dump, all within the allocation range.
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

#define NUM_PAGES    100000
#define PAGE_SIZE    4096
#define MAX_ENTRIES  (NUM_PAGES + 64)  /* slight over-alloc for safety */

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

/* Each block writes exactly one page (identified by blockIdx.x). */
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

/* Read dump into a dynamically allocated array (caller frees). */
static int read_dump_dyn(entry_t **out_p, int max)
{
    FILE *f = fopen(PROCFS_DUMP, "r");
    if (!f) return -errno;

    entry_t *buf = (entry_t *)malloc(max * sizeof(entry_t));
    if (!buf) { fclose(f); return -ENOMEM; }

    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (n < max && sscanf(line, "0x%lx %lu", &buf[n].addr, &buf[n].ts) == 2)
            n++;
    }
    int err = ferror(f) ? errno : 0;
    fclose(f);
    if (err) { free(buf); return -err; }
    *out_p = buf;
    return n;
}

int main(void)
{
    printf("[tc01] large_page_count: %d pages (%.0f MB)\n",
           NUM_PAGES, (double)NUM_PAGES * PAGE_SIZE / (1024*1024));

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    size_t alloc_bytes = (size_t)NUM_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    /* Do not memset — let UVM decide initial residency. */
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc01] pid=%d alloc=0x%lx size=%zu MB\n",
           getpid(), base, alloc_bytes / (1024*1024));

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    /* Launch with NUM_PAGES blocks — each writes one page. */
    gpu_write_pages<<<NUM_PAGES, 1>>>(managed, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc01] GPU writes complete\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e = NULL;
    int n = read_dump_dyn(&e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc01] dump: %d entries (want %d)\n", n, NUM_PAGES);

    /* Count entries within the allocation range. */
    int in_range = 0, zero_ts = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr >= base && e[i].addr < base + alloc_bytes) in_range++;
        if (e[i].ts == 0) zero_ts++;
    }
    printf("[tc01] in_range=%d zero_ts=%d\n", in_range, zero_ts);

    free(e);
    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != NUM_PAGES || in_range != NUM_PAGES || zero_ts != 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (n != NUM_PAGES)       printf("[tc01]   expected %d entries, got %d\n", NUM_PAGES, n);
    if (in_range != NUM_PAGES) printf("[tc01]   %d entries outside allocation range\n", NUM_PAGES - in_range);
    if (zero_ts)               printf("[tc01]   %d entries have zero timestamp\n", zero_ts);
    return failed;
}
