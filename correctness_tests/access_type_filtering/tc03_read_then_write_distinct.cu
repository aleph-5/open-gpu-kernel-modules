/*
 * tc03_read_then_write_distinct.cu
 *
 * Split an allocation into two ranges:
 *   Range R: GPU reads only (no writes)
 *   Range W: GPU writes only
 *
 * PASS: dump contains pages from W but NOT from R.
 *
 * This verifies that the dirty tracker correctly distinguishes read-fault
 * pages from write-fault pages even when they're from the same allocation.
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

#define HALF_PAGES  8
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

/* Read-only kernel for first HALF_PAGES. */
__global__ void gpu_read_range(const int *base, int num_pages, long long *sink)
{
    int ipp = PAGE_SIZE / sizeof(int);
    long long s = 0;
    for (int p = 0; p < num_pages; p++)
        for (int i = 0; i < ipp; i++) s += base[p * ipp + i];
    atomicAdd((unsigned long long *)sink, (unsigned long long)s);
}

/* Write-only kernel for second HALF_PAGES. */
__global__ void gpu_write_range(int *base, int start_page, int num_pages, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = start_page; p < start_page + num_pages; p++)
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

static int count_in_range(entry_t *e, int n, unsigned long base, int start_p, int end_p)
{
    int found = 0;
    for (int p = start_p; p < end_p; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc03] read_then_write_distinct: R=%d W=%d pages\n", HALF_PAGES, HALF_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));

    /* Initialize on CPU so all pages start with CPU residency. */
    for (int i = 0; i < TOTAL_PAGES * (int)(PAGE_SIZE / sizeof(int)); i++)
        managed[i] = i;
    CUDA_CHECK(cudaDeviceSynchronize());

    long long *sink = NULL;
    CUDA_CHECK(cudaMallocManaged(&sink, sizeof(long long)));
    *sink = 0;
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc03] pid=%d alloc=0x%lx R=[0,%d) W=[%d,%d)\n",
           getpid(), base, HALF_PAGES, HALF_PAGES, TOTAL_PAGES);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start failed: %s\n", strerror(-rc)); return 1; }

    /* Read the first half. */
    gpu_read_range<<<1, 1>>>(managed, HALF_PAGES, sink);
    /* Write the second half. */
    gpu_write_range<<<1, 1>>>(managed, HALF_PAGES, HALF_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc03] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }

    int r_found = count_in_range(e, n, base, 0, HALF_PAGES);
    int w_found = count_in_range(e, n, base, HALF_PAGES, TOTAL_PAGES);
    printf("[tc03] dump: n=%d R_found=%d/%d W_found=%d/%d (want R=0 W=%d)\n",
           n, r_found, HALF_PAGES, w_found, HALF_PAGES, HALF_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(managed));

    int failed = (r_found != 0 || w_found != HALF_PAGES);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (r_found != 0)        printf("[tc03]   %d read-only pages incorrectly dirtied\n", r_found);
    if (w_found != HALF_PAGES) printf("[tc03]   only %d/%d write pages recorded\n", w_found, HALF_PAGES);
    return failed;
}
