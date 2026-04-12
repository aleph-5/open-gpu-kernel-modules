/*
 * tc01_read_only_no_dirty.cu
 *
 * GPU read-only accesses must NOT dirty pages. If the GPU only reads from a
 * page (no write fault), the dirty tracker must not record it.
 *
 * The UVM dirty tracker hooks into write faults (PTE downgrade → write fault
 * on first write). A read fault does not trigger the dirty recording path.
 *
 * Flow:
 *   allocate managed memory; initialize on CPU (forces CPU residency)
 *   start(delta)
 *   GPU kernel reads (but does not write) all pages → sync
 *   cutover → dump
 *   PASS: dump is empty (0 entries — reads do not dirty pages)
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

#define NUM_PAGES   16
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

/* Read-only kernel: sums all values into a global accumulator (volatile to
 * prevent optimization) but never writes to the managed range. */
__global__ void gpu_read_only(const int *base, int num_pages, long long *result)
{
    int ipp = PAGE_SIZE / sizeof(int);
    long long sum = 0;
    for (int p = 0; p < num_pages; p++)
        for (int i = 0; i < ipp; i++)
            sum += base[p * ipp + i];
    atomicAdd((unsigned long long *)result, (unsigned long long)sum);
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
    printf("[tc01] read_only_no_dirty: %d pages\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));

    /* Initialize on CPU so pages start with CPU residency.
     * This forces the GPU to fault in when it reads them. */
    for (int i = 0; i < NUM_PAGES * (int)(PAGE_SIZE / sizeof(int)); i++)
        managed[i] = i;
    CUDA_CHECK(cudaDeviceSynchronize());

    long long *result = NULL;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(long long)));
    *result = 0;
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc01] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    /* GPU reads only — no writes to managed. */
    gpu_read_only<<<1, 1>>>(managed, NUM_PAGES, result);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc01] GPU read sum=%lld (discarded)\n", *result);

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc01] dump entries: n=%d (want 0)\n", n);

    stop_track();
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (n != 0) printf("[tc01]   %d pages dirtied by read-only access (expected 0)\n", n);
    return failed;
}
