/*
 * tc04_prefetch_no_dirty.cu
 *
 * cudaMemPrefetchAsync() moves pages to the GPU but does NOT constitute a
 * write — pages migrated via prefetch must NOT appear in the dirty dump.
 *
 * This test prefetches an allocation to the GPU and then reads it. Only if
 * the GPU subsequently *writes* to a page should it appear in the dump.
 *
 * Flow:
 *   allocate and init on CPU
 *   start(delta)
 *   cudaMemPrefetchAsync → GPU device (data migration, not a write fault)
 *   GPU reads all pages (no writes) → sync
 *   cutover → dump
 *   PASS: 0 entries (prefetch + read-only = not dirty)
 *
 *   Second epoch: same pages, now GPU writes them
 *   cutover → dump2
 *   PASS: all pages present in dump2
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

#define NUM_PAGES   8
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

__global__ void gpu_read_all(const int *base, int num_pages, long long *sink)
{
    int ipp = PAGE_SIZE / sizeof(int);
    long long s = 0;
    for (int p = 0; p < num_pages; p++)
        for (int i = 0; i < ipp; i++) s += base[p * ipp + i];
    atomicAdd((unsigned long long *)sink, (unsigned long long)s);
}

__global__ void gpu_write_all(int *base, int num_pages, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = 0; p < num_pages; p++)
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
    printf("[tc04] prefetch_no_dirty: %d pages\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));

    /* Initialize on CPU. */
    for (int i = 0; i < NUM_PAGES * (int)(PAGE_SIZE / sizeof(int)); i++)
        managed[i] = i;
    CUDA_CHECK(cudaDeviceSynchronize());

    long long *sink = NULL;
    CUDA_CHECK(cudaHostAlloc(&sink, sizeof(long long), cudaHostAllocDefault));
    *sink = 0;

    unsigned long base = (unsigned long)managed;
    printf("[tc04] pid=%d alloc=0x%lx device=%d\n", getpid(), base, device);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc04] start failed: %s\n", strerror(-rc)); return 1; }

    // Prefetch to GPU — migration, NOT a write fault
    // CUDA 13.0 signature: (ptr, size, cudaMemLocation, flags, stream)
    cudaMemLocation loc = {cudaMemLocationTypeDevice, device};
    CUDA_CHECK(cudaMemPrefetchAsync(managed, NUM_PAGES * PAGE_SIZE, loc, 0, 0));

    /* Read-only kernel. */
    gpu_read_all<<<1, 1>>>(managed, NUM_PAGES, sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Epoch 1: prefetch + read → expect 0 dirty pages. */
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e1[MAX_ENTRIES];
    int n1 = read_dump(e1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc04] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }
    printf("[tc04] epoch1 (prefetch+read): n=%d (want 0)\n", n1);

    /* Epoch 2: write same pages → expect all dirty. */
    gpu_write_all<<<1, 1>>>(managed, NUM_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e2[MAX_ENTRIES];
    int n2 = read_dump(e2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc04] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    int w_found = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n2; i++) if (e2[i].addr == pa) { w_found++; break; }
    }
    printf("[tc04] epoch2 (write): n=%d w_found=%d/%d (want %d)\n", n2, w_found, NUM_PAGES, NUM_PAGES);

    stop_track();
    CUDA_CHECK(cudaFreeHost(sink));
    CUDA_CHECK(cudaFree(managed));

    int failed = (n1 != 0 || w_found != NUM_PAGES);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (n1 != 0)           printf("[tc04]   %d pages dirty after prefetch+read (expected 0)\n", n1);
    if (w_found != NUM_PAGES) printf("[tc04]   only %d/%d pages recorded after write epoch\n", w_found, NUM_PAGES);
    return failed;
}
