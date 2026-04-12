/*
 * tc04_multiple_allocs_independent.cu
 *
 * Multiple non-contiguous managed allocations in the same process are all
 * tracked independently in the same session. Writing to one must not cause
 * pages from another to appear in the dump (no aliasing between allocations).
 *
 * Flow:
 *   start(delta)
 *   write only alloc[1] (middle of 3 allocations) → cutover → dump
 *   PASS: only alloc[1] pages appear; alloc[0] and alloc[2] are absent.
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

#define NUM_ALLOCS  3
#define PAGES_EACH  16
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

static int count_in_alloc(entry_t *e, int n, unsigned long base, int num_pages)
{
    int found = 0;
    for (int p = 0; p < num_pages; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++) if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc04] multiple_allocs_independent: %d allocs × %d pages\n",
           NUM_ALLOCS, PAGES_EACH);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *allocs[NUM_ALLOCS];
    unsigned long bases[NUM_ALLOCS];

    for (int a = 0; a < NUM_ALLOCS; a++) {
        CUDA_CHECK(cudaMallocManaged(&allocs[a], PAGES_EACH * PAGE_SIZE));
        memset(allocs[a], 0, PAGES_EACH * PAGE_SIZE);
        bases[a] = (unsigned long)allocs[a];
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc04] pid=%d allocs: 0x%lx 0x%lx 0x%lx\n",
           getpid(), bases[0], bases[1], bases[2]);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc04] start failed: %s\n", strerror(-rc)); return 1; }

    /* Write only the middle allocation. */
    gpu_write_all<<<1, 32>>>(allocs[1], PAGES_EACH, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc04] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }

    int found[NUM_ALLOCS];
    for (int a = 0; a < NUM_ALLOCS; a++)
        found[a] = count_in_alloc(e, n, bases[a], PAGES_EACH);

    printf("[tc04] dump: n=%d alloc0=%d alloc1=%d/%d alloc2=%d (want 0,%d,0)\n",
           n, found[0], found[1], PAGES_EACH, found[2], PAGES_EACH);

    stop_track();
    for (int a = 0; a < NUM_ALLOCS; a++)
        CUDA_CHECK(cudaFree(allocs[a]));

    int failed = (found[0] != 0 || found[1] != PAGES_EACH || found[2] != 0);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (found[0] != 0)        printf("[tc04]   alloc0: %d spurious pages\n", found[0]);
    if (found[1] != PAGES_EACH) printf("[tc04]   alloc1: only %d/%d pages\n", found[1], PAGES_EACH);
    if (found[2] != 0)        printf("[tc04]   alloc2: %d spurious pages\n", found[2]);
    return failed;
}
