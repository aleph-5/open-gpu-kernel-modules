/*
 * tc03_address_monotonic_in_dump.cu
 *
 * Some backend implementations (bitmap, sorted xarray) may output entries in
 * address order. This test checks whether the dump is monotonically increasing
 * by address. It does NOT fail if addresses are unordered — that is also valid
 * — but it REPORTS the ordering so the developer knows the backend behaviour.
 *
 * What DOES cause failure: any duplicate address (even if ordering is mixed).
 *
 * Flow:
 *   write pages in REVERSE order (highest addr first) via kernel
 *   cutover → dump
 *   check: no dups, report whether output is sorted ascending
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

#define NUM_PAGES   64
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

/* Write pages in reverse order: block 0 writes the last page, etc. */
__global__ void gpu_write_reverse(int *base, int num_pages)
{
    int p = num_pages - 1 - blockIdx.x;
    if (p < 0) return;
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
    printf("[tc03] address_monotonic_in_dump: %d pages (written in reverse)\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    size_t alloc_bytes = (size_t)NUM_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc03] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_reverse<<<NUM_PAGES, 1>>>(managed, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc03] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc03] dump: %d entries (want %d)\n", n, NUM_PAGES);

    /* Check duplicates. */
    int dups = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (e[i].addr == e[j].addr) { dups++; break; }

    /* Check monotonic ordering (informational). */
    int out_of_order = 0;
    for (int i = 1; i < n; i++)
        if (e[i].addr < e[i-1].addr) out_of_order++;

    /* Check all pages present. */
    int missing = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n; i++) if (e[i].addr == pa) { found = 1; break; }
        if (!found) missing++;
    }

    printf("[tc03] dups=%d out_of_order=%d/%d missing=%d\n",
           dups, out_of_order, n-1, missing);
    if (out_of_order == 0)
        printf("[tc03] dump is monotonically ascending (sorted backend)\n");
    else
        printf("[tc03] dump is NOT monotonically sorted (%d inversions) — informational only\n",
               out_of_order);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    /* Only duplicate addresses and missing pages are hard failures. */
    int failed = (n != NUM_PAGES || dups != 0 || missing != 0);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (n != NUM_PAGES) printf("[tc03]   count: got %d want %d\n", n, NUM_PAGES);
    if (dups)           printf("[tc03]   %d duplicate addresses\n", dups);
    if (missing)        printf("[tc03]   %d pages missing from dump\n", missing);
    return failed;
}
