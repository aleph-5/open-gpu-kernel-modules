/*
 * tc01_alloc_after_start_tracked
 *
 * A cudaMallocManaged allocation made AFTER tracking starts must appear in
 * the dirty tracking dump when written.  The pre-existing tests all allocate
 * before start_track; this test covers the complementary path where a new VA
 * block is faulted in mid-session.
 *
 * Flow:
 *   1. start_track (delta)
 *   2. cudaMallocManaged  ← new allocation after tracking is live
 *   3. GPU write all pages
 *   4. cutover + dump
 *   5. Verify every written page appears in the dump
 *
 * Exit: 0 PASS, 1 FAIL
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
#define NUM_INTS    (NUM_PAGES * PAGE_SIZE / sizeof(int))
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

__global__ void gpu_write(int *data, int n) {
    for (int i = 0; i < n; i++) data[i] = i + 1;
}

static int procfs_write_exact(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -errno;
    ssize_t n = write(fd, val, strlen(val));
    int saved = errno;
    close(fd);
    return (n < 0) ? -saved : 0;
}

static int start_track(void)
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

static int page_tracked(entry_t *e, int n, unsigned long a)
{
    unsigned long pa = a & ~(unsigned long)(PAGE_SIZE - 1);
    for (int i = 0; i < n; i++)
        if (e[i].addr == pa) return 1;
    return 0;
}

int main(void)
{
    printf("[tc01] alloc_after_start_tracked\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    /* Warm up CUDA context before starting tracking. */
    int *warmup = NULL;
    CUDA_CHECK(cudaMallocManaged(&warmup, PAGE_SIZE));
    gpu_write<<<1, 1>>>(warmup, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(warmup));

    int rc = start_track();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    /* Allocate AFTER tracking has started. */
    int *data = NULL;
    CUDA_CHECK(cudaMallocManaged(&data, (size_t)NUM_PAGES * PAGE_SIZE));
    printf("[tc01] pid=%d alloc=0x%lx pages=%d (allocated after start)\n",
           getpid(), (unsigned long)data, NUM_PAGES);

    gpu_write<<<1, 1>>>(data, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }

    int miss = 0;
    for (int p = 0; p < NUM_PAGES; p++)
        if (!page_tracked(e, n, (unsigned long)data + p * PAGE_SIZE)) miss++;

    printf("[tc01] dump: %d entries, %d/%d pages present\n", n, NUM_PAGES - miss, NUM_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(data));

    int failed = (n < 0 || miss > 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (miss > 0)
        fprintf(stderr, "[tc01]   %d page(s) written after start not recorded\n", miss);
    return failed;
}
