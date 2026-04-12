/*
 * tc01_stop_while_kernel_running.cu
 *
 * Race: call stop() while a GPU kernel is mid-execution writing pages.
 * The kernel may or may not have had its faults recorded yet.
 *
 * Safety invariant: stop() must not crash or deadlock the driver.
 * Correctness invariant: after stop(), the driver must have cleanly torn
 * down tracking state — a subsequent start+cutover+dump must return 0
 * entries (fresh empty state).
 *
 * Flow:
 *   start (delta)
 *   launch long-running kernel (writes many pages in a loop)
 *   immediately stop (without syncing the kernel)
 *   join kernel (cudaDeviceSynchronize — kernel must complete eventually)
 *   re-start (delta)
 *   cutover → dump → verify 0 entries
 *   stop
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

#define NUM_PAGES   256
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

/* Slow kernel: spins a delay loop so stop() races with active GPU faults. */
__global__ void gpu_write_slow(int *base, int num_pages, int iters_per_page)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = 0; p < num_pages; p++) {
        for (int i = 0; i < ipp; i++) base[p * ipp + i] = p * 100 + i;
        /* Spin to slow down between pages. */
        volatile int dummy = 0;
        for (int j = 0; j < iters_per_page; j++) dummy += j;
        (void)dummy;
    }
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
    printf("[tc01] stop_while_kernel_running (lifecycle race)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc01] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    /* Launch slow kernel — do NOT sync before stop. */
    gpu_write_slow<<<1, 1>>>(managed, NUM_PAGES, 10000);

    /* Race: stop while kernel is still running. */
    rc = stop_track();
    if (rc) {
        fprintf(stderr, "[tc01] stop failed (unexpected): %s\n", strerror(-rc));
        /* Still sync before exit to avoid GPU cleanup issues. */
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }
    printf("[tc01] stop returned OK (no crash/deadlock)\n");

    /* Now let the kernel finish. */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Fresh start: state must be clean after stop. */
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] re-start failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }
    printf("[tc01] fresh-session dump after race: n=%d (want 0)\n", n);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (n != 0) printf("[tc01]   got %d ghost entries after stop+restart (state not cleaned)\n", n);
    return failed;
}
