/*
 * tc01_double_start.cu
 *
 * Calling start() twice without an intervening stop() must fail with a
 * well-defined error (EBUSY) rather than corrupting state or crashing.
 *
 * Flow:
 *   start(delta) → OK
 *   start(delta) again → expect non-zero return (EBUSY or similar)
 *   stop → clean up
 *   PASS: second start returns error; driver state is clean after stop.
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

__global__ void gpu_write_page(int *page) { page[0] = 42; }

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
    printf("[tc01] double_start (procfs robustness)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc01] pid=%d\n", getpid());

    /* First start — must succeed. */
    int rc1 = start_track_delta();
    if (rc1 != 0) {
        fprintf(stderr, "[tc01] FAIL: first start returned error: %s\n", strerror(-rc1));
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }
    printf("[tc01] first start: OK\n");

    /* Second start without stop — must return error. */
    int rc2 = start_track_delta();
    printf("[tc01] second start: rc=%d (%s) (want non-zero)\n",
           rc2, rc2 ? strerror(-rc2) : "OK");

    /* Clean up first session regardless. */
    stop_track();

    /* Verify driver is functional after the double-start attempt. */
    int rc3 = start_track_delta();
    if (rc3) { fprintf(stderr, "[tc01] re-start after cleanup failed: %s\n", strerror(-rc3)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_page<<<1, 1>>>(managed);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();
    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    stop_track();

    printf("[tc01] post-recovery dump: n=%d\n", n < 0 ? -1 : n);

    CUDA_CHECK(cudaFree(managed));

    int failed = (rc2 == 0); /* second start must have returned an error */
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (rc2 == 0) printf("[tc01]   double start succeeded unexpectedly (want EBUSY)\n");
    return failed;
}
