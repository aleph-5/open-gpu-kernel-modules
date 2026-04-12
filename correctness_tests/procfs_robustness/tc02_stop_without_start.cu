/*
 * tc02_stop_without_start.cu
 *
 * Calling stop() when tracking has not been started must return an error
 * (not crash or hang). After the spurious stop(), a real start+write+cutover+
 * dump session must succeed normally.
 *
 * Flow:
 *   stop() → expect non-zero (not started)
 *   start(delta) → write page → cutover → dump → verify page present → stop
 *   PASS: spurious stop returns error; subsequent session works.
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
    printf("[tc02] stop_without_start (procfs robustness)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc02] pid=%d\n", getpid());

    /* Spurious stop — no active session. */
    int rc_stop = stop_track();
    printf("[tc02] spurious stop: rc=%d (%s) (want non-zero)\n",
           rc_stop, rc_stop ? strerror(-rc_stop) : "OK");

    /* Now do a valid session. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_page<<<1, 1>>>(managed);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    unsigned long pa = (unsigned long)managed;
    int found = 0;
    for (int i = 0; i < n; i++) if (e[i].addr == pa) { found = 1; break; }
    printf("[tc02] post-spurious-stop session: n=%d found=%d (want found=1)\n", n, found);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (rc_stop == 0 || !found);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (rc_stop == 0) printf("[tc02]   spurious stop succeeded unexpectedly (want error)\n");
    if (!found)       printf("[tc02]   page not found after spurious stop + fresh session\n");
    return failed;
}
