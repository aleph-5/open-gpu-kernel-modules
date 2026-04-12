/*
 * tc02_rapid_start_stop_cycle.cu
 *
 * Stress test: rapidly cycle through start→stop N times without any GPU work.
 * Verifies that the driver does not leak resources, deadlock, or corrupt state
 * across repeated lifecycle transitions.
 *
 * After all cycles, do one final functional session to verify the driver is
 * still usable:
 *   start → write page → cutover → dump → stop
 *   PASS: page appears in dump with non-zero timestamp.
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

#define NUM_CYCLES  64
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

__global__ void gpu_write_page(int *page, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) page[i] = tag * 1000 + i;
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
    printf("[tc02] rapid_start_stop_cycle: %d cycles\n", NUM_CYCLES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc02] pid=%d page=0x%lx\n", getpid(), (unsigned long)managed);

    /* Rapid cycle: start → stop, no GPU work. */
    for (int c = 0; c < NUM_CYCLES; c++) {
        int rc = start_track_delta();
        if (rc) {
            fprintf(stderr, "[tc02] cycle %d: start failed: %s\n", c, strerror(-rc));
            CUDA_CHECK(cudaFree(managed));
            return 1;
        }
        rc = stop_track();
        if (rc) {
            fprintf(stderr, "[tc02] cycle %d: stop failed: %s\n", c, strerror(-rc));
            CUDA_CHECK(cudaFree(managed));
            return 1;
        }
    }
    printf("[tc02] %d start/stop cycles completed without error\n", NUM_CYCLES);

    /* Verify driver is still functional after all cycles. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] final start failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_page<<<1, 1>>>(managed, 99);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] final cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] final dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    unsigned long pa_P = (unsigned long)managed;
    int found = 0;
    unsigned long ts = 0;
    for (int i = 0; i < n; i++)
        if (e[i].addr == pa_P) { found = 1; ts = e[i].ts; break; }

    printf("[tc02] final session: n=%d found=%d ts=%lu (want found=1 ts>0)\n", n, found, ts);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (!found || ts == 0);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (!found) printf("[tc02]   page not recorded in final session after %d cycles\n", NUM_CYCLES);
    if (ts == 0) printf("[tc02]   zero timestamp in final session\n");
    return failed;
}
