/*
 * tc02_concurrent_same_page.cu
 *
 * 64 CUDA streams all write the same single page simultaneously.
 * The first-write-wins invariant means exactly 1 entry must appear in the
 * dump (regardless of which stream "wins" the race to record).
 *
 * Flow:
 *   start (delta)
 *   launch 64 kernels on 64 streams, each writing page P
 *   sync all streams
 *   cutover → dump
 *   PASS: dump contains P exactly once (count == 1, ts > 0)
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

#define PAGE_SIZE    4096
#define NUM_STREAMS  64
#define MAX_ENTRIES  4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_page(int *page, int stream_id)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) page[i] = stream_id * 1000 + i;
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
    printf("[tc02] concurrent_same_page: %d streams → single page\n", NUM_STREAMS);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long pa_P = (unsigned long)managed;
    printf("[tc02] pid=%d P=0x%lx\n", getpid(), pa_P);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); return 1; }

    /* All streams write the same page concurrently. */
    for (int i = 0; i < NUM_STREAMS; i++)
        gpu_write_page<<<1, 1, 0, streams[i]>>>(managed, i);

    for (int i = 0; i < NUM_STREAMS; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }

    int p_count = 0;
    unsigned long p_ts = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr == pa_P) { p_count++; p_ts = e[i].ts; }
    }
    printf("[tc02] dump: n=%d, P_count=%d P_ts=%lu (want count=1)\n",
           n, p_count, p_ts);

    stop_track();
    for (int i = 0; i < NUM_STREAMS; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(managed));

    int failed = (p_count != 1 || p_ts == 0);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (p_count == 0) printf("[tc02]   P not recorded at all\n");
    if (p_count > 1)  printf("[tc02]   P appears %d times (expected 1, first-write-wins broken)\n", p_count);
    if (p_ts == 0)    printf("[tc02]   P has zero timestamp\n");
    return failed;
}
