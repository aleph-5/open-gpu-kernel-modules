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

#define NUM_STREAMS      4
#define PAGES_PER_STREAM 64
#define NUM_PAGES        (NUM_STREAMS * PAGES_PER_STREAM)
#define THREADS_PER_BLOCK 256
#define PAGE_SIZE        4096
#define INTS_PER_PAGE    (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES      8192

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void stream_write(int *base, int npages)
{
    int page = blockIdx.x;
    if (page >= npages) return;
    int *p = base + page * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = page * 500 + i + 1;
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
    printf("[tc06] concurrent_streams: %d streams x %d pages (delta mode)\n",
           NUM_STREAMS, PAGES_PER_STREAM);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc06] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc06] start failed: %s\n", strerror(-rc)); return 1; }

    for (int s = 0; s < NUM_STREAMS; s++) {
        int *base = managed + s * PAGES_PER_STREAM * INTS_PER_PAGE;
        stream_write<<<PAGES_PER_STREAM, THREADS_PER_BLOCK, 0, streams[s]>>>(base, PAGES_PER_STREAM);
    }
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    printf("[tc06] all streams complete\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc06] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc06] dump failed: %s\n", strerror(-n)); stop_track(); free(e); return 1; }
    printf("[tc06] dump returned %d entries (expected %d)\n", n, NUM_PAGES);

    int total_missing = 0;
    for (int s = 0; s < NUM_STREAMS; s++) {
        unsigned long stream_base = (unsigned long)managed + s * PAGES_PER_STREAM * PAGE_SIZE;
        int page_found[PAGES_PER_STREAM];
        memset(page_found, 0, sizeof(page_found));

        for (int i = 0; i < n; i++) {
            if (e[i].addr < stream_base) continue;
            unsigned long off = e[i].addr - stream_base;
            if (off >= (unsigned long)PAGES_PER_STREAM * PAGE_SIZE) continue;
            int pg = (int)(off / PAGE_SIZE);
            page_found[pg] = 1;
        }

        int missing = 0;
        for (int pg = 0; pg < PAGES_PER_STREAM; pg++)
            if (!page_found[pg]) missing++;
        printf("[tc06] stream %d: %d/%d pages captured\n",
               s, PAGES_PER_STREAM - missing, PAGES_PER_STREAM);
        total_missing += missing;
    }

    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));

    stop_track();
    CUDA_CHECK(cudaFree(managed));
    free(e);

    int failed = (total_missing > 0);
    printf("[tc06] %s\n", failed ? "FAIL" : "PASS");
    if (total_missing > 0) printf("[tc06]   %d pages missing across all streams\n", total_missing);
    return failed;
}
