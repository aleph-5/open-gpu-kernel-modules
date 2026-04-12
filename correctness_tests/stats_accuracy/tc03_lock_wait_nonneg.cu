/*
 * tc03_lock_wait_nonneg.cu
 *
 * The "lock_wait" (or "contention") stats counter must be non-negative after
 * a tracking session. This is a sanity check that the counter is properly
 * initialized and not inadvertently underflowed.
 *
 * Flow:
 *   enable stats
 *   start(delta) → write pages with concurrent streams (to maximize lock contention) → cutover → dump
 *   read stats → verify all numeric counters are >= 0
 *   stop → disable stats
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START        "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP         "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER      "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP         "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"
#define PROCFS_STATS        "/proc/driver/nvidia-uvm/dirty_ds_stats"
#define PROCFS_STATS_TOGGLE "/proc/driver/nvidia-uvm/dirty_ds_stats_toggle"

#define NUM_PAGES   32
#define NUM_STREAMS 8
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

__global__ void gpu_write_range(int *base, int start_page, int end_page, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = start_page; p < end_page; p++)
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
        if (n < max && sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2) n++;
    }
    if (ferror(f)) { int saved = errno; fclose(f); return -saved; }
    fclose(f);
    return n;
}

/* Read all "key: value" numeric stats from the stats file.
 * Returns number of counters read, prints any negative values. */
static int check_stats_nonneg(void)
{
    FILE *f = fopen(PROCFS_STATS, "r");
    if (!f) {
        printf("[tc03] stats file not available (skip counter check)\n");
        return 0;
    }
    char line[256];
    int negatives = 0;
    int total = 0;
    while (fgets(line, sizeof(line), f)) {
        char key[128];
        long long val;
        if (sscanf(line, "%127[^:]: %lld", key, &val) == 2) {
            total++;
            printf("[tc03]   stat: %-40s = %lld\n", key, val);
            if (val < 0) {
                printf("[tc03]   NEGATIVE counter: %s = %lld\n", key, val);
                negatives++;
            }
        }
    }
    fclose(f);
    printf("[tc03] stats: %d counters, %d negative\n", total, negatives);
    return negatives;
}

int main(void)
{
    printf("[tc03] lock_wait_nonneg: %d pages, %d streams\n", NUM_PAGES, NUM_STREAMS);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    procfs_write_exact(PROCFS_STATS_TOGGLE, "enable");

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc03] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start failed: %s\n", strerror(-rc)); return 1; }

    int pages_per_stream = NUM_PAGES / NUM_STREAMS;
    for (int i = 0; i < NUM_STREAMS; i++) {
        int start_p = i * pages_per_stream;
        int end_p   = start_p + pages_per_stream;
        gpu_write_range<<<1, 32, 0, streams[i]>>>(managed, start_p, end_p, i);
    }
    for (int i = 0; i < NUM_STREAMS; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc03] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc03] dump: %d entries (want %d)\n", n, NUM_PAGES);

    int neg_stats = check_stats_nonneg();

    for (int i = 0; i < NUM_STREAMS; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));

    procfs_write_exact(PROCFS_STATS_TOGGLE, "disable");
    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != NUM_PAGES || neg_stats != 0);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (n != NUM_PAGES) printf("[tc03]   dump: got %d want %d\n", n, NUM_PAGES);
    if (neg_stats)      printf("[tc03]   %d negative stat counters (overflow/underflow)\n", neg_stats);
    return failed;
}
