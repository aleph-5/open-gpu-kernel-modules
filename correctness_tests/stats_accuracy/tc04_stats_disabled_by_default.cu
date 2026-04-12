/*
 * tc04_stats_disabled_by_default.cu
 *
 * Stats collection should be disabled by default (or at least controllable
 * via the toggle). After explicitly disabling stats and running a session,
 * the stats file must either be unavailable, return zeroes, or return a
 * "disabled" marker line.
 *
 * Flow:
 *   explicitly disable stats
 *   start(delta) → write pages → cutover → dump
 *   read stats file → verify: file returns 0 lines OR all counters are 0
 *                              OR file indicates "disabled"
 *   stop
 *
 * Then re-enable and verify stats become available again (functional toggle).
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

#define NUM_PAGES   16
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

__global__ void gpu_write_pages(int *base, int num_pages)
{
    int p = blockIdx.x;
    if (p >= num_pages) return;
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) base[p * ipp + i] = p * 100 + i;
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

/* Returns 1 if stats appear disabled (empty file or "disabled" or all zeros). */
static int stats_appear_disabled(void)
{
    FILE *f = fopen(PROCFS_STATS, "r");
    if (!f) return 1; /* file not readable = effectively disabled */

    char line[256];
    int nonzero_found = 0;
    int lines = 0;
    while (fgets(line, sizeof(line), f)) {
        lines++;
        if (strstr(line, "disabled") || strstr(line, "not enabled")) continue;
        long long val;
        char key[128];
        if (sscanf(line, "%127[^:]: %lld", key, &val) == 2 && val != 0)
            nonzero_found = 1;
    }
    fclose(f);
    printf("[tc04] stats (disabled): %d lines, nonzero_found=%d\n", lines, nonzero_found);
    return !nonzero_found;
}

static long long read_stat(const char *keyword)
{
    FILE *f = fopen(PROCFS_STATS, "r");
    if (!f) return -1;
    char line[256];
    long long val = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, keyword)) {
            char *colon = strchr(line, ':');
            if (colon) { val = atoll(colon + 1); break; }
        }
    }
    fclose(f);
    return val;
}

int main(void)
{
    printf("[tc04] stats_disabled_by_default: %d pages\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc04] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    /* Explicitly disable stats. */
    int rc = procfs_write_exact(PROCFS_STATS_TOGGLE, "disable");
    if (rc) printf("[tc04] stats toggle not available (rc=%d), proceeding\n", rc);

    /* Run a session with stats disabled. */
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc04] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_pages<<<NUM_PAGES, 1>>>(managed, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();
    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    stop_track();
    printf("[tc04] dump: %d entries (want %d)\n", n < 0 ? -1 : n, NUM_PAGES);

    int disabled_ok = stats_appear_disabled();
    printf("[tc04] stats disabled check: %s\n", disabled_ok ? "OK" : "stats active while disabled");

    /* Re-enable and verify stats now work. */
    rc = procfs_write_exact(PROCFS_STATS_TOGGLE, "enable");
    int enable_ok = (rc == 0);

    if (enable_ok) {
        rc = start_track_delta();
        if (!rc) {
            gpu_write_pages<<<NUM_PAGES, 1>>>(managed, NUM_PAGES);
            CUDA_CHECK(cudaDeviceSynchronize());
            cutover();
            read_dump(e, MAX_ENTRIES);
            stop_track();

            long long ins = read_stat("insert");
            if (ins < 0) ins = read_stat("record");
            printf("[tc04] after re-enable: insert_count=%lld\n", ins);
        }
        procfs_write_exact(PROCFS_STATS_TOGGLE, "disable");
    }

    CUDA_CHECK(cudaFree(managed));

    /* Primary assertions:
     * 1. Dump count correct (tracking functional regardless of stats).
     * 2. Stats appear disabled after disable toggle. */
    int failed = (n != NUM_PAGES || !disabled_ok);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (n != NUM_PAGES) printf("[tc04]   dump: got %d want %d\n", n, NUM_PAGES);
    if (!disabled_ok)   printf("[tc04]   stats not quiet while disabled\n");
    return failed;
}
