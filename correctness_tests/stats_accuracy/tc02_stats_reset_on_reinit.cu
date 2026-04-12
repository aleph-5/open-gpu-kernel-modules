/*
 * tc02_stats_reset_on_reinit.cu
 *
 * After stop() + start(), stats counters must reset (not accumulate across
 * sessions). This verifies that a fresh session starts from a clean stats
 * baseline.
 *
 * Flow:
 *   enable stats
 *   session 1: start → write S1_PAGES → cutover → dump → stop
 *   read stats1 (insert count ≈ S1_PAGES)
 *   session 2: start → write S2_PAGES → cutover → dump → stop
 *   read stats2 (insert count ≈ S2_PAGES, NOT S1+S2)
 *   PASS: dump counts are correct; stats2 reflects only session 2.
 *
 * If stats are not available, the dump-count correctness still validates the
 * functional reset behaviour.
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

#define S1_PAGES  32
#define S2_PAGES  16
#define TOTAL_PAGES (S1_PAGES + S2_PAGES)
#define PAGE_SIZE  4096
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
    int p = start_page + blockIdx.x;
    if (p >= end_page) return;
    int ipp = PAGE_SIZE / sizeof(int);
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
    printf("[tc02] stats_reset_on_reinit: S1=%d S2=%d pages\n", S1_PAGES, S2_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    procfs_write_exact(PROCFS_STATS_TOGGLE, "enable");

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc02] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    /* Session 1. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start1 failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<S1_PAGES, 1>>>(managed, 0, S1_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();
    entry_t e[MAX_ENTRIES];
    int n1 = read_dump(e, MAX_ENTRIES);
    stop_track();

    long long stats1 = read_stat("insert");
    if (stats1 < 0) stats1 = read_stat("record");
    printf("[tc02] session1: dump=%d stats_inserts=%lld (want ~%d)\n", n1 < 0 ? -1 : n1, stats1, S1_PAGES);

    /* Session 2 (distinct pages). */
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start2 failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<S2_PAGES, 1>>>(managed, S1_PAGES, S1_PAGES + S2_PAGES, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();
    int n2 = read_dump(e, MAX_ENTRIES);
    stop_track();

    long long stats2 = read_stat("insert");
    if (stats2 < 0) stats2 = read_stat("record");
    printf("[tc02] session2: dump=%d stats_inserts=%lld (want ~%d, NOT %d)\n",
           n2 < 0 ? -1 : n2, stats2, S2_PAGES, S1_PAGES + S2_PAGES);

    procfs_write_exact(PROCFS_STATS_TOGGLE, "disable");
    CUDA_CHECK(cudaFree(managed));

    int dump1_ok = (n1 == S1_PAGES);
    int dump2_ok = (n2 == S2_PAGES);
    /* Stats: if available, session2 stats must be <= S2_PAGES (reset after stop). */
    int stats_ok = (stats2 < 0 || stats2 <= S2_PAGES);

    int failed = (!dump1_ok || !dump2_ok || !stats_ok);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (!dump1_ok) printf("[tc02]   session1 dump: got %d want %d\n", n1, S1_PAGES);
    if (!dump2_ok) printf("[tc02]   session2 dump: got %d want %d\n", n2, S2_PAGES);
    if (!stats_ok) printf("[tc02]   stats not reset: stats2=%lld > %d\n", stats2, S2_PAGES);
    return failed;
}
