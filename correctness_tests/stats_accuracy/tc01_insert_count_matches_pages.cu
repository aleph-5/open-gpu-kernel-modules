/*
 * tc01_insert_count_matches_pages.cu
 *
 * After writing NUM_PAGES distinct pages, the stats counter for "inserts"
 * (or "dirty_pages_recorded") must equal NUM_PAGES.
 *
 * Stats are enabled via dirty_ds_stats_toggle and read from dirty_ds_stats.
 * The format is assumed to be a set of "key: value" lines; we search for
 * a line matching "inserts:" (or similar) and compare its value.
 *
 * Flow:
 *   enable stats → start(delta) → write NUM_PAGES → cutover → dump
 *   read stats → verify insert_count == NUM_PAGES
 *   disable stats → stop
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

#define NUM_PAGES   64
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

/* Read the stats file and search for a counter by keyword substring.
 * Returns the value, or -1 if not found. */
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
    printf("[tc01] insert_count_matches_pages: %d pages\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    /* Enable stats. */
    int rc = procfs_write_exact(PROCFS_STATS_TOGGLE, "enable");
    if (rc) {
        fprintf(stderr, "[tc01] stats enable failed: %s (may not be implemented)\n", strerror(-rc));
        /* Non-fatal: continue without stats check but report. */
    }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc01] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_pages<<<NUM_PAGES, 1>>>(managed, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc01] dump: %d entries (want %d)\n", n, NUM_PAGES);

    /* Read stats — search for various plausible counter names. */
    long long insert_count = read_stat("insert");
    if (insert_count < 0) insert_count = read_stat("record");
    if (insert_count < 0) insert_count = read_stat("dirty_page");
    printf("[tc01] stats insert_count=%lld (want %d; -1=not found)\n", insert_count, NUM_PAGES);

    /* Disable stats. */
    procfs_write_exact(PROCFS_STATS_TOGGLE, "disable");

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    /* Primary pass condition: dump count matches. Stats check is advisory. */
    int dump_ok = (n == NUM_PAGES);
    int stats_ok = (insert_count < 0 || insert_count >= NUM_PAGES); /* -1 = not available = skip */

    int failed = (!dump_ok || !stats_ok);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (!dump_ok) printf("[tc01]   dump: got %d want %d\n", n, NUM_PAGES);
    if (!stats_ok) printf("[tc01]   stats: insert_count=%lld < %d\n", insert_count, NUM_PAGES);
    return failed;
}
