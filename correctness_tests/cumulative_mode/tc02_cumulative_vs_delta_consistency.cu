/*
 * tc02_cumulative_vs_delta_consistency.cu
 *
 * Run delta mode and cumulative mode over identical write patterns.
 * After two windows of writes, the cumulative dump must be a superset of
 * each individual delta dump (delta W1 ⊆ cumulative, delta W2 ⊆ cumulative).
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

#define NUM_PAGES   16
#define HALF        (NUM_PAGES / 2)
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

__global__ void gpu_write_range(int *base, int start_page, int end_page)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = start_page; p < end_page; p++)
        for (int i = 0; i < ipp; i++)
            base[p * ipp + i] = p * 100 + i;
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

static int start_track(const char *mode)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d %s", getpid(), mode);
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

static int addr_in_set(unsigned long addr, entry_t *e, int n)
{
    for (int i = 0; i < n; i++)
        if (e[i].addr == addr) return 1;
    return 0;
}

/* Count entries from 'sub' that are NOT present in 'super_'. */
static int count_not_in_superset(entry_t *sub, int sub_n, entry_t *super_, int super_n)
{
    int missing = 0;
    for (int i = 0; i < sub_n; i++)
        if (!addr_in_set(sub[i].addr, super_, super_n)) missing++;
    return missing;
}

int main(void)
{
    printf("[tc02] cumulative_vs_delta_consistency: %d pages, half=%d\n", NUM_PAGES, HALF);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc02] pid=%d alloc=0x%lx\n", getpid(), base);

    /* --- Delta run: collect delta_w1 and delta_w2. --- */
    int rc = start_track("delta");
    if (rc) { fprintf(stderr, "[tc02] start-delta failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, 0, HALF);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover-d1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t delta_w1[MAX_ENTRIES];
    int nd1 = read_dump(delta_w1, MAX_ENTRIES);
    if (nd1 < 0) { fprintf(stderr, "[tc02] dump-d1 failed\n"); stop_track(); return 1; }

    gpu_write_range<<<1, 32>>>(managed, HALF, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover-d2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t delta_w2[MAX_ENTRIES];
    int nd2 = read_dump(delta_w2, MAX_ENTRIES);
    if (nd2 < 0) { fprintf(stderr, "[tc02] dump-d2 failed\n"); stop_track(); return 1; }

    stop_track();
    printf("[tc02] delta: W1=%d entries, W2=%d entries\n", nd1, nd2);

    /* Reset for cumulative run. */
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* --- Cumulative run. --- */
    rc = start_track("cumulative");
    if (rc) { fprintf(stderr, "[tc02] start-cumulative failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, 0, HALF);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover-c1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t cum_snap1[MAX_ENTRIES];
    int nc1 = read_dump(cum_snap1, MAX_ENTRIES);
    if (nc1 < 0) { fprintf(stderr, "[tc02] dump-c1 failed\n"); stop_track(); return 1; }

    gpu_write_range<<<1, 32>>>(managed, HALF, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover-c2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t cum_snap2[MAX_ENTRIES];
    int nc2 = read_dump(cum_snap2, MAX_ENTRIES);
    if (nc2 < 0) { fprintf(stderr, "[tc02] dump-c2 failed\n"); stop_track(); return 1; }

    stop_track();
    printf("[tc02] cumulative: snap1=%d entries, snap2=%d entries\n", nc1, nc2);

    CUDA_CHECK(cudaFree(managed));

    /* Verify set containment: delta_w1 ⊆ cum_snap2, delta_w2 ⊆ cum_snap2. */
    int missing_w1 = count_not_in_superset(delta_w1, nd1, cum_snap2, nc2);
    int missing_w2 = count_not_in_superset(delta_w2, nd2, cum_snap2, nc2);
    printf("[tc02] delta_w1 ⊆ cum_snap2: missing=%d (want 0)\n", missing_w1);
    printf("[tc02] delta_w2 ⊆ cum_snap2: missing=%d (want 0)\n", missing_w2);

    int failed = (missing_w1 != 0 || missing_w2 != 0);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (missing_w1) printf("[tc02]   %d delta-W1 pages not in cumulative\n", missing_w1);
    if (missing_w2) printf("[tc02]   %d delta-W2 pages not in cumulative\n", missing_w2);
    return failed;
}
