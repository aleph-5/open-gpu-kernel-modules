/*
 * tc01_post_cutover_writes_excluded.cu
 *
 * Pages written AFTER cutover() must NOT appear in the snapshot that the
 * cutover created. They should appear in the NEXT epoch's dump.
 *
 * Flow:
 *   start(delta)
 *   write phase-A pages → cutover → dump1
 *   write phase-B pages → cutover → dump2
 *   PASS:
 *     dump1 contains phase-A pages only (not phase-B)
 *     dump2 contains phase-B pages only (not phase-A — delta mode)
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

#define PHASE_PAGES  8
#define TOTAL_PAGES  (PHASE_PAGES * 2)
#define PAGE_SIZE    4096
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
        if (n < max && sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2)
            n++;
    }
    if (ferror(f)) { int saved = errno; fclose(f); return -saved; }
    fclose(f);
    return n;
}

static int count_in_range(entry_t *e, int n, unsigned long base, int start_p, int end_p)
{
    int found = 0;
    for (int p = start_p; p < end_p; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++) if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc01] post_cutover_writes_excluded: A=%d B=%d pages\n",
           PHASE_PAGES, PHASE_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc01] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    /* Phase A. */
    gpu_write_range<<<1, 32>>>(managed, 0, PHASE_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    /* Phase B — written AFTER cutover. */
    gpu_write_range<<<1, 32>>>(managed, PHASE_PAGES, TOTAL_PAGES, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc01] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }

    int a_in_1 = count_in_range(snap1, n1, base, 0, PHASE_PAGES);
    int b_in_1 = count_in_range(snap1, n1, base, PHASE_PAGES, TOTAL_PAGES);
    printf("[tc01] dump1: n=%d A=%d/%d B=%d (want A=%d B=0)\n",
           n1, a_in_1, PHASE_PAGES, b_in_1, PHASE_PAGES);

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc01] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    int a_in_2 = count_in_range(snap2, n2, base, 0, PHASE_PAGES);
    int b_in_2 = count_in_range(snap2, n2, base, PHASE_PAGES, TOTAL_PAGES);
    printf("[tc01] dump2: n=%d A=%d B=%d/%d (want A=0 B=%d)\n",
           n2, a_in_2, b_in_2, PHASE_PAGES, PHASE_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (a_in_1 != PHASE_PAGES || b_in_1 != 0 ||
                  a_in_2 != 0 || b_in_2 != PHASE_PAGES);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (a_in_1 != PHASE_PAGES) printf("[tc01]   dump1: only %d/%d phase-A pages\n", a_in_1, PHASE_PAGES);
    if (b_in_1 != 0)           printf("[tc01]   dump1: %d post-cutover pages leaked into snapshot\n", b_in_1);
    if (a_in_2 != 0)           printf("[tc01]   dump2: %d phase-A pages leaked (delta not isolated)\n", a_in_2);
    if (b_in_2 != PHASE_PAGES) printf("[tc01]   dump2: only %d/%d phase-B pages\n", b_in_2, PHASE_PAGES);
    return failed;
}
