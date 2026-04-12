/*
 * tc04_cumulative_stop_start_accumulation.cu
 *
 * In DELTA mode: stop + restart resets the tracking state. Pages from before
 * the stop must NOT survive into the new session.
 *
 * This is the expected behavior (the API explicitly resets on restart). The test
 * documents and validates it: pages written in session 1 must not appear in the
 * session-2 dump, even in "cumulative" mode within session 2.
 *
 * Flow:
 *   session 1 (cumulative): write pages 0..W1-1 → cutover+dump1
 *   stop
 *   session 2 (cumulative): write pages W1..W1+W2-1 → cutover+dump2
 *   stop
 *   PASS: dump2 contains only session-2 pages; session-1 pages are absent.
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

#define W1          8
#define W2          8
#define TOTAL_PAGES (W1 + W2)
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

static int start_track_cumulative(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d cumulative", getpid());
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

static int count_in_range(entry_t *e, int n, unsigned long base, int s, int end)
{
    int found = 0;
    for (int p = s; p < end; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc04] cumulative_stop_start_accumulation: W1=%d W2=%d\n", W1, W2);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc04] pid=%d alloc=0x%lx\n", getpid(), base);

    /* Session 1. */
    int rc = start_track_cumulative();
    if (rc) { fprintf(stderr, "[tc04] start1 failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, 0, W1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc04] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }
    printf("[tc04] session1 dump: %d entries\n", n1);

    rc = stop_track();
    if (rc) { fprintf(stderr, "[tc04] stop1 failed: %s\n", strerror(-rc)); return 1; }

    /* Session 2: fresh cumulative start. */
    rc = start_track_cumulative();
    if (rc) { fprintf(stderr, "[tc04] start2 failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, W1, TOTAL_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc04] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    int s1_in_snap2 = count_in_range(snap2, n2, base, 0, W1);
    int s2_in_snap2 = count_in_range(snap2, n2, base, W1, TOTAL_PAGES);
    printf("[tc04] session2 dump: n=%d session1_pages=%d/%d session2_pages=%d/%d (want 0,%d)\n",
           n2, s1_in_snap2, W1, s2_in_snap2, W2, W2);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    /* stop+restart MUST reset: session-1 pages must not survive. */
    int failed = (s1_in_snap2 != 0 || s2_in_snap2 != W2);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (s1_in_snap2 != 0) printf("[tc04]   %d session-1 pages survived stop+restart\n", s1_in_snap2);
    if (s2_in_snap2 != W2) printf("[tc04]   session-2: only %d/%d pages captured\n", s2_in_snap2, W2);
    return failed;
}
