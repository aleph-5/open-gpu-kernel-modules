/*
 * tc01_cumulative_basic.cu
 *
 * Start in cumulative mode. Write W1 distinct pages in window 1, cutover+dump.
 * Write W2 distinct pages in window 2, cutover+dump. The second dump must
 * contain ALL W1+W2 pages (no duplicates, no missing).
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

static int count_duplicates(entry_t *e, int n)
{
    int dups = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (e[i].addr == e[j].addr) dups++;
    return dups;
}

int main(void)
{
    printf("[tc01] cumulative_basic: W1=%d W2=%d total=%d\n", W1, W2, TOTAL_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc01] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_cumulative();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    /* Window 1: write pages 0..W1-1. */
    gpu_write_range<<<1, 32>>>(managed, 0, W1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc01] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }

    int w1_in_snap1 = count_in_range(snap1, n1, base, 0, W1);
    int w2_in_snap1 = count_in_range(snap1, n1, base, W1, TOTAL_PAGES);
    printf("[tc01] snap1: n=%d W1=%d/%d W2=%d/%d (expect %d,0)\n",
           n1, w1_in_snap1, W1, w2_in_snap1, W2, W1);

    /* Window 2: write pages W1..TOTAL_PAGES-1. */
    gpu_write_range<<<1, 32>>>(managed, W1, TOTAL_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc01] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    int w1_in_snap2 = count_in_range(snap2, n2, base, 0, W1);
    int w2_in_snap2 = count_in_range(snap2, n2, base, W1, TOTAL_PAGES);
    int dups        = count_duplicates(snap2, n2);
    printf("[tc01] snap2: n=%d W1=%d/%d W2=%d/%d dups=%d (expect %d,%d,0)\n",
           n2, w1_in_snap2, W1, w2_in_snap2, W2, dups, W1, W2);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (w1_in_snap1 != W1 || w2_in_snap1 != 0 ||
                  w1_in_snap2 != W1 || w2_in_snap2 != W2 || dups != 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (w1_in_snap1 != W1)  printf("[tc01]   snap1 missing W1 pages (%d/%d)\n", w1_in_snap1, W1);
    if (w2_in_snap1 != 0)   printf("[tc01]   snap1 has W2 pages prematurely (%d)\n", w2_in_snap1);
    if (w1_in_snap2 != W1)  printf("[tc01]   snap2 missing W1 pages (%d/%d)\n", w1_in_snap2, W1);
    if (w2_in_snap2 != W2)  printf("[tc01]   snap2 missing W2 pages (%d/%d)\n", w2_in_snap2, W2);
    if (dups != 0)           printf("[tc01]   %d duplicate entries in snap2\n", dups);
    return failed;
}
