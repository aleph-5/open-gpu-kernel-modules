/*
 * tc01_hint_seq_session.cu
 *
 * Verify that selecting WRITE_SEQ via the procfs hint (which selects the
 * sorted-vector backend, uvm_dirty_ds_vector_ops) yields a correct dump for a
 * sequential write workload.
 *
 * Flow:
 *   write hint=WRITE_SEQ → start(delta) → write every page sequentially →
 *     cutover → dump → assert every page present, addresses sorted, no extras.
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
#define PROCFS_HINT    "/proc/driver/nvidia-uvm/dirty_tracking_hint"

#define TOTAL_PAGES 128
#define PAGE_SIZE   4096
#define MAX_ENTRIES 4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_all(int *base, int total_pages)
{
    int p = blockIdx.x;
    if (p >= total_pages) return;
    int ipp = PAGE_SIZE / sizeof(int);
    int *page = base + p * ipp;
    page[0] = p;
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
    char buf[32]; snprintf(buf, sizeof(buf), "%d delta", getpid());
    return procfs_write_exact(PROCFS_START, buf);
}
static int stop_track(void)
{
    char buf[16]; snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_STOP, buf);
}
static int cutover(void)
{
    char buf[16]; snprintf(buf, sizeof(buf), "%d", getpid());
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
    printf("[tc01] hint_seq_session: %d pages, hint=WRITE_SEQ\n", TOTAL_PAGES);
    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int rc = procfs_write_exact(PROCFS_HINT, "WRITE_SEQ");
    if (rc) { fprintf(stderr, "[tc01] hint write failed: %s\n", strerror(-rc)); return 1; }

    size_t alloc_bytes = (size_t)TOTAL_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long base = (unsigned long)managed;
    printf("[tc01] pid=%d alloc=0x%lx\n", getpid(), base);

    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_all<<<TOTAL_PAGES, 1>>>(managed, TOTAL_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }
    printf("[tc01] dump: %d entries (want %d)\n", n, TOTAL_PAGES);

    int out_of_range = 0;
    int unsorted = 0;
    int missing = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr < base || e[i].addr >= base + alloc_bytes) out_of_range++;
        if (i > 0 && e[i].addr <= e[i - 1].addr) unsorted++;
    }
    for (int p = 0; p < TOTAL_PAGES; p++) {
        unsigned long expected = base + (unsigned long)p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n; i++) if (e[i].addr == expected) { found = 1; break; }
        if (!found) missing++;
    }

    stop_track();
	procfs_write_exact(PROCFS_HINT, "WRITE_SEQ"); // restore default
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != TOTAL_PAGES || out_of_range || unsorted || missing);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (n != TOTAL_PAGES) printf("[tc01]   count: got %d want %d\n", n, TOTAL_PAGES);
    if (out_of_range)     printf("[tc01]   %d out-of-range entries\n", out_of_range);
    if (unsorted)         printf("[tc01]   %d unsorted neighbour pairs\n", unsorted);
    if (missing)          printf("[tc01]   %d expected pages missing\n", missing);
    return failed;
}
