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

#define NUM_PAGES   8
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

__global__ void gpu_write_range(int *data, int ps, int pe)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = ps; p < pe; p++)
        for (int i = 0; i < ipp; i++)
            data[p * ipp + i] = p * ipp + i + 1;
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

static int page_tracked(entry_t *e, int n, unsigned long a)
{
    unsigned long pa = a & ~(unsigned long)(PAGE_SIZE - 1);
    for (int i = 0; i < n; i++)
        if (e[i].addr == pa) return 1;
    return 0;
}

int main(void)
{
    printf("[tc02] reinit_clears_entries (delta mode)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int half = NUM_PAGES / 2;
    printf("[tc02] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    /* Epoch 1: write first half. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start1 failed: %s (%d)\n", strerror(-rc), rc); return 1; }

    gpu_write_range<<<1, 100>>>(managed, 0, half);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc02] epoch1: wrote pages 0..%d\n", half - 1);

    rc = stop_track();
    if (rc) { fprintf(stderr, "[tc02] stop1 failed: %s (%d)\n", strerror(-rc), rc); return 1; }

    /* Epoch 2: fresh start, write second half. */
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start2 failed: %s (%d)\n", strerror(-rc), rc); return 1; }
    printf("[tc02] restart done - old entries should be cleared\n");

    gpu_write_range<<<1, 1>>>(managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc02] epoch2: wrote pages %d..%d\n", half, NUM_PAGES - 1);

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover failed: %s (%d)\n", strerror(-rc), rc); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] dump failed: %s (%d)\n", strerror(-n), n); stop_track(); return 1; }
    printf("[tc02] dump: %d entries\n", n);
    for (int i = 0; i < n; i++)
        printf("[tc02]   entry %d: addr=0x%lx ts=%lu\n", i, e[i].addr, e[i].ts);

    int ghost = 0, present = 0, missing = 0;
    for (int p = 0; p < half; p++)
        if (page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) ghost++;
    for (int p = half; p < NUM_PAGES; p++) {
        if (page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) present++;
        else missing++;
    }
    printf("[tc02] ghost=%d missing=%d present=%d\n", ghost, missing, present);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (ghost > 0 || missing > 0);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (ghost > 0)  printf("[tc02]   %d epoch-1 pages survived restart\n", ghost);
    if (missing > 0) printf("[tc02]   %d epoch-2 pages not recorded\n", missing);
    return failed;
}
