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

__global__ void gpu_write_range(int *base, int start_page, int end_page)
{
    int ipp = PAGE_SIZE / (int)sizeof(int);
    for (int p = start_page; p < end_page; p++) {
        for (int i = 0; i < ipp; i++) {
            base[p * ipp + i] = p * 1000 + i;
        }
    }
}

static int procfs_write_exact(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0)
        return -errno;

    ssize_t n = write(fd, val, strlen(val));
    int saved = errno;
    close(fd);

    if (n < 0)
        return -saved;

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
    if (!f)
        return -errno;

    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#')
            continue;
        if (n < max && sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2)
            n++;
    }

    if (ferror(f)) {
        int saved = errno;
        fclose(f);
        return -saved;
    }

    fclose(f);
    return n;
}

static int count_in_range(const entry_t *e, int n, unsigned long base, int start_page, int end_page)
{
    int found = 0;
    for (int p = start_page; p < end_page; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++) {
            if (e[i].addr == pa) {
                found++;
                break;
            }
        }
    }
    return found;
}

int main(void)
{
    printf("[tc01] cutover_dump_border (delta mode)\n");

    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    int half = NUM_PAGES / 2;

    printf("[tc01] pid=%d alloc=0x%lx pages=%d\n", getpid(), base, NUM_PAGES);

    int rc;

    rc = start_track_delta();
    if (rc) {
        fprintf(stderr, "[tc01] start failed: %s (%d)\n", strerror(-rc), rc);
        return 1;
    }

    gpu_write_range<<<1, 32>>>(managed, 0, half);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) {
        fprintf(stderr, "[tc01] cutover1 failed: %s (%d)\n", strerror(-rc), rc);
        stop_track();
        return 1;
    }

    gpu_write_range<<<1, 32>>>(managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) {
        fprintf(stderr, "[tc01] dump1 failed: %s (%d)\n", strerror(-n1), n1);
        stop_track();
        return 1;
    }

    int r1_in_snap1 = count_in_range(snap1, n1, base, 0, half);
    int r2_in_snap1 = count_in_range(snap1, n1, base, half, NUM_PAGES);
    printf("[tc01] dump1: n=%d round1=%d/%d round2=%d/%d (expect %d,0)\n",
           n1, r1_in_snap1, half, r2_in_snap1, half, half);

    rc = cutover();
    if (rc) {
        fprintf(stderr, "[tc01] cutover2 failed: %s (%d)\n", strerror(-rc), rc);
        stop_track();
        return 1;
    }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) {
        fprintf(stderr, "[tc01] dump2 failed: %s (%d)\n", strerror(-n2), n2);
        stop_track();
        return 1;
    }

    int r1_in_snap2 = count_in_range(snap2, n2, base, 0, half);
    int r2_in_snap2 = count_in_range(snap2, n2, base, half, NUM_PAGES);
    printf("[tc01] dump2: n=%d round1=%d/%d round2=%d/%d (expect 0,%d)\n",
           n2, r1_in_snap2, half, r2_in_snap2, half, half);

    rc = stop_track();
    if (rc) {
        fprintf(stderr, "[tc01] stop failed: %s (%d)\n", strerror(-rc), rc);
        return 1;
    }

    CUDA_CHECK(cudaFree(managed));

    int failed = (r1_in_snap1 != half) || (r2_in_snap1 != 0) || (r1_in_snap2 != 0) || (r2_in_snap2 != half);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");

    return failed;
}
