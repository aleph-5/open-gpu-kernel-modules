#include <cuda_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_PAUSE   "/proc/driver/nvidia-uvm/dirty_tracking_pause"
#define PROCFS_RESUME  "/proc/driver/nvidia-uvm/dirty_tracking_resume"
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

static int pause_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_PAUSE, buf);
}

static int resume_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_RESUME, buf);
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
    printf("[tc02] pause_resume_semantics (delta mode)\n");

    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }

    const int half = NUM_PAGES / 2;

    int *managed1 = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed1, NUM_PAGES * PAGE_SIZE));
    memset(managed1, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base1 = (unsigned long)managed1;
    printf("[tc02] pid=%d alloc1=0x%lx\n", getpid(), base1);

    int rc = start_track_delta();
    if (rc) {
        fprintf(stderr, "[tc02] start failed: %s (%d)\n", strerror(-rc), rc);
        return 1;
    }

    // Phase A: active -> writes should be recorded.
    gpu_write_range<<<1, 32>>>(managed1, 0, half);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) {
        fprintf(stderr, "[tc02] cutover(active) failed: %s (%d)\n", strerror(-rc), rc);
        stop_track();
        return 1;
    }

    entry_t snap_a[MAX_ENTRIES];
    int na = read_dump(snap_a, MAX_ENTRIES);
    if (na < 0) {
        fprintf(stderr, "[tc02] dump(active) failed: %s (%d)\n", strerror(-na), na);
        stop_track();
        return 1;
    }

    int a_present = count_in_range(snap_a, na, base1, 0, half);
    printf("[tc02] active dump: n=%d present=%d/%d (expect %d)\n", na, a_present, half, half);

    // Phase B: pause -> writes should NOT be recorded.
    rc = pause_track();
    if (rc) {
        fprintf(stderr, "[tc02] pause failed: %s (%d)\n", strerror(-rc), rc);
        stop_track();
        return 1;
    }

    gpu_write_range<<<1, 32>>>(managed1, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) {
        fprintf(stderr, "[tc02] cutover(paused) failed: %s (%d)\n", strerror(-rc), rc);
        stop_track();
        return 1;
    }

    entry_t snap_p[MAX_ENTRIES];
    int np = read_dump(snap_p, MAX_ENTRIES);
    if (np < 0) {
        fprintf(stderr, "[tc02] dump(paused) failed: %s (%d)\n", strerror(-np), np);
        stop_track();
        return 1;
    }

    int p_present = count_in_range(snap_p, np, base1, half, NUM_PAGES);
    printf("[tc02] paused dump: n=%d present=%d/%d (expect 0)\n", np, p_present, half);

    // Phase C: resume -> new GPU writes should be recorded again.
    rc = resume_track();
    if (rc) {
        fprintf(stderr, "[tc02] resume failed: %s (%d)\n", strerror(-rc), rc);
        stop_track();
        return 1;
    }

    int *managed2 = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed2, NUM_PAGES * PAGE_SIZE));
    memset(managed2, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base2 = (unsigned long)managed2;
    printf("[tc02] alloc2=0x%lx\n", base2);

    gpu_write_range<<<1, 32>>>(managed2, 0, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) {
        fprintf(stderr, "[tc02] cutover(resumed) failed: %s (%d)\n", strerror(-rc), rc);
        stop_track();
        return 1;
    }

    entry_t snap_r[MAX_ENTRIES];
    int nr = read_dump(snap_r, MAX_ENTRIES);
    if (nr < 0) {
        fprintf(stderr, "[tc02] dump(resumed) failed: %s (%d)\n", strerror(-nr), nr);
        stop_track();
        return 1;
    }

    int r_present = count_in_range(snap_r, nr, base2, 0, NUM_PAGES);
    printf("[tc02] resumed dump: n=%d present=%d/%d (expect %d)\n", nr, r_present, NUM_PAGES, NUM_PAGES);

    rc = stop_track();
    if (rc) {
        fprintf(stderr, "[tc02] stop failed: %s (%d)\n", strerror(-rc), rc);
        return 1;
    }

    CUDA_CHECK(cudaFree(managed1));
    CUDA_CHECK(cudaFree(managed2));

    int failed = (a_present != half) || (p_present != 0) || (r_present != NUM_PAGES);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");

    return failed;
}
