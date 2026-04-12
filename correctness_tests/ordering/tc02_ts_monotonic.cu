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

__global__ void gpu_write_page(int *base, int page_idx)
{
    int ipp = PAGE_SIZE / sizeof(int);
    int *p = base + page_idx * ipp;
    for (int i = 0; i < ipp; i++) p[i] = page_idx * 100 + i;
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

static int cmp_by_addr(const void *a, const void *b)
{
    const entry_t *ea = (const entry_t *)a;
    const entry_t *eb = (const entry_t *)b;
    if (ea->addr < eb->addr) return -1;
    if (ea->addr > eb->addr) return  1;
    return 0;
}

int main(void)
{
    printf("[tc02] ts_monotonic (%d pages, one kernel per page, delta mode)\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc02] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); return 1; }

    /* Write one page at a time, syncing after each to preserve order. */
    for (int p = 0; p < NUM_PAGES; p++) {
        gpu_write_page<<<1, 1>>>(managed, p);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("[tc02] all %d pages written sequentially\n", NUM_PAGES);

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc02] dump returned %d entries\n", n);

    int missing = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        unsigned long pa = (unsigned long)managed + p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found = 1; break; }
        if (!found) {
            printf("[tc02]   page %d (0x%lx) not in table\n", p, pa);
            missing++;
        }
    }

    /* Sort by address then check timestamps non-decreasing. */
    qsort(e, n, sizeof(entry_t), cmp_by_addr);

    int ts_violations = 0;
    for (int i = 1; i < n; i++) {
        if (e[i].ts < e[i-1].ts) {
            printf("[tc02]   ts inversion: [%d].ts=%lu < [%d].ts=%lu\n",
                   i, e[i].ts, i-1, e[i-1].ts);
            ts_violations++;
        }
    }

    printf("[tc02] missing=%d ts_violations=%d\n", missing, ts_violations);
    for (int i = 0; i < n; i++)
        printf("[tc02]   [%d] addr=0x%lx ts=%lu\n", i, e[i].addr, e[i].ts);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (missing > 0 || ts_violations > 0);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (missing > 0)       printf("[tc02]   %d pages missing\n", missing);
    if (ts_violations > 0) printf("[tc02]   %d timestamp inversions\n", ts_violations);
    return failed;
}
