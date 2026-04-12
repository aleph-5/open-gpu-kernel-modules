/*
 * tc05_two_allocs_one_table.cu
 *
 * Two independent managed allocations are written under a single delta tracking
 * session. Both must appear in the same cutover+dump snapshot.
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

#define NUM_PAGES   4
#define PAGE_SIZE   4096
#define NUM_INTS    (NUM_PAGES * PAGE_SIZE / sizeof(int))
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

__global__ void gpu_write(int *data, int n) {
    for (int i = 0; i < n; i++) data[i] = i + 1;
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

static int count_missing(entry_t *e, int n, int *base, int npages)
{
    int miss = 0;
    for (int p = 0; p < npages; p++)
        if (!page_tracked(e, n, (unsigned long)base + p * PAGE_SIZE)) miss++;
    return miss;
}

int main(void)
{
    printf("[tc05] two_allocs_one_table (delta mode)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, NUM_PAGES * PAGE_SIZE));
    memset(alloc_a, 0, NUM_PAGES * PAGE_SIZE);
    memset(alloc_b, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc05] alloc_a=0x%lx alloc_b=0x%lx pages=%d each\n",
           (unsigned long)alloc_a, (unsigned long)alloc_b, NUM_PAGES);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write<<<1, 1>>>(alloc_a, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] wrote alloc_a\n");

    gpu_write<<<1, 1>>>(alloc_b, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] wrote alloc_b\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc05] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc05] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }

    int miss_a = count_missing(e, n, alloc_a, NUM_PAGES);
    int miss_b = count_missing(e, n, alloc_b, NUM_PAGES);

    printf("[tc05] total entries: %d\n", n);
    printf("[tc05] alloc_a: %d/%d pages present\n", NUM_PAGES - miss_a, NUM_PAGES);
    printf("[tc05] alloc_b: %d/%d pages present\n", NUM_PAGES - miss_b, NUM_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(alloc_a));
    CUDA_CHECK(cudaFree(alloc_b));

    int failed = (n < 0 || miss_a > 0 || miss_b > 0);
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (miss_a > 0) printf("[tc05]   alloc_a: %d pages missing\n", miss_a);
    if (miss_b > 0) printf("[tc05]   alloc_b: %d pages missing\n", miss_b);
    return failed;
}
