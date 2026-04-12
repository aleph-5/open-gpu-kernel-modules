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

#define NUM_PAGES   100
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

int main(void)
{
    printf("[tc01] basic_lifecycle (delta mode)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc01] pid=%d alloc=0x%lx pages=%d\n",
           getpid(), (unsigned long)managed, NUM_PAGES);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s (%d)\n", strerror(-rc), rc); return 1; }

    gpu_write<<<1, 100>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s (%d)\n", strerror(-rc), rc); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s (%d)\n", strerror(-n), n); stop_track(); return 1; }

    int miss = 0;
    for (int p = 0; p < NUM_PAGES; p++)
        if (!page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) miss++;
    printf("[tc01] dump: %d entries, %d/%d pages missing\n", n, miss, NUM_PAGES);

    rc = stop_track();
    if (rc) { fprintf(stderr, "[tc01] stop failed: %s (%d)\n", strerror(-rc), rc); return 1; }
    printf("[tc01] table destroyed\n");

    /* After stop, cutover must fail (table not started). */
    int post_stop_rc = cutover();
    printf("[tc01] post-stop cutover rc=%d (expected non-zero)\n", post_stop_rc);

    CUDA_CHECK(cudaFree(managed));

    int failed = (n < 0 || miss > 0 || post_stop_rc == 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (n < 0)            printf("[tc01]   dump read failed\n");
    if (miss > 0)         printf("[tc01]   %d pages not recorded\n", miss);
    if (post_stop_rc == 0) printf("[tc01]   cutover succeeded after stop (should have failed)\n");
    return failed;
}
