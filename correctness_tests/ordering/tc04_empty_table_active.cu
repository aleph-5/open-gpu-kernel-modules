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

__global__ void gpu_write_all(int *data, int n) {
    for (int i = 0; i < n; i++) data[i] = i + 7;
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

int main(void)
{
    printf("[tc04] empty_table_active (delta mode)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc04] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc04] start failed: %s\n", strerror(-rc)); return 1; }

    /* Cutover immediately - no GPU writes yet. */
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] pre-write cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t pre[MAX_ENTRIES];
    int n_pre = read_dump(pre, MAX_ENTRIES);
    printf("[tc04] pre-write dump: %d entries (expected 0 - active but empty)\n", n_pre);

    /* Now write and query again. */
    gpu_write_all<<<1, 256>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] post-write cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t post[MAX_ENTRIES];
    int n_post = read_dump(post, MAX_ENTRIES);
    printf("[tc04] post-write dump: %d entries (expected %d)\n", n_post, NUM_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    /* n_pre == 0  → table active but empty (correct).
     * n_pre < 0   → cutover or dump error (bug).
     * n_pre > 0   → stale entries in a fresh table (bug).
     * n_post must cover at least NUM_PAGES. */
    int failed = (n_pre != 0 || n_post < NUM_PAGES);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (n_pre < 0)          printf("[tc04]   pre-write dump failed (rc=%d)\n", n_pre);
    if (n_pre > 0)          printf("[tc04]   %d stale entries in fresh table\n", n_pre);
    if (n_post < NUM_PAGES) printf("[tc04]   only %d/%d pages captured\n", n_post, NUM_PAGES);
    return failed;
}
