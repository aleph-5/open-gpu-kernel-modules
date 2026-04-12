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

int main(void)
{
    printf("[tc03] stop_start_empty_table (delta mode)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Table A: write some pages, stop. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start-A failed: %s\n", strerror(-rc)); return 1; }
    printf("[tc03] table A initialized\n");

    gpu_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover-A failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n_before = read_dump(e, MAX_ENTRIES);
    printf("[tc03] table A entries before stop: %d\n", n_before);

    rc = stop_track();
    if (rc) { fprintf(stderr, "[tc03] stop-A failed: %s\n", strerror(-rc)); return 1; }
    printf("[tc03] table A destroyed\n");

    /* Table B: fresh start, NO GPU writes. */
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start-B failed: %s\n", strerror(-rc)); return 1; }
    printf("[tc03] table B initialized - no writes will follow\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover-B failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    int n_after = read_dump(e, MAX_ENTRIES);
    printf("[tc03] table B entries (no writes since start): %d\n", n_after);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    /* n_before > 0: table A recorded pages.
     * n_after == 0: table B is empty (fresh start). */
    int failed = (n_before <= 0 || n_after != 0);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (n_before <= 0) printf("[tc03]   table A was empty before stop (expected >0)\n");
    if (n_after < 0)   printf("[tc03]   table B dump failed (n=%d)\n", n_after);
    else if (n_after != 0) printf("[tc03]   table B has %d stale entries (expected 0)\n", n_after);
    return failed;
}
