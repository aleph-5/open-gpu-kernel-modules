/*
 * tc08_empty_epoch_between_writes.cu
 *
 * An epoch with no writes must produce an empty dump (0 entries). Intercalate
 * empty epochs between write epochs and verify isolation is maintained.
 *
 * Flow (delta mode):
 *   start(delta)
 *   epoch 1: write pages → cutover → dump1 (N entries)
 *   epoch 2: no writes  → cutover → dump2 (0 entries)
 *   epoch 3: write pages → cutover → dump3 (N entries)
 *   PASS: dump1=N, dump2=0, dump3=N
 *
 * Also verifies that pages from epoch 1 don't bleed into epoch 2's empty dump.
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

__global__ void gpu_write_range(int *base, int num_pages, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = 0; p < num_pages; p++)
        for (int i = 0; i < ipp; i++) base[p * ipp + i] = tag * 1000 + p * 100 + i;
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
    printf("[tc08] empty_epoch_between_writes: %d pages\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc08] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc08] start failed: %s\n", strerror(-rc)); return 1; }

    /* Epoch 1: write. */
    gpu_write_range<<<1, 32>>>(managed, NUM_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc08] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc08] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }
    printf("[tc08] epoch1 (write): n=%d (want %d)\n", n1, NUM_PAGES);

    /* Epoch 2: no writes. */
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc08] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc08] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }
    printf("[tc08] epoch2 (no-write): n=%d (want 0)\n", n2);

    /* Epoch 3: write same pages again (PTEs re-downgraded by prior cutover). */
    gpu_write_range<<<1, 32>>>(managed, NUM_PAGES, 3);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc08] cutover3 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap3[MAX_ENTRIES];
    int n3 = read_dump(snap3, MAX_ENTRIES);
    if (n3 < 0) { fprintf(stderr, "[tc08] dump3 failed: %s\n", strerror(-n3)); stop_track(); return 1; }
    printf("[tc08] epoch3 (write): n=%d (want %d)\n", n3, NUM_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (n1 != NUM_PAGES || n2 != 0 || n3 != NUM_PAGES);
    printf("[tc08] %s\n", failed ? "FAIL" : "PASS");
    if (n1 != NUM_PAGES) printf("[tc08]   epoch1: got %d want %d\n", n1, NUM_PAGES);
    if (n2 != 0)         printf("[tc08]   epoch2: got %d (should be 0 — empty epoch not isolated)\n", n2);
    if (n3 != NUM_PAGES) printf("[tc08]   epoch3: got %d want %d\n", n3, NUM_PAGES);
    return failed;
}
