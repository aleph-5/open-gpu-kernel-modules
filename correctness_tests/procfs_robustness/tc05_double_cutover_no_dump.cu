/*
 * tc05_double_cutover_no_dump.cu
 *
 * Calling cutover() twice without an intervening dump() must return EBUSY
 * on the second call (a snapshot is already pending). The first cutover's
 * snapshot must remain intact and be readable via dump().
 *
 * Flow:
 *   start(delta) → write pages → cutover1 → (no dump)
 *   cutover2 → expect non-zero (EBUSY)
 *   dump → must return the snapshot from cutover1 (pages present)
 *   stop
 *   PASS: cutover2 fails; dump still contains cutover1's pages.
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
        for (int i = 0; i < ipp; i++)
            base[p * ipp + i] = tag * 1000 + p * 100 + i;
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
    printf("[tc05] double_cutover_no_dump (procfs robustness)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc05] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, NUM_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* First cutover — must succeed. */
    int rc1 = cutover();
    if (rc1) { fprintf(stderr, "[tc05] first cutover failed: %s\n", strerror(-rc1)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }
    printf("[tc05] cutover1: OK\n");

    /* Second cutover without dump — must fail with EBUSY. */
    int rc2 = cutover();
    printf("[tc05] cutover2 (without dump): rc=%d (%s) (want non-zero/EBUSY)\n",
           rc2, rc2 ? strerror(-rc2) : "OK");

    /* Dump must still return the cutover1 snapshot. */
    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc05] dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    int found = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++) if (e[i].addr == pa) { found++; break; }
    }
    printf("[tc05] dump after double-cutover: n=%d found=%d/%d (want %d)\n",
           n, found, NUM_PAGES, NUM_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (rc2 == 0 || found != NUM_PAGES);
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (rc2 == 0)         printf("[tc05]   second cutover succeeded (expected EBUSY)\n");
    if (found != NUM_PAGES) printf("[tc05]   only %d/%d pages in dump after double-cutover\n", found, NUM_PAGES);
    return failed;
}
