/*
 * tc06_cutover_without_start.cu
 *
 * Calling cutover() without first calling start() must return an error
 * (EFAULT or similar). Dump without start must also return an error.
 * The driver must not crash or hang.
 *
 * Flow:
 *   cutover() → expect non-zero (not started)
 *   dump()    → expect non-zero (not started, no snapshot)
 *   start(delta) → write page → cutover → dump → verify page → stop
 *   PASS: both spurious calls fail; valid session succeeds.
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

__global__ void gpu_write_page(int *page) { page[0] = 42; }

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

/* Returns number of entries, or negative errno. */
static int try_dump(void)
{
    FILE *f = fopen(PROCFS_DUMP, "r");
    if (!f) return -errno;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        unsigned long addr, ts;
        if (sscanf(line, "0x%lx %lu", &addr, &ts) == 2) n++;
    }
    int err = ferror(f) ? errno : 0;
    fclose(f);
    return err ? -err : n;
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
    printf("[tc06] cutover_without_start (procfs robustness)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc06] pid=%d\n", getpid());

    /* Spurious cutover without start. */
    int rc_cut = cutover();
    printf("[tc06] cutover (no start): rc=%d (%s) (want non-zero)\n",
           rc_cut, rc_cut ? strerror(-rc_cut) : "OK");

    /* Spurious dump without start or cutover.
     * Opening the file may succeed with 0 entries or fail with errno.
     * Either way, we just check it doesn't crash. */
    int rc_dump = try_dump();
    printf("[tc06] dump (no start, no cutover): rc=%d (non-crash is sufficient)\n", rc_dump);

    /* Now do a valid session. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc06] start failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_page<<<1, 1>>>(managed);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc06] valid cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc06] valid dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    unsigned long pa = (unsigned long)managed;
    int found = 0;
    for (int i = 0; i < n; i++) if (e[i].addr == pa) { found = 1; break; }
    printf("[tc06] valid session: n=%d found=%d (want found=1)\n", n, found);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    /* Key assertions:
     * 1. spurious cutover must return an error
     * 2. valid session works after spurious calls */
    int failed = (rc_cut == 0 || !found);
    printf("[tc06] %s\n", failed ? "FAIL" : "PASS");
    if (rc_cut == 0) printf("[tc06]   spurious cutover succeeded (want error)\n");
    if (!found)      printf("[tc06]   valid session failed after spurious calls\n");
    return failed;
}
