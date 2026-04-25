/*
 * tc03_hint_switch_between_sessions.cu
 *
 * Verify that the hint can be switched between independent tracking sessions
 * and that each session uses the backend selected by the hint that was in
 * effect at start time.
 *
 * Flow:
 *   hint=WRITE_SEQ → start → write 64 sequential pages → cutover → dump A
 *   stop
 *   hint=WRITE_RAND → start → write 64 different pages → cutover → dump B
 *   stop
 *   Assert:
 *     - dump A contains exactly the 64 sequentially-written pages
 *     - dump B contains exactly the 64 second-session pages and none of A's
 *       pages (i.e. session-B's ds was freshly initialized after the swap)
 *     - both dumps are sorted by address
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
#define PROCFS_HINT    "/proc/driver/nvidia-uvm/dirty_tracking_hint"

#define TOTAL_PAGES 128
#define HALF        (TOTAL_PAGES / 2)
#define PAGE_SIZE   4096
#define MAX_ENTRIES 4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_range(int *base, int start, int count)
{
    int t = blockIdx.x;
    if (t >= count) return;
    int p = start + t;
    int ipp = PAGE_SIZE / sizeof(int);
    base[p * ipp] = p;
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
    char buf[32]; snprintf(buf, sizeof(buf), "%d delta", getpid());
    return procfs_write_exact(PROCFS_START, buf);
}
static int stop_track(void)
{
    char buf[16]; snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_STOP, buf);
}
static int cutover(void)
{
    char buf[16]; snprintf(buf, sizeof(buf), "%d", getpid());
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

/*
 * Validate that `e[0..n)` contains exactly the pages [start, start+count) of
 * the allocation at `base`, sorted by address, with no duplicates or extras.
 * Returns 0 on success, nonzero on failure.
 */
static int verify_range(const entry_t *e, int n, unsigned long base,
                        int start, int count, const char *label)
{
    int failed = 0;
    if (n != count) {
        printf("[tc03]   %s: count: got %d want %d\n", label, n, count);
        failed = 1;
    }
    int unsorted = 0, dup = 0;
    for (int i = 1; i < n; i++) {
        if (e[i].addr < e[i - 1].addr) unsorted++;
        if (e[i].addr == e[i - 1].addr) dup++;
    }
    if (unsorted) { printf("[tc03]   %s: %d unsorted neighbour pairs\n", label, unsorted); failed = 1; }
    if (dup)      { printf("[tc03]   %s: %d duplicate addresses\n", label, dup); failed = 1; }

    int unexpected = 0, missing = 0;
    for (int i = 0; i < n; i++) {
        unsigned long off = e[i].addr - base;
        unsigned long p   = off / PAGE_SIZE;
        if ((long)p < start || (long)p >= start + count) unexpected++;
    }
    for (int p = start; p < start + count; p++) {
        unsigned long want = base + (unsigned long)p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n; i++) if (e[i].addr == want) { found = 1; break; }
        if (!found) missing++;
    }
    if (unexpected) { printf("[tc03]   %s: %d unexpected entries (outside [%d, %d))\n",
                              label, unexpected, start, start + count); failed = 1; }
    if (missing)    { printf("[tc03]   %s: %d expected pages missing\n", label, missing); failed = 1; }
    return failed;
}

int main(void)
{
    printf("[tc03] hint_switch_between_sessions\n");
    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    size_t alloc_bytes = (size_t)TOTAL_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long base = (unsigned long)managed;
    printf("[tc03] pid=%d alloc=0x%lx\n", getpid(), base);

    int failed = 0;

    /* --- Session A: WRITE_SEQ, write pages [0, HALF). --- */
    int rc = procfs_write_exact(PROCFS_HINT, "WRITE_SEQ");
    if (rc) { fprintf(stderr, "[tc03] hint A failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start A failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }
    gpu_write_range<<<HALF, 1>>>(managed, 0, HALF);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();
    entry_t a[MAX_ENTRIES];
    int na = read_dump(a, MAX_ENTRIES);
    printf("[tc03] session A (WRITE_SEQ): %d entries (want %d)\n", na, HALF);
    if (na < 0) { stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }
    failed |= verify_range(a, na, base, 0, HALF, "session A");
    stop_track();

    /* --- Session B: WRITE_RAND, write pages [HALF, TOTAL_PAGES). --- */
    rc = procfs_write_exact(PROCFS_HINT, "WRITE_RAND");
    if (rc) { fprintf(stderr, "[tc03] hint B failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start B failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }
    gpu_write_range<<<HALF, 1>>>(managed, HALF, HALF);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();
    entry_t b[MAX_ENTRIES];
    int nb = read_dump(b, MAX_ENTRIES);
    printf("[tc03] session B (WRITE_RAND): %d entries (want %d)\n", nb, HALF);
    if (nb < 0) { stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }
    failed |= verify_range(b, nb, base, HALF, HALF, "session B");
    stop_track();

    /* Cross-session check: B must not contain any of A's pages. */
    int crossover = 0;
    for (int i = 0; i < nb; i++) {
        unsigned long p = (b[i].addr - base) / PAGE_SIZE;
        if ((long)p < HALF) crossover++;
    }
    if (crossover) {
        printf("[tc03]   session B contains %d page(s) from session A — backend was not freshly initialized\n",
               crossover);
        failed = 1;
    }

    procfs_write_exact(PROCFS_HINT, "WRITE_SEQ"); /* restore default */
    CUDA_CHECK(cudaFree(managed));

    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
