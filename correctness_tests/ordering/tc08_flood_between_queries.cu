/*
 * tc08_flood_between_queries.cu
 *
 * Two-phase write with cross-phase timestamp ordering check (delta mode).
 *
 * Phase A: small kernel writes HALF_PAGES pages.
 *          cutover+dump1 → record max_ts_A.
 *
 * Phase B: massive kernel (1024 blocks x 256 threads = 262,144 threads)
 *          writes the second HALF_PAGES pages.
 *          cutover+dump2 → all phase-B entries must have ts >= max_ts_A.
 *
 * In delta mode, dump2 contains only phase-B pages (phase-A pages were
 * consumed by dump1).  The cross-epoch ordering invariant still holds:
 * all phase-B faults must have timestamps >= the maximum phase-A timestamp
 * because the phase-B kernel launched after cudaDeviceSynchronize of phase A.
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

#define HALF_PAGES        512
#define NUM_PAGES         (HALF_PAGES * 2)
#define PHASE_A_THREADS   32
#define PHASE_B_THREADS   256
#define PAGE_SIZE         4096
#define INTS_PER_PAGE     (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES       16384

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void phase_write(int *base, int npages, int tag)
{
    int page = blockIdx.x;
    if (page >= npages) return;
    int *p = base + page * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = tag * 100000 + page * 100 + i;
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
    printf("[tc08] flood_between_queries: phase-A=%d pages, phase-B=%d pages (delta mode)\n",
           HALF_PAGES, HALF_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *phase_a_base = managed;
    int *phase_b_base = managed + HALF_PAGES * INTS_PER_PAGE;
    unsigned long va_a = (unsigned long)phase_a_base;
    unsigned long va_b = (unsigned long)phase_b_base;
    printf("[tc08] pid=%d phase-A=0x%lx phase-B=0x%lx\n", getpid(), va_a, va_b);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc08] start failed: %s\n", strerror(-rc)); return 1; }

    /* Phase A. */
    phase_write<<<HALF_PAGES, PHASE_A_THREADS>>>(phase_a_base, HALF_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc08] phase A complete\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc08] cutover-A failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *snap_a = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    entry_t *snap_b = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!snap_a || !snap_b) { fprintf(stderr, "malloc failed\n"); return 1; }

    int na = read_dump(snap_a, MAX_ENTRIES);
    if (na < 0) { fprintf(stderr, "[tc08] dump-A failed: %s\n", strerror(-na)); stop_track(); free(snap_a); free(snap_b); return 1; }

    unsigned long max_ts_a = 0;
    int a_present = 0;
    for (int i = 0; i < na; i++) {
        if (snap_a[i].addr < va_a || snap_a[i].addr >= va_b) continue;
        a_present++;
        if (snap_a[i].ts > max_ts_a) max_ts_a = snap_a[i].ts;
    }
    printf("[tc08] phase-A: %d entries, max_ts=%lu\n", a_present, max_ts_a);

    /* Phase B: massive fault flood. */
    phase_write<<<HALF_PAGES, PHASE_B_THREADS>>>(phase_b_base, HALF_PAGES, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc08] phase B complete\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc08] cutover-B failed: %s\n", strerror(-rc)); stop_track(); free(snap_a); free(snap_b); return 1; }

    int nb = read_dump(snap_b, MAX_ENTRIES);
    if (nb < 0) { fprintf(stderr, "[tc08] dump-B failed: %s\n", strerror(-nb)); stop_track(); free(snap_a); free(snap_b); return 1; }

    int b_missing = 0, b_early = 0;
    unsigned long min_ts_b = (unsigned long)-1;

    for (int p = 0; p < HALF_PAGES; p++) {
        unsigned long pa = va_b + p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < nb; i++) {
            if (snap_b[i].addr != pa) continue;
            found = 1;
            if (snap_b[i].ts < min_ts_b) min_ts_b = snap_b[i].ts;
            if (snap_b[i].ts < max_ts_a) {
                printf("[tc08]   page %d: ts=%lu < max_ts_a=%lu (ordering violation)\n",
                       p, snap_b[i].ts, max_ts_a);
                b_early++;
            }
        }
        if (!found) b_missing++;
    }

    printf("[tc08] phase-B: %d entries, min_ts=%lu, max_ts_a=%lu\n",
           nb, min_ts_b == (unsigned long)-1 ? 0 : min_ts_b, max_ts_a);
    printf("[tc08] b_missing=%d ordering_violations=%d\n", b_missing, b_early);

    stop_track();
    CUDA_CHECK(cudaFree(managed));
    free(snap_a);
    free(snap_b);

    int failed = (a_present < HALF_PAGES || b_missing > 0 || b_early > 0);
    printf("[tc08] %s\n", failed ? "FAIL" : "PASS");
    if (a_present < HALF_PAGES) printf("[tc08]   phase-A: only %d/%d pages\n", a_present, HALF_PAGES);
    if (b_missing > 0)          printf("[tc08]   %d phase-B pages not recorded\n", b_missing);
    if (b_early > 0)            printf("[tc08]   %d phase-B entries ts < phase-A max\n", b_early);
    return failed;
}
