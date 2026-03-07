#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

#define PROCFS_START  "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP   "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY  "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_PAGES  "/proc/driver/nvidia-uvm/dirty_pages"

/* Two-phase write with a cross-phase timestamp ordering check.
 *
 * Phase A: small kernel writes the first HALF_PAGES pages.
 *          Query → record max_ts_A (highest timestamp seen in phase A).
 *
 * Phase B: massive kernel — 1024 blocks x 256 threads = 262,144 threads —
 *          writes the second HALF_PAGES pages.
 *          Query → check every phase-B entry has ts >= max_ts_A.
 *
 * Ordering invariant:
 *   All phase-B faults happen AFTER phase-A completes (cudaDeviceSynchronize
 *   separates the two kernels).  So every phase-B timestamp must be >=
 *   the maximum phase-A timestamp.  Any phase-B entry with ts < max_ts_A
 *   is an ordering violation — the tracker recorded a later fault with an
 *   earlier timestamp. */

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

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void phase_write(int *base, int npages, int tag) {
    int page = blockIdx.x;
    if (page >= npages) return;
    int *p = base + page * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = tag * 100000 + page * 100 + i;
}

static void procfs_write(const char *path, const char *val) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    write(fd, val, strlen(val));
    close(fd);
}

static void set_query_pid(pid_t p) {
    char b[32];
    snprintf(b, sizeof(b), "%d\n", p);
    procfs_write(PROCFS_QUERY, b);
}

static int read_pages(entry_t *out, int max) {
    FILE *f = fopen(PROCFS_PAGES, "r");
    if (!f) return -1;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) { fclose(f); return -2; }
            continue;
        }
        if (n < max &&
            sscanf(line, "0x%lx %lu %d", &out[n].addr, &out[n].ts, &out[n].pid) == 3)
            n++;
    }
    fclose(f);
    return n;
}

int main(void) {
    printf("[tc08] flood_between_queries: phase-A=%d pages/%d threads, phase-B=%d pages/%d threads\n",
           HALF_PAGES, HALF_PAGES * PHASE_A_THREADS,
           HALF_PAGES, HALF_PAGES * PHASE_B_THREADS);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *phase_a_base = managed;
    int *phase_b_base = managed + HALF_PAGES * INTS_PER_PAGE;
    unsigned long va_a = (unsigned long)phase_a_base;
    unsigned long va_b = (unsigned long)phase_b_base;

    pid_t pid = getpid();
    printf("[tc08] pid=%d  phase-A=0x%lx  phase-B=0x%lx\n", pid, va_a, va_b);
    set_query_pid(pid);

    procfs_write(PROCFS_START, "1\n");

    /* phase A: small quiet kernel */
    phase_write<<<HALF_PAGES, PHASE_A_THREADS>>>(phase_a_base, HALF_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc08] phase A complete\n");

    entry_t *snap_a = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    entry_t *snap_b = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!snap_a || !snap_b) { fprintf(stderr, "malloc failed\n"); return 1; }

    int na = read_pages(snap_a, MAX_ENTRIES);

    /* find max timestamp in phase A */
    unsigned long max_ts_a = 0;
    int a_present = 0;
    for (int i = 0; i < na; i++) {
        if (snap_a[i].addr < va_a || snap_a[i].addr >= va_b) continue;
        a_present++;
        if (snap_a[i].ts > max_ts_a) max_ts_a = snap_a[i].ts;
    }
    printf("[tc08] phase-A: %d entries, max_ts=%lu\n", a_present, max_ts_a);

    /* phase B: massive fault flood */
    phase_write<<<HALF_PAGES, PHASE_B_THREADS>>>(phase_b_base, HALF_PAGES, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc08] phase B complete (%d threads)\n", HALF_PAGES * PHASE_B_THREADS);

    int nb = read_pages(snap_b, MAX_ENTRIES);

    /* ordering check: all phase-B entries must have ts >= max_ts_a */
    int b_missing   = 0;
    int b_early     = 0;
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
           nb - a_present, min_ts_b == (unsigned long)-1 ? 0 : min_ts_b, max_ts_a);
    printf("[tc08] b_missing=%d  ordering_violations=%d\n", b_missing, b_early);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(snap_a);
    free(snap_b);

    int failed = (na < 0 || nb < 0 || a_present < HALF_PAGES
                  || b_missing > 0 || b_early > 0);
    printf("[tc08] %s\n", failed ? "FAIL" : "PASS");
    if (na < 0)             printf("[tc08]   phase-A table not active\n");
    if (nb < 0)             printf("[tc08]   phase-B table not active\n");
    if (a_present < HALF_PAGES) printf("[tc08]   phase-A: only %d/%d pages tracked\n", a_present, HALF_PAGES);
    if (b_missing > 0)      printf("[tc08]   %d phase-B pages not recorded\n", b_missing);
    if (b_early > 0)        printf("[tc08]   %d phase-B entries recorded with ts < phase-A max\n", b_early);
    return failed;
}
