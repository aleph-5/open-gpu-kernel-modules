/*
 * tc02_range_sequential_reads_threaded.cu — address_range_filtering tests
 *
 * A single managed allocation is divided into 4 equal quarters (Q0..Q3).
 * Four CPU threads each launch a GPU write kernel to their own quarter via
 * separate CUDA streams, all dispatched concurrently.
 *
 * After all writes complete, the dirty_range filter is applied in two
 * successive reads WITHOUT resetting the tracking table between them:
 *
 *   Read 1 — range = [Q0_base, Q2_base)  (quarters 0 and 1 only):
 *     Expected: Q0 and Q1 pages present, Q2 and Q3 pages absent.
 *
 *   Read 2 — range = [Q2_base, Q4_base)  (quarters 2 and 3 only):
 *     Expected: Q0 and Q1 pages absent, Q2 and Q3 pages present.
 *
 * This probes two things simultaneously:
 *   a) Entries written by concurrent faults are all recorded and persist
 *      across reads (non-destructive, entries survive in the xarray).
 *   b) Changing the global dirty_range filter between reads produces the
 *      correct independent view of the same snapshot each time.
 *
 * The concurrent write phase stresses the fault handler's xarray insertion
 * path before the range filter is exercised.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <cuda_runtime.h>

#define PROCFS_START  "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP   "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY  "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_PAGES  "/proc/driver/nvidia-uvm/dirty_pages"
#define PROCFS_RANGE  "/proc/driver/nvidia-uvm/dirty_range"

#define NUM_QUARTERS        4
#define PAGES_PER_QUARTER   16
#define TOTAL_PAGES         (NUM_QUARTERS * PAGES_PER_QUARTER)
#define PAGE_SIZE           4096
#define INTS_PER_PAGE       (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES         8192

#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

/*
 * Write all ints in [page_start, page_end) pages of data.
 * One block per quarter; threads within the block cover the ints stride.
 */
__global__ void write_quarter(int *base, int page_start, int page_end) {
    int pg = blockIdx.x + page_start;
    if (pg >= page_end) return;
    int *p = base + pg * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = pg * 1000 + i + 1;
}

typedef struct {
    int *managed_base;
    int  quarter_idx;
    int  device;
} thread_arg_t;

static void *write_thread(void *arg) {
    thread_arg_t *a = (thread_arg_t *)arg;
    int ps = a->quarter_idx * PAGES_PER_QUARTER;
    int pe = ps + PAGES_PER_QUARTER;
    CUDA_CHECK(cudaSetDevice(a->device));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    write_quarter<<<PAGES_PER_QUARTER, 256, 0, s>>>(a->managed_base, ps, pe);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
    return NULL;
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

static void set_range(unsigned long s, unsigned long e) {
    char b[64];
    snprintf(b, sizeof(b), "0x%lx 0x%lx\n", s, e);
    procfs_write(PROCFS_RANGE, b);
}

static void reset_range(void) {
    procfs_write(PROCFS_RANGE, "0x0 0xffffffffffffffff\n");
}

static int read_dirty(entry_t *out, int max) {
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

/*
 * Returns the number of tracked pages that fall inside the given quarter.
 * Also separately counts pages that should NOT be in the quarter (out_count).
 */
static void tally_quarter(entry_t *e, int n, unsigned long base, int q,
                           int *present, int *absent) {
    unsigned long q_base = base + (unsigned long)q * PAGES_PER_QUARTER * PAGE_SIZE;
    unsigned long q_end  = q_base + (unsigned long)PAGES_PER_QUARTER * PAGE_SIZE;
    *present = 0;
    /* count distinct pages in quarter that appear in e[] */
    for (int p = 0; p < PAGES_PER_QUARTER; p++) {
        unsigned long paddr = q_base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++) {
            if (e[i].addr == paddr) { (*present)++; break; }
        }
    }
    *absent = PAGES_PER_QUARTER - *present;
}

int main(void) {
    printf("[tc02] range_sequential_reads_threaded — %d quarters x %d pages, %d threads\n",
           NUM_QUARTERS, PAGES_PER_QUARTER, NUM_QUARTERS);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    pid_t pid = getpid();
    printf("[tc02] pid=%d  base=0x%lx  total_pages=%d\n", pid, base, TOTAL_PAGES);
    for (int q = 0; q < NUM_QUARTERS; q++)
        printf("[tc02]   Q%d: 0x%lx - 0x%lx\n",
               q,
               base + (unsigned long)q * PAGES_PER_QUARTER * PAGE_SIZE,
               base + (unsigned long)(q + 1) * PAGES_PER_QUARTER * PAGE_SIZE);

    set_query_pid(pid);
    reset_range();
    procfs_write(PROCFS_START, "1\n");

    /* Four CPU threads launch concurrent GPU writes to their respective quarters. */
    thread_arg_t args[NUM_QUARTERS];
    pthread_t threads[NUM_QUARTERS];
    for (int q = 0; q < NUM_QUARTERS; q++) {
        args[q].managed_base = managed;
        args[q].quarter_idx  = q;
        args[q].device       = dev;
        pthread_create(&threads[q], NULL, write_thread, &args[q]);
    }
    for (int q = 0; q < NUM_QUARTERS; q++)
        pthread_join(threads[q], NULL);
    printf("[tc02] all %d write threads joined\n", NUM_QUARTERS);

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* ---- Read 1: range = Q0 + Q1 --------------------------------------- */
    unsigned long q0_base = base;
    unsigned long q2_base = base + 2UL * PAGES_PER_QUARTER * PAGE_SIZE;
    unsigned long q4_base = base + 4UL * PAGES_PER_QUARTER * PAGE_SIZE; /* past end */

    set_range(q0_base, q2_base);
    int n1 = read_dirty(e, MAX_ENTRIES);
    int present1[NUM_QUARTERS], absent1[NUM_QUARTERS];
    for (int q = 0; q < NUM_QUARTERS; q++)
        tally_quarter(e, n1 >= 0 ? n1 : 0, base, q, &present1[q], &absent1[q]);

    printf("[tc02] read1 (range=Q0+Q1): total=%d\n", n1);
    for (int q = 0; q < NUM_QUARTERS; q++)
        printf("[tc02]   Q%d present=%d absent=%d (want present=%d absent=%d)\n",
               q, present1[q], absent1[q],
               q < 2 ? PAGES_PER_QUARTER : 0,
               q < 2 ? 0 : PAGES_PER_QUARTER);

    /* ---- Read 2: range = Q2 + Q3 (no table reset) ---------------------- */
    set_range(q2_base, q4_base);
    int n2 = read_dirty(e, MAX_ENTRIES);
    int present2[NUM_QUARTERS], absent2[NUM_QUARTERS];
    for (int q = 0; q < NUM_QUARTERS; q++)
        tally_quarter(e, n2 >= 0 ? n2 : 0, base, q, &present2[q], &absent2[q]);

    printf("[tc02] read2 (range=Q2+Q3): total=%d\n", n2);
    for (int q = 0; q < NUM_QUARTERS; q++)
        printf("[tc02]   Q%d present=%d absent=%d (want present=%d absent=%d)\n",
               q, present2[q], absent2[q],
               q >= 2 ? PAGES_PER_QUARTER : 0,
               q >= 2 ? 0 : PAGES_PER_QUARTER);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(e);

    int failed = 0;

    if (n1 < 0) {
        printf("[tc02] FAIL: read1 table not active (n=%d)\n", n1);
        failed = 1;
    } else {
        /* Q0 and Q1 must be fully present */
        for (int q = 0; q < 2; q++)
            if (present1[q] != PAGES_PER_QUARTER) {
                printf("[tc02] FAIL: read1 Q%d — %d/%d pages missing\n",
                       q, PAGES_PER_QUARTER - present1[q], PAGES_PER_QUARTER);
                failed = 1;
            }
        /* Q2 and Q3 must be fully absent */
        for (int q = 2; q < NUM_QUARTERS; q++)
            if (present1[q] != 0) {
                printf("[tc02] FAIL: read1 Q%d leaked through filter (%d pages)\n",
                       q, present1[q]);
                failed = 1;
            }
    }

    if (n2 < 0) {
        printf("[tc02] FAIL: read2 table not active (n=%d)\n", n2);
        failed = 1;
    } else {
        /* Q0 and Q1 must be fully absent */
        for (int q = 0; q < 2; q++)
            if (present2[q] != 0) {
                printf("[tc02] FAIL: read2 Q%d leaked through filter (%d pages)\n",
                       q, present2[q]);
                failed = 1;
            }
        /* Q2 and Q3 must be fully present (entries survived read1) */
        for (int q = 2; q < NUM_QUARTERS; q++)
            if (present2[q] != PAGES_PER_QUARTER) {
                printf("[tc02] FAIL: read2 Q%d — %d/%d pages missing (consumed by read1?)\n",
                       q, PAGES_PER_QUARTER - present2[q], PAGES_PER_QUARTER);
                failed = 1;
            }
    }

    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
