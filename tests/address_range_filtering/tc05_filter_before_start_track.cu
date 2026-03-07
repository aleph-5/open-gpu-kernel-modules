/*
 * tc05_filter_before_start_track.cu — address_range_filtering tests
 *
 * The dirty_range filter is a pair of global variables
 * (dirty_query_start / dirty_query_end) that live independently of the
 * per-pid xarray table.  The recording path (uvm_dirty_page_table_record,
 * called from the GPU fault handler) has no knowledge of dirty_range —
 * it always inserts unconditionally if the page is not already present.
 * The filter is applied only at read time inside nv_procfs_read_dirty_pages.
 *
 * This test verifies that conclusion:
 *
 *   Step 1 — set dirty_range to the FIRST HALF of the allocation before
 *             start_track is called.
 *   Step 2 — start_track (table is initialised, all GPU PTEs invalidated).
 *   Step 3 — four CPU threads write ALL pages concurrently (both halves).
 *   Step 4 — Read A: dirty_range still covers first half only.
 *             Expected: first-half pages present, second-half pages absent.
 *             If the filter were applied at insert time, only first-half pages
 *             would ever enter the xarray; if it is read-time, both halves are
 *             in the xarray but only the first half is returned.
 *   Step 5 — Read B: reset dirty_range to full address space (no table reset).
 *             Expected: BOTH halves present.
 *             This is the decisive probe: if second-half pages appear in Read B,
 *             they were in the xarray all along — the filter is read-time only.
 *             If second-half pages are absent in Read B, they were never
 *             recorded — the filter somehow acts at insert time (unexpected).
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

#define NUM_THREADS      4
#define PAGES_PER_THREAD 8
#define NUM_PAGES        (NUM_THREADS * PAGES_PER_THREAD)  /* 32 total */
#define HALF_PAGES       (NUM_PAGES / 2)
#define PAGE_SIZE        4096
#define INTS_PER_PAGE    (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES      4096

#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void write_range(int *base, int page_start, int page_end) {
    int pg = blockIdx.x + page_start;
    if (pg >= page_end) return;
    int *p = base + pg * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = pg * 1000 + i + 1;
}

typedef struct {
    int *managed_base;
    int  page_start;
    int  page_end;
    int  device;
} thread_arg_t;

static void *write_thread(void *arg) {
    thread_arg_t *a = (thread_arg_t *)arg;
    CUDA_CHECK(cudaSetDevice(a->device));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    write_range<<<PAGES_PER_THREAD, 256, 0, s>>>(a->managed_base, a->page_start, a->page_end);
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

static int count_in_half(entry_t *e, int n, unsigned long base, int second_half) {
    unsigned long lo = base + (second_half ? (unsigned long)HALF_PAGES * PAGE_SIZE : 0);
    unsigned long hi = lo + (unsigned long)HALF_PAGES * PAGE_SIZE;
    int c = 0;
    for (int i = 0; i < n; i++)
        if (e[i].addr >= lo && e[i].addr < hi) c++;
    return c;
}

int main(void) {
    printf("[tc05] filter_before_start_track — %d pages (%d per thread), %d threads\n",
           NUM_PAGES, PAGES_PER_THREAD, NUM_THREADS);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    pid_t pid = getpid();
    printf("[tc05] pid=%d  base=0x%lx\n", pid, base);
    printf("[tc05] first_half=[0x%lx, 0x%lx)  second_half=[0x%lx, 0x%lx)\n",
           base,
           base + (unsigned long)HALF_PAGES * PAGE_SIZE,
           base + (unsigned long)HALF_PAGES * PAGE_SIZE,
           base + (unsigned long)NUM_PAGES  * PAGE_SIZE);

    set_query_pid(pid);

    /* Step 1: set filter to FIRST HALF before start_track. */
    set_range(base, base + (unsigned long)HALF_PAGES * PAGE_SIZE);
    printf("[tc05] dirty_range set to first half BEFORE start_track\n");

    /* Step 2: start tracking (table init + GPU PTE invalidation). */
    procfs_write(PROCFS_START, "1\n");
    printf("[tc05] start_track issued\n");

    /* Step 3: write ALL pages concurrently — both halves. */
    thread_arg_t args[NUM_THREADS];
    pthread_t threads[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].managed_base = managed;
        args[t].page_start   = t * PAGES_PER_THREAD;
        args[t].page_end     = args[t].page_start + PAGES_PER_THREAD;
        args[t].device       = dev;
        pthread_create(&threads[t], NULL, write_thread, &args[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);
    printf("[tc05] all %d write threads joined (both halves written)\n", NUM_THREADS);

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Step 4: Read A — range still set to first half. */
    int nA    = read_dirty(e, MAX_ENTRIES);
    int fh_A  = (nA >= 0) ? count_in_half(e, nA, base, 0) : -1;
    int sh_A  = (nA >= 0) ? count_in_half(e, nA, base, 1) : -1;
    printf("[tc05] readA (range=first_half): total=%d  first_half=%d (want %d)  second_half=%d (want 0)\n",
           nA, fh_A, HALF_PAGES, sh_A);

    /* Step 5: Read B — reset range, decisive probe for xarray contents. */
    reset_range();
    int nB    = read_dirty(e, MAX_ENTRIES);
    int fh_B  = (nB >= 0) ? count_in_half(e, nB, base, 0) : -1;
    int sh_B  = (nB >= 0) ? count_in_half(e, nB, base, 1) : -1;
    printf("[tc05] readB (range=all):        total=%d  first_half=%d (want %d)  second_half=%d\n",
           nB, fh_B, HALF_PAGES, sh_B);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(e);

    int failed = 0;

    if (nA < 0) {
        printf("[tc05] FAIL readA: table not active (n=%d)\n", nA); failed = 1;
    } else {
        if (fh_A != HALF_PAGES) {
            printf("[tc05] FAIL readA: first half — %d/%d pages missing\n",
                   HALF_PAGES - fh_A, HALF_PAGES);
            failed = 1;
        }
        if (sh_A != 0) {
            printf("[tc05] FAIL readA: second half leaked through pre-set filter (%d pages)\n", sh_A);
            failed = 1;
        }
    }

    if (nB < 0) {
        printf("[tc05] FAIL readB: table not active (n=%d)\n", nB); failed = 1;
    } else {
        if (fh_B != HALF_PAGES) {
            printf("[tc05] FAIL readB: first half — %d/%d pages missing\n",
                   HALF_PAGES - fh_B, HALF_PAGES);
            failed = 1;
        }
        /*
         * Decisive assertion: second-half pages MUST appear after range reset.
         * If they are absent here, the filter acted at insert time (unexpected).
         */
        if (sh_B != HALF_PAGES) {
            printf("[tc05] FAIL readB: second half — %d/%d pages missing after range reset\n",
                   HALF_PAGES - sh_B, HALF_PAGES);
            printf("[tc05]   => second-half pages were NOT recorded: filter acts at INSERT time\n");
            failed = 1;
        } else {
            printf("[tc05]   => second-half pages present after reset: filter is READ-TIME only\n");
        }
    }

    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
