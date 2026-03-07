/*
 * tc04_inverted_and_zero_range.cu — address_range_filtering tests
 *
 * The kernel reads the dirty page table via:
 *
 *   start_index = dirty_query_start >> PAGE_SHIFT
 *   end_index   = (dirty_query_end  - 1) >> PAGE_SHIFT
 *   xa_find(&table->pages, &index, end_index, XA_PRESENT)
 *
 * xa_find() returns NULL immediately if the starting index exceeds the max.
 * This opens three degenerate range cases that are never guarded by the
 * kernel code:
 *
 *   Case A — zero-width (start == end):
 *     end_index = (start - 1) >> PAGE_SHIFT < start_index
 *     xa_find returns nothing → 0 entries expected.
 *
 *   Case B — inverted (start > end, both non-zero):
 *     end_index < start_index in page space → 0 entries expected.
 *
 *   Case C — end underflow (end == 0):
 *     end_index = (0 - 1) >> PAGE_SHIFT = ULONG_MAX >> PAGE_SHIFT (~huge)
 *     xa_find walks from start_index to a near-infinite max.
 *     Expected behaviour: all pages at or above start_index appear.
 *     (This is an "end wraps to all-pages" side-effect, not a defined API.)
 *
 * Four CPU threads write 4 pages each concurrently before any filter is set,
 * so all 16 pages are in the xarray.  Each case is then applied as a read-
 * only filter probe without resetting the table between probes.
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

#define NUM_THREADS     4
#define PAGES_PER_THREAD 4
#define NUM_PAGES       (NUM_THREADS * PAGES_PER_THREAD)
#define PAGE_SIZE       4096
#define INTS_PER_PAGE   (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES     4096

#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void write_range(int *base, 
    int page_start, 
    int page_end) {
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

static void procfs_write(const char *path, 
    const char *val) {
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

/*
 * Write the range directly as hex pairs.  The kernel sscanf uses %lx which
 * accepts the 0x prefix.
 */
static void set_range_raw(unsigned long s, 
    unsigned long e) {
    char b[64];
    snprintf(b, sizeof(b), "0x%lx 0x%lx\n", s, e);
    procfs_write(PROCFS_RANGE, b);
}

static void reset_range(void) {
    procfs_write(PROCFS_RANGE, "0x0 0xffffffffffffffff\n");
}

static int read_dirty(entry_t *out, 
    int max) {
    FILE *f = fopen(PROCFS_PAGES, "r");
    if (!f) return -1;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) { 
                fclose(f); 
                return -2; 
            }
            continue;
        }
        if (n < max &&
            sscanf(line, "0x%lx %lu %d", &out[n].addr, &out[n].ts, &out[n].pid) == 3)
            n++;
    }
    fclose(f);
    return n;
}

static int count_in_alloc(entry_t *e, 
    int n, 
    unsigned long base) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (e[i].addr >= base && e[i].addr < base + (unsigned long)NUM_PAGES * PAGE_SIZE)
            c++;
    return c;
}

int main(void) {
    printf("[tc04] inverted_and_zero_range — %d pages, %d threads\n", NUM_PAGES, NUM_THREADS);

    if (geteuid() != 0) { 
        fprintf(stderr, "ERROR: must run as root\n"); 
        return 1; 
    }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    pid_t pid = getpid();
    printf("[tc04] pid=%d  base=0x%lx  pages=%d\n", pid, base, NUM_PAGES);

    set_query_pid(pid);
    reset_range();
    procfs_write(PROCFS_START, "1\n");

    /* All four threads write concurrently; all 16 pages land in the xarray. */
    thread_arg_t args[NUM_THREADS];
    pthread_t threads[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].managed_base = managed;
        args[t].page_start = t * PAGES_PER_THREAD;
        args[t].page_end = args[t].page_start + PAGES_PER_THREAD;
        args[t].device = dev;
        pthread_create(&threads[t], NULL, write_thread, &args[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++) pthread_join(threads[t], NULL);
    printf("[tc04] all %d write threads joined\n", NUM_THREADS);

    /* Sanity: verify all pages are recorded with the full range. */
    reset_range();
    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n_sanity = read_dirty(e, MAX_ENTRIES);
    int recorded = (n_sanity >= 0) ? count_in_alloc(e, n_sanity, base) : -1;
    printf("[tc04] sanity (range=all): %d/%d pages recorded\n", recorded, NUM_PAGES);
    if (recorded != NUM_PAGES) {
        printf("[tc04] FAIL: not all pages recorded before filter tests (got %d)\n", recorded);
        procfs_write(PROCFS_STOP, "1\n");
        CUDA_CHECK(cudaFree(managed));
        free(e);
        return 1;
    }

    int failed = 0;

    /* ---- Case A: zero-width range (start == end == mid-allocation) ------- */
    unsigned long mid = base + (unsigned long)(NUM_PAGES / 2) * PAGE_SIZE;
    set_range_raw(mid, mid); 
    int nA = read_dirty(e, MAX_ENTRIES);
    int cA = (nA >= 0) ? count_in_alloc(e, nA, base) : -1;
    printf("[tc04] caseA (start==end==mid): total=%d  in_alloc=%d  (want 0)\n", nA, cA);
    if (nA < 0) {
        printf("[tc04] FAIL caseA: table not active\n"); failed = 1;
    } else if (cA != 0) {
        printf("[tc04] FAIL caseA: expected 0 pages, got %d\n", cA); failed = 1;
    }

    /* ---- Case B: inverted range (start > end, both non-zero) ------------- */
    unsigned long inv_start = base + (unsigned long)(NUM_PAGES - 2) * PAGE_SIZE;
    unsigned long inv_end   = base + 2UL * PAGE_SIZE; 
    set_range_raw(inv_start, inv_end);
    int nB = read_dirty(e, MAX_ENTRIES);
    int cB = (nB >= 0) ? count_in_alloc(e, nB, base) : -1;
    printf("[tc04] caseB (start=page%d, end=page2, inverted): total=%d  in_alloc=%d  (want 0)\n", NUM_PAGES - 2, nB, cB);
    if (nB < 0) {
        printf("[tc04] FAIL caseB: table not active\n"); 
        failed = 1;
    } else if (cB != 0) {
        printf("[tc04] FAIL caseB: expected 0 pages, got %d\n", cB); 
        failed = 1;
    }

    /* ---- Case C: end = 0 (start = base > 0) ------------------------------ */
    /*
     * dirty_query_end = 0, dirty_query_start = base > 0.
     * The kernel guard (dirty_query_end <= dirty_query_start) fires because
     * 0 <= base, so the invalid-range branch is taken and 0 entries are returned.
     * Expected: 0 pages in allocation (same as Cases A and B).
     */
    set_range_raw(base, 0UL);
    int nC = read_dirty(e, MAX_ENTRIES);
    int cC = (nC >= 0) ? count_in_alloc(e, nC, base) : -1;
    printf("[tc04] caseC (start=base, end=0x0): total=%d  in_alloc=%d  (want 0)\n", nC, cC);
    if (nC < 0) {
        printf("[tc04] FAIL caseC: table not active\n"); failed = 1;
    } else if (cC != 0) {
        printf("[tc04] FAIL caseC: expected 0 pages (kernel should reject end=0), got %d\n", cC); failed = 1;
    }

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(e);

    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
