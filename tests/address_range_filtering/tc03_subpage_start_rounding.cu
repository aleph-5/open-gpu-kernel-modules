/*
 * tc03_subpage_start_rounding.cu — address_range_filtering tests
 *
 * The kernel converts dirty_query_start to a page index by right-shifting:
 *
 *   start_index = dirty_query_start >> PAGE_SHIFT
 *
 * This truncates toward zero, so a start address that is not page-aligned
 * rounds DOWN to the page that contains it.  Concretely: if the allocation
 * base is `B` and we write dirty_range start as `B + PAGE_SIZE/2`, then
 * start_index == B >> PAGE_SHIFT, which is the index of page 0.  Page 0
 * therefore falls inside the filter window and should appear in the output.
 *
 * Similarly, dirty_query_end is exclusive via:
 *   end_index = (dirty_query_end - 1) >> PAGE_SHIFT
 * so a sub-page end address rounds DOWN too — a page is included only if
 * its index <= end_index.
 *
 * Test layout (8 pages, 4 CPU threads):
 *   Thread 0 → pages 0-1    Thread 1 → pages 2-3
 *   Thread 2 → pages 4-5    Thread 3 → pages 6-7
 *
 * Two filter probes after all writes:
 *
 *   Probe A — start = base + PAGE_SIZE/2, end = base + 4*PAGE_SIZE
 *     start_index == page 0 (rounded down)
 *     end_index   == page 3 (rounded down from base+4*PAGE_SIZE-1)
 *     Expected: pages 0-3 present, pages 4-7 absent.
 *     Specifically page 0 presence confirms the "rounds down" behaviour.
 *
 *   Probe B — start = base + PAGE_SIZE/2, end = base + PAGE_SIZE
 *     start_index == page 0, end_index == page 0
 *     A one-page window opened by a sub-page start: page 0 only.
 *     Expected: exactly 1 page (page 0) present, pages 1-7 absent.
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

#define NUM_PAGES       8
#define PAGES_PER_THREAD 2
#define NUM_THREADS     (NUM_PAGES / PAGES_PER_THREAD)
#define PAGE_SIZE       4096
#define INTS_PER_PAGE   (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES     1024

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
    if (fd < 0) { 
        perror(path); 
        exit(1); 
    }
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
            if (strstr(line, "not active")) { 
                fclose(f); 
                return -2; 
            }
            continue;
        }
        if (n < max && sscanf(line, "0x%lx %lu %d", &out[n].addr, &out[n].ts, &out[n].pid) == 3)
            n++;
    }
    fclose(f);
    return n;
}

static int page_present(entry_t *e, int n, unsigned long base, int pg) {
    unsigned long addr = base + (unsigned long)pg * PAGE_SIZE;
    for (int i = 0; i < n; i++)
        if (e[i].addr == addr) 
            return 1;
    return 0;
}

int main(void) {
    printf("[tc03] subpage_start_rounding — %d pages, %d threads\n", NUM_PAGES, NUM_THREADS);

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
    printf("[tc03] pid=%d  base=0x%lx\n", pid, base);
    printf("[tc03] PAGE_SIZE=%d  PAGE_SHIFT=12\n", PAGE_SIZE);
    printf("[tc03] subpage offset used: PAGE_SIZE/2 = %d bytes\n", PAGE_SIZE / 2);

    set_query_pid(pid);
    reset_range();
    procfs_write(PROCFS_START, "1\n");

    thread_arg_t args[NUM_THREADS];
    pthread_t threads[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].managed_base = managed;
        args[t].page_start = t * PAGES_PER_THREAD;
        args[t].page_end = args[t].page_start + PAGES_PER_THREAD;
        args[t].device = dev;
        pthread_create(&threads[t], NULL, write_thread, &args[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);
    printf("[tc03] all %d write threads joined\n", NUM_THREADS);

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int failed = 0;

    /* ---- Probe A: start = base + PAGE_SIZE/2, end = base + 4*PAGE_SIZE -- */
    unsigned long start_A = base + PAGE_SIZE / 2;
    unsigned long end_A   = base + 4UL * PAGE_SIZE;
    /*
     * Kernel: start_index = start_A >> 12 = (base + PAGE_SIZE/2) >> 12 = base>>12
     *         end_index   = (end_A - 1) >> 12 = (base + 4*PAGE_SIZE - 1) >> 12 = base>>12 + 3
     * => pages 0..3 in range, pages 4..7 out.
     */
    set_range(start_A, end_A);
    int nA = read_dirty(e, MAX_ENTRIES);
    printf("[tc03] probeA (start=base+PAGE_SIZE/2, end=base+4*PAGE_SIZE): total=%d\n", nA);

    if (nA < 0) {
        printf("[tc03] FAIL probeA: table not active (n=%d)\n", nA);
        failed = 1;
    } else {
        for (int p = 0; p < NUM_PAGES; p++) {
            int present = page_present(e, nA, base, p);
            int want    = (p < 4) ? 1 : 0;
            printf("[tc03]   page %d: %s (want %s)%s\n",
                   p, present ? "present" : "absent",
                   want ? "present" : "absent",
                   (present == want) ? "" : "  *** MISMATCH");
            if (present != want) failed = 1;
        }
        /* Specific call-out for the rounded-down page */
        printf("[tc03]   page 0 (the rounded-down boundary page): %s\n",
               page_present(e, nA, base, 0) ? "PRESENT (start rounded down)" : "ABSENT");
    }

    /* ---- Probe B: start = base + PAGE_SIZE/2, end = base + PAGE_SIZE ----- */
    unsigned long start_B = base + PAGE_SIZE / 2;
    unsigned long end_B   = base + PAGE_SIZE;
    /*
     * Kernel: start_index = base>>12 + 0 (page 0)
     *         end_index   = (base + PAGE_SIZE - 1) >> 12 = base>>12 + 0 (page 0)
     * => single-page window: page 0 only.
     */
    set_range(start_B, end_B);
    int nB = read_dirty(e, MAX_ENTRIES);
    printf("[tc03] probeB (start=base+PAGE_SIZE/2, end=base+PAGE_SIZE): total=%d\n", nB);

    if (nB < 0) {
        printf("[tc03] FAIL probeB: table not active (n=%d)\n", nB);
        failed = 1;
    } else {
        for (int p = 0; p < NUM_PAGES; p++) {
            int present = page_present(e, nB, base, p);
            int want    = (p == 0) ? 1 : 0;
            printf("[tc03]   page %d: %s (want %s)%s\n",
                   p, present ? "present" : "absent",
                   want ? "present" : "absent",
                   (present == want) ? "" : "  *** MISMATCH");
            if (present != want) failed = 1;
        }
    }

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(e);

    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
