/*
 * tc01_two_alloc_range_isolation.cu — address_range_filtering tests
 *
 * Two CPU threads each launch a GPU write kernel against their own managed
 * allocation (alloc_a, alloc_b) via separate CUDA streams, all within the
 * same process.  Both allocations are tracked under the same pid because
 * start_track uses current->tgid.
 *
 * After both writes complete, the dirty_range filter is probed in two phases:
 *
 *   Phase 1 — range set to [alloc_a_base, alloc_a_base + alloc_size):
 *     Expected: alloc_a pages present, alloc_b pages absent.
 *     This checks that the filter correctly excludes pages outside the range
 *     even when two allocations are stored in the same xarray.
 *
 *   Phase 2 — range reset to full address space (no table reset):
 *     Expected: pages from BOTH allocations present.
 *     This confirms entries are not consumed by the Phase 1 read, i.e.
 *     the procfs read is non-destructive.
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

#define NUM_PAGES     16
#define PAGE_SIZE     4096
#define INTS_PER_PAGE (PAGE_SIZE / sizeof(int))
#define NUM_INTS      (NUM_PAGES * INTS_PER_PAGE)
#define MAX_ENTRIES   4096

#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

/* One block, 256 threads — each thread covers a strided range of ints. */
__global__ void write_pages(int *data, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        data[i] = i + 1;
}

typedef struct {
    int *managed;
    int  device;
} thread_arg_t;

static void *write_thread(void *arg) {
    thread_arg_t *a = (thread_arg_t *)arg;
    /* Attach this CPU thread to the primary CUDA context. */
    CUDA_CHECK(cudaSetDevice(a->device));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    write_pages<<<1, 256, 0, s>>>(a->managed, NUM_INTS);
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

/*
 * dirty_range_write uses sscanf(kbuf, "%lx %lx", ...) — %lx handles the 0x
 * prefix, and dirty_query_end is exclusive (kernel does (end-1)>>PAGE_SHIFT).
 */
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

static int count_in_alloc(entry_t *e, int n, unsigned long base) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (e[i].addr >= base && e[i].addr < base + NUM_PAGES * PAGE_SIZE)
            c++;
    return c;
}

int main(void) {
    printf("[tc01] two_alloc_range_isolation — %d pages per alloc, 2 threads\n", NUM_PAGES);

    if (geteuid() != 0) { 
        fprintf(stderr, "ERROR: must run as root\n"); 
        return 1; 
    }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, NUM_PAGES * PAGE_SIZE));
    memset(alloc_a, 0, NUM_PAGES * PAGE_SIZE);
    memset(alloc_b, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base_a = (unsigned long)alloc_a;
    unsigned long base_b = (unsigned long)alloc_b;

    pid_t pid = getpid();
    printf("[tc01] pid=%d  alloc_a=0x%lx  alloc_b=0x%lx  pages=%d each\n", pid, base_a, base_b, NUM_PAGES);

    set_query_pid(pid);
    reset_range();
    procfs_write(PROCFS_START, "1\n");

    /* Two CPU threads launch concurrent GPU writes to separate allocations. */
    thread_arg_t args[2] = { { alloc_a, dev }, { alloc_b, dev } };
    pthread_t threads[2];
    for (int t = 0; t < 2; t++)
        pthread_create(&threads[t], NULL, write_thread, &args[t]);
    for (int t = 0; t < 2; t++)
        pthread_join(threads[t], NULL);
    printf("[tc01] both write threads joined\n");

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* ---- Phase 1: range = alloc_a only ---------------------------------- */
    set_range(base_a, base_a + (unsigned long)NUM_PAGES * PAGE_SIZE);
    int n1   = read_dirty(e, MAX_ENTRIES);
    int a_p1 = (n1 >= 0) ? count_in_alloc(e, n1, base_a) : -1;
    int b_p1 = (n1 >= 0) ? count_in_alloc(e, n1, base_b) : -1;
    printf("[tc01] phase1 (range=alloc_a): total=%d  in_a=%d (want %d)  in_b=%d (want 0)\n",
           n1, a_p1, NUM_PAGES, b_p1);

    /* ---- Phase 2: reset range — both allocs should be visible ----------- */
    reset_range();
    int n2   = read_dirty(e, MAX_ENTRIES);
    int a_p2 = (n2 >= 0) ? count_in_alloc(e, n2, base_a) : -1;
    int b_p2 = (n2 >= 0) ? count_in_alloc(e, n2, base_b) : -1;
    printf("[tc01] phase2 (range=all):    total=%d  in_a=%d (want %d)  in_b=%d (want %d)\n",
           n2, a_p2, NUM_PAGES, b_p2, NUM_PAGES);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(alloc_a));
    CUDA_CHECK(cudaFree(alloc_b));
    free(e);

    int failed = 0;

    if (n1 < 0) {
        printf("[tc01] FAIL: phase1 table not active (n=%d)\n", n1);
        failed = 1;
    } else {
        if (a_p1 != NUM_PAGES) {
            printf("[tc01] FAIL: phase1 alloc_a — %d/%d pages missing\n",
                   NUM_PAGES - a_p1, NUM_PAGES);
            failed = 1;
        }
        if (b_p1 != 0) {
            printf("[tc01] FAIL: phase1 alloc_b leaked through filter (%d pages)\n", b_p1);
            failed = 1;
        }
    }

    if (n2 < 0) {
        printf("[tc01] FAIL: phase2 table not active (n=%d)\n", n2);
        failed = 1;
    } else {
        if (a_p2 != NUM_PAGES) {
            printf("[tc01] FAIL: phase2 alloc_a — %d/%d pages missing (non-destructive read?)\n",
                   NUM_PAGES - a_p2, NUM_PAGES);
            failed = 1;
        }
        if (b_p2 != NUM_PAGES) {
            printf("[tc01] FAIL: phase2 alloc_b — %d/%d pages missing\n",
                   NUM_PAGES - b_p2, NUM_PAGES);
            failed = 1;
        }
    }

    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
