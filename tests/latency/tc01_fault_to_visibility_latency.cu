/*
 * tc01_fault_to_visibility_latency.cu — latency tests
 *
 * After cudaDeviceSynchronize() returns, every UVM page fault that occurred
 * during the kernel has already been processed by the fault handler — the GPU
 * cannot make forward progress past the sync point without the host having
 * seen all faults.  Therefore the dirty_pages xarray must be fully populated
 * at the instant sync returns.
 *
 * This test measures the wall-clock gap between sync completion and the
 * moment all NUM_PAGES entries first appear in the procfs interface.  A
 * spin-poll (no sleep between iterations) is used so that scheduling jitter
 * does not inflate the measurement.
 *
 * Pass criteria:
 *   1. All NUM_PAGES entries become visible within DEADLINE_US microseconds
 *      of cudaDeviceSynchronize() returning.
 *   2. Entries appear at all (no timeout after TIMEOUT_S seconds).
 *   3. The final entry count equals exactly NUM_PAGES (no extras, no missing).
 *
 * Why this is hard to pass:
 *   If the driver batches fault recording asynchronously — even by a single
 *   scheduler quantum — the latency will exceed the 5 ms deadline on any
 *   system with normal kernel preemption latency.  An implementation that
 *   stamps entries in a background kthread rather than inline in the fault
 *   handler will also fail.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <cuda_runtime.h>

#define PROCFS_START  "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP   "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY  "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_PAGES  "/proc/driver/nvidia-uvm/dirty_pages"

#define NUM_PAGES      64
#define PAGE_SIZE      4096
#define INTS_PER_PAGE  (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES    4096

#define DEADLINE_US    5000UL   /* 5 ms — tight visibility deadline after sync */
#define TIMEOUT_S      5        /* give up after 5 s to avoid hanging forever  */

#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

/* One block, 256 threads — strided writes touch every int on every page. */
__global__ void write_pages(int *data, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        data[i] = i + 1;
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

static long ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int main(void) {
    printf("[tc01] fault_to_visibility_latency — %d pages, deadline=%luus\n",
           NUM_PAGES, DEADLINE_US);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc01] pid=%d  alloc=0x%lx\n", pid, (unsigned long)managed);
    set_query_pid(pid);
    procfs_write(PROCFS_START, "1\n");

    write_pages<<<1, 256>>>(managed, NUM_PAGES * (int)INTS_PER_PAGE);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* t_sync: the earliest moment at which ALL faults are guaranteed committed */
    long t_sync    = ns_now();
    long t_timeout = t_sync + (long)TIMEOUT_S * 1000000000L;

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int polls = 0, n = 0;
    long t_visible = 0;
    int timed_out = 0;

    /*
     * Spin-poll without sleeping.  Each iteration opens, reads, and closes
     * the procfs file — the overhead is intentionally included in the
     * latency measurement because that is what a real consumer would pay.
     */
    while (1) {
        n = read_dirty(e, MAX_ENTRIES);
        polls++;
        long now = ns_now();
        if (n == NUM_PAGES) {
            t_visible = now;
            break;
        }
        if (now >= t_timeout) {
            t_visible = now;
            timed_out = 1;
            break;
        }
    }

    long latency_us = (t_visible - t_sync) / 1000L;
    printf("[tc01] visibility latency: %ldus  (deadline %luus)  polls=%d  entries=%d/%d\n",
           latency_us, DEADLINE_US, polls, n < 0 ? 0 : n, NUM_PAGES);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(e);

    int failed = 0;

    if (timed_out || n != NUM_PAGES) {
        printf("[tc01] FAIL: entries never reached %d (got %d) within %ds\n",
               NUM_PAGES, n < 0 ? -1 : n, TIMEOUT_S);
        failed = 1;
    }

    if (!failed && latency_us > (long)DEADLINE_US) {
        printf("[tc01] FAIL: latency %ldus exceeds deadline %luus\n",
               latency_us, DEADLINE_US);
        failed = 1;
    }

    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
