/*
 * tc06_concurrent_range_race.cu — address_range_filtering tests
 *
 * dirty_query_start and dirty_query_end are plain unsigned long with no
 * spinlock — they are updated by two separate stores inside dirty_range_write:
 *
 *   sscanf(kbuf, "%lx %lx", &dirty_query_start, &dirty_query_end);
 *
 * On a multi-core machine a concurrent reader can observe dirty_query_start
 * after the first store but before the second, producing an intermediate
 * range that is neither the intended A nor the intended B.
 *
 * Setup:
 *   32 managed pages; all GPU-written to the xarray before the race begins.
 *   Range A = first half  [base,               base + 16*PAGE_SIZE)
 *   Range B = second half [base + 16*PAGE_SIZE, base + 32*PAGE_SIZE)
 *
 * Race (two CPU threads, no GPU involvement):
 *   Writer thread: alternates set_range(B→A), set_range(A→B), N_ITERS_WRITER
 *                  times as fast as possible.
 *   Reader thread: reads dirty_pages N_ITERS_READER times, classifies each:
 *     clean_A — only first-half pages   (coherent range A)
 *     clean_B — only second-half pages  (coherent range B)
 *     empty   — 0 pages (kernel rejected inverted/zero intermediate range)
 *     torn    — BOTH halves present (start from A, end from B — full range
 *               slipped through because end > start passed validation)
 *
 * The torn case is the interesting one: it means dirty_query_start was
 * already overwritten to A.start (low) while dirty_query_end still held
 * B.end (high), exposing the full allocation.
 *
 * FAIL if torn > 0: the race is empirically confirmed and the reported
 * page set is incorrect for either intended range.
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

#define NUM_PAGES        32
#define HALF_PAGES       (NUM_PAGES / 2)
#define PAGE_SIZE        4096
#define INTS_PER_PAGE    (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES      4096

#define N_ITERS_WRITER   500000
#define N_ITERS_READER   10000

#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void write_all(int *base, int num_pages) {
    int pg = blockIdx.x;
    if (pg >= num_pages) return;
    int *p = base + pg * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = pg * 1000 + i + 1;
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

typedef struct {
    char bufA[64];
    char bufB[64];
    volatile int stop;
} writer_arg_t;

typedef struct {
    unsigned long base;
    long          clean_a;
    long          clean_b;
    long          empty;
    long          torn;
    long          error;
    volatile int *stop;
} reader_arg_t;

static void *writer_thread(void *arg) {
    writer_arg_t *a = (writer_arg_t *)arg;
    size_t lenA = strlen(a->bufA);
    size_t lenB = strlen(a->bufB);

    for (int i = 0; i < N_ITERS_WRITER; i++) {
        const char *buf = (i & 1) ? a->bufB : a->bufA;
        size_t len      = (i & 1) ? lenB    : lenA;
        int fd = open(PROCFS_RANGE, O_WRONLY);
        if (fd >= 0) { write(fd, buf, len); close(fd); }
    }
    a->stop = 1;
    return NULL;
}

static int read_dirty_classify(unsigned long base,
                               long *fh_out, long *sh_out) {
    FILE *f = fopen(PROCFS_PAGES, "r");
    if (!f) return -1;
    long fh = 0, sh = 0;
    char line[256];
    unsigned long lo = base;
    unsigned long hi = base + (unsigned long)HALF_PAGES * PAGE_SIZE;
    unsigned long top = base + (unsigned long)NUM_PAGES  * PAGE_SIZE;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) { fclose(f); return -2; }
            continue;
        }
        unsigned long addr; unsigned long ts; int pid;
        if (sscanf(line, "0x%lx %lu %d", &addr, &ts, &pid) != 3) continue;
        if (addr >= lo && addr < hi)  fh++;
        if (addr >= hi && addr < top) sh++;
    }
    fclose(f);
    *fh_out = fh;
    *sh_out = sh;
    return 0;
}

static void *reader_thread(void *arg) {
    reader_arg_t *a = (reader_arg_t *)arg;

    for (int i = 0; i < N_ITERS_READER; i++) {
        if (*a->stop) break;
        long fh = 0, sh = 0;
        int rc = read_dirty_classify(a->base, &fh, &sh);
        if (rc < 0) { a->error++; continue; }
        if      (fh > 0 && sh == 0) a->clean_a++;
        else if (fh == 0 && sh > 0) a->clean_b++;
        else if (fh == 0 && sh == 0) a->empty++;
        else                         a->torn++;
    }
    return NULL;
}

int main(void) {
    printf("[tc06] concurrent_range_race — %d pages, writer=%d iters, reader=%d iters\n",
           NUM_PAGES, N_ITERS_WRITER, N_ITERS_READER);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    pid_t pid = getpid();
    printf("[tc06] pid=%d  base=0x%lx\n", pid, base);
    printf("[tc06] range_A=[0x%lx, 0x%lx)  range_B=[0x%lx, 0x%lx)\n",
           base,
           base + (unsigned long)HALF_PAGES * PAGE_SIZE,
           base + (unsigned long)HALF_PAGES * PAGE_SIZE,
           base + (unsigned long)NUM_PAGES  * PAGE_SIZE);

    set_query_pid(pid);
    procfs_write(PROCFS_RANGE, "0x0 0xffffffffffffffff\n");
    procfs_write(PROCFS_START, "1\n");

    /* Write all pages to the xarray before the race. */
    write_all<<<NUM_PAGES, 256>>>(managed, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc06] all %d pages written to xarray\n", NUM_PAGES);

    /* Verify all pages recorded. */
    long fh0 = 0, sh0 = 0;
    read_dirty_classify(base, &fh0, &sh0);
    printf("[tc06] sanity: first_half=%ld  second_half=%ld  (want %d each)\n",
           fh0, sh0, HALF_PAGES);
    if (fh0 != HALF_PAGES || sh0 != HALF_PAGES) {
        printf("[tc06] FAIL: not all pages recorded before race\n");
        procfs_write(PROCFS_STOP, "1\n");
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }

    /* Prepare writer and reader arguments. */
    writer_arg_t warg;
    snprintf(warg.bufA, sizeof(warg.bufA), "0x%lx 0x%lx\n",
             base,
             base + (unsigned long)HALF_PAGES * PAGE_SIZE);
    snprintf(warg.bufB, sizeof(warg.bufB), "0x%lx 0x%lx\n",
             base + (unsigned long)HALF_PAGES * PAGE_SIZE,
             base + (unsigned long)NUM_PAGES  * PAGE_SIZE);
    warg.stop = 0;

    reader_arg_t rarg = {
        .base    = base,
        .clean_a = 0, .clean_b = 0, .empty = 0, .torn = 0, .error = 0,
        .stop    = &warg.stop,
    };

    printf("[tc06] starting race: writer alternates A/B, reader classifies dirty_pages reads\n");

    pthread_t wt, rt;
    pthread_create(&wt, NULL, writer_thread, &warg);
    pthread_create(&rt, NULL, reader_thread, &rarg);
    pthread_join(wt, NULL);
    pthread_join(rt, NULL);

    long total = rarg.clean_a + rarg.clean_b + rarg.empty + rarg.torn + rarg.error;
    printf("[tc06] race done — %ld reads classified:\n", total);
    printf("[tc06]   clean_A (first half only): %ld\n",  rarg.clean_a);
    printf("[tc06]   clean_B (second half only): %ld\n", rarg.clean_b);
    printf("[tc06]   empty   (0 pages — inverted/zero range rejected): %ld\n", rarg.empty);
    printf("[tc06]   torn    (BOTH halves — intermediate range leaked): %ld\n", rarg.torn);
    printf("[tc06]   error   (procfs read failed): %ld\n", rarg.error);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));

    int failed = (rarg.torn > 0);
    if (failed)
        printf("[tc06] FAIL — %ld torn reads observed: dirty_query_start/end race confirmed\n",
               rarg.torn);
    else
        printf("[tc06] PASS — no torn reads observed in this run\n");

    printf("[tc06] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
