/*
 * tc05_filter_before_start_track.cu
 *
 * Verifies that ALL pages written after start_track appear in the dump,
 * regardless of when the pages' PTEs were set up (before or after start).
 *
 * Start tracking, then let four CPU threads write ALL pages (both halves)
 * concurrently.  Take a single cutover+dump and verify both halves appear.
 *
 * This confirms the recording path inserts unconditionally: there is no
 * insert-time filter that could silently drop pages.
 *
 * (The old server-side dirty_range filter and its "set before start" behavior
 * have been removed from the API; this test validates the replacement invariant.)
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

#define NUM_THREADS      4
#define PAGES_PER_THREAD 8
#define NUM_PAGES        (NUM_THREADS * PAGES_PER_THREAD)
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

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void write_range(int *base, int page_start, int page_end)
{
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

static void *write_thread(void *arg)
{
    thread_arg_t *a = (thread_arg_t *)arg;
    CUDA_CHECK(cudaSetDevice(a->device));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    write_range<<<PAGES_PER_THREAD, 256, 0, s>>>(a->managed_base, a->page_start, a->page_end);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
    return NULL;
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

static int count_in_half(entry_t *e, int n, unsigned long base, int second_half)
{
    unsigned long lo = base + (second_half ? (unsigned long)HALF_PAGES * PAGE_SIZE : 0);
    unsigned long hi = lo + (unsigned long)HALF_PAGES * PAGE_SIZE;
    int c = 0;
    for (int i = 0; i < n; i++)
        if (e[i].addr >= lo && e[i].addr < hi) c++;
    return c;
}

int main(void)
{
    printf("[tc05] all_pages_captured - %d pages (%d per thread), %d threads (delta mode)\n",
           NUM_PAGES, PAGES_PER_THREAD, NUM_THREADS);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc05] pid=%d base=0x%lx\n", getpid(), base);
    printf("[tc05] first_half=[0x%lx, 0x%lx)  second_half=[0x%lx, 0x%lx)\n",
           base, base + HALF_PAGES * PAGE_SIZE,
           base + HALF_PAGES * PAGE_SIZE, base + NUM_PAGES * PAGE_SIZE);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] start failed: %s\n", strerror(-rc)); return 1; }
    printf("[tc05] start_track issued\n");

    /* Write ALL pages concurrently - both halves. */
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

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc05] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc05] dump failed: %s\n", strerror(-n)); stop_track(); free(e); return 1; }

    int fh = count_in_half(e, n, base, 0);
    int sh = count_in_half(e, n, base, 1);
    printf("[tc05] dump: total=%d first_half=%d (want %d) second_half=%d (want %d)\n",
           n, fh, HALF_PAGES, sh, HALF_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));
    free(e);

    int failed = (fh != HALF_PAGES || sh != HALF_PAGES);
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (fh != HALF_PAGES) printf("[tc05]   first_half: %d/%d missing\n", HALF_PAGES - fh, HALF_PAGES);
    if (sh != HALF_PAGES) printf("[tc05]   second_half: %d/%d missing\n", HALF_PAGES - sh, HALF_PAGES);
    return failed;
}
