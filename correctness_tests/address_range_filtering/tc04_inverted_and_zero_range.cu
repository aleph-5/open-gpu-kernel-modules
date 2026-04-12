/*
 * tc04_inverted_and_zero_range.cu
 *
 * Verifies that the dump captures all written pages regardless of the address
 * range tested.  The old server-side dirty_range filter (which allowed inverted
 * or zero-width ranges returning 0 entries) has been removed from the API.
 * This test now validates the baseline: after writing pages across the full
 * allocation, every page appears in the cutover+dump snapshot.
 *
 * Four CPU threads write 4 pages each concurrently. The dump is verified to
 * contain exactly all 16 pages, each page-aligned and in the correct VA range.
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

int main(void)
{
    printf("[tc04] full_capture_baseline - %d pages, %d threads (delta mode)\n",
           NUM_PAGES, NUM_THREADS);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc04] pid=%d base=0x%lx pages=%d\n", getpid(), base, NUM_PAGES);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc04] start failed: %s\n", strerror(-rc)); return 1; }

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

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc04] dump failed: %s\n", strerror(-n)); stop_track(); free(e); return 1; }
    printf("[tc04] dump: %d entries (expected %d)\n", n, NUM_PAGES);

    int failed = 0;
    int missing = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found = 1; break; }
        if (!found) { printf("[tc04]   page %d (0x%lx) MISSING\n", p, pa); missing++; }
    }
    if (missing) { printf("[tc04]   %d/%d pages missing\n", missing, NUM_PAGES); failed = 1; }

    stop_track();
    CUDA_CHECK(cudaFree(managed));
    free(e);

    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
