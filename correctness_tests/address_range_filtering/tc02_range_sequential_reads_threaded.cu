/*
 * tc02_range_sequential_reads_threaded.cu
 *
 * Four CPU threads launch concurrent GPU writes to distinct quarters of one
 * allocation. After all writes complete, a single cutover+dump is taken.
 * Verifies that concurrent writes from all four threads are fully captured in
 * one snapshot and that client-side quarter-range filtering is correct.
 *
 * (The old server-side dirty_range filter has been removed; range isolation
 * is now done by the test after the dump.)
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

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void write_quarter(int *base, int page_start, int page_end)
{
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

static void *write_thread(void *arg)
{
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

static int count_in_quarter(entry_t *e, int n, unsigned long base, int q)
{
    unsigned long q_base = base + (unsigned long)q * PAGES_PER_QUARTER * PAGE_SIZE;
    unsigned long q_end  = q_base + PAGES_PER_QUARTER * PAGE_SIZE;
    int c = 0;
    for (int i = 0; i < n; i++)
        if (e[i].addr >= q_base && e[i].addr < q_end) c++;
    return c;
}

int main(void)
{
    printf("[tc02] range_sequential_reads_threaded - %d quarters x %d pages (delta mode)\n",
           NUM_QUARTERS, PAGES_PER_QUARTER);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc02] pid=%d base=0x%lx total_pages=%d\n", getpid(), base, TOTAL_PAGES);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); return 1; }

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

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] dump failed: %s\n", strerror(-n)); stop_track(); free(e); return 1; }

    printf("[tc02] dump: %d total entries (expected %d)\n", n, TOTAL_PAGES);

    int failed = 0;
    for (int q = 0; q < NUM_QUARTERS; q++) {
        int present = count_in_quarter(e, n, base, q);
        printf("[tc02]   Q%d: %d/%d pages captured\n", q, present, PAGES_PER_QUARTER);
        if (present != PAGES_PER_QUARTER) {
            printf("[tc02]   FAIL Q%d: %d/%d pages missing\n",
                   q, PAGES_PER_QUARTER - present, PAGES_PER_QUARTER);
            failed = 1;
        }
    }

    stop_track();
    CUDA_CHECK(cudaFree(managed));
    free(e);

    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
