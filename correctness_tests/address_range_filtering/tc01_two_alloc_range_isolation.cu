/*
 * tc01_two_alloc_range_isolation.cu
 *
 * Two managed allocations are written under a single delta tracking session.
 * After cutover+dump, all pages from both allocations appear in the snapshot.
 * Client-side range filtering (by address) is used to verify that each
 * allocation's pages are independently identifiable and do not bleed into each
 * other's address range.
 *
 * NOTE: The old server-side dirty_range filter has been removed from the API.
 * Filtering is now done in the test after dump (client-side). The kernel stores
 * all pages and returns them all on dump; the user selects the range of interest.
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

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void write_pages(int *data, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        data[i] = i + 1;
}

typedef struct {
    int *managed;
    int  device;
} thread_arg_t;

static void *write_thread(void *arg)
{
    thread_arg_t *a = (thread_arg_t *)arg;
    CUDA_CHECK(cudaSetDevice(a->device));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    write_pages<<<1, 256, 0, s>>>(a->managed, NUM_INTS);
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

/* Count pages that fall inside [base, base + NUM_PAGES*PAGE_SIZE). */
static int count_in_alloc(entry_t *e, int n, unsigned long base)
{
    int c = 0;
    for (int i = 0; i < n; i++)
        if (e[i].addr >= base && e[i].addr < base + NUM_PAGES * PAGE_SIZE)
            c++;
    return c;
}

int main(void)
{
    printf("[tc01] two_alloc_range_isolation - %d pages per alloc (delta mode)\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

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
    printf("[tc01] pid=%d alloc_a=0x%lx alloc_b=0x%lx pages=%d each\n",
           getpid(), base_a, base_b, NUM_PAGES);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    thread_arg_t args[2] = { { alloc_a, dev }, { alloc_b, dev } };
    pthread_t threads[2];
    for (int t = 0; t < 2; t++)
        pthread_create(&threads[t], NULL, write_thread, &args[t]);
    for (int t = 0; t < 2; t++)
        pthread_join(threads[t], NULL);
    printf("[tc01] both write threads joined\n");

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-n)); stop_track(); free(e); return 1; }

    /* Client-side range filtering: count pages per allocation. */
    int in_a = count_in_alloc(e, n, base_a);
    int in_b = count_in_alloc(e, n, base_b);

    printf("[tc01] total dump entries: %d\n", n);
    printf("[tc01] in alloc_a: %d (want %d)\n", in_a, NUM_PAGES);
    printf("[tc01] in alloc_b: %d (want %d)\n", in_b, NUM_PAGES);

    /* Verify no overlap: no page appears in both allocations' ranges. */
    int overlap = 0;
    for (int i = 0; i < n; i++) {
        int in_range_a = (e[i].addr >= base_a && e[i].addr < base_a + NUM_PAGES * PAGE_SIZE);
        int in_range_b = (e[i].addr >= base_b && e[i].addr < base_b + NUM_PAGES * PAGE_SIZE);
        if (in_range_a && in_range_b) overlap++;
    }

    stop_track();
    CUDA_CHECK(cudaFree(alloc_a));
    CUDA_CHECK(cudaFree(alloc_b));
    free(e);

    int failed = (in_a != NUM_PAGES || in_b != NUM_PAGES || overlap != 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (in_a != NUM_PAGES) printf("[tc01]   alloc_a: %d/%d pages missing\n", NUM_PAGES - in_a, NUM_PAGES);
    if (in_b != NUM_PAGES) printf("[tc01]   alloc_b: %d/%d pages missing\n", NUM_PAGES - in_b, NUM_PAGES);
    if (overlap)            printf("[tc01]   %d pages appear in both ranges (impossible)\n", overlap);
    return failed;
}
