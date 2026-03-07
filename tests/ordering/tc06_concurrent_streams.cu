#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

#define PROCFS_START  "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP   "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY  "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_PAGES  "/proc/driver/nvidia-uvm/dirty_pages"

/* 4 streams, each writing 64 pages with 256 threads/block = 65,536 threads in flight.
 * All 4 streams are dispatched without sync between them.
 *
 * Ordering check (intra-stream):
 *   Within a single CUDA stream, blocks execute in dispatch order: block 0
 *   before block 1, etc.  So within stream S, page[0] faults before page[63].
 *   The tracker must preserve this: for stream S's pages sorted by page-index,
 *   timestamps must be non-decreasing.
 *
 *   Cross-stream ordering is deliberately interleaved and is NOT checked. */

#define NUM_STREAMS      4
#define PAGES_PER_STREAM 64
#define NUM_PAGES        (NUM_STREAMS * PAGES_PER_STREAM)
#define THREADS_PER_BLOCK 256
#define PAGE_SIZE        4096
#define INTS_PER_PAGE    (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES      8192

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void stream_write(int *base, int npages) {
    int page = blockIdx.x;
    if (page >= npages) return;
    int *p = base + page * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = page * 500 + i + 1;
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

static int read_pages(entry_t *out, int max) {
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

int main(void) {
    printf("[tc06] concurrent_streams: %d streams x %d pages, %d total threads\n",
           NUM_STREAMS, PAGES_PER_STREAM, NUM_STREAMS * PAGES_PER_STREAM * THREADS_PER_BLOCK);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc06] pid=%d  alloc=0x%lx\n", pid, (unsigned long)managed);
    set_query_pid(pid);

    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    procfs_write(PROCFS_START, "1\n");

    for (int s = 0; s < NUM_STREAMS; s++) {
        int *base = managed + s * PAGES_PER_STREAM * INTS_PER_PAGE;
        stream_write<<<PAGES_PER_STREAM, THREADS_PER_BLOCK, 0, streams[s]>>>(base, PAGES_PER_STREAM);
    }
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    printf("[tc06] all streams complete\n");

    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n = read_pages(e, MAX_ENTRIES);
    printf("[tc06] dirty_pages returned %d entries (expected %d)\n", n, NUM_PAGES);

    /* for each stream, collect its entries sorted by page index (= address order)
     * and verify timestamps are non-decreasing */
    int total_missing   = 0;
    int total_inversions = 0;

    for (int s = 0; s < NUM_STREAMS; s++) {
        unsigned long stream_base = (unsigned long)managed + s * PAGES_PER_STREAM * PAGE_SIZE;

        /* gather entries belonging to this stream, indexed by page within stream */
        unsigned long page_ts[PAGES_PER_STREAM];
        int page_found[PAGES_PER_STREAM];
        memset(page_found, 0, sizeof(page_found));

        for (int i = 0; i < n; i++) {
            if (e[i].addr < stream_base) continue;
            unsigned long off = e[i].addr - stream_base;
            if (off >= (unsigned long)PAGES_PER_STREAM * PAGE_SIZE) continue;
            int pg = (int)(off / PAGE_SIZE);
            page_ts[pg]    = e[i].ts;
            page_found[pg] = 1;
        }

        int missing = 0, inversions = 0;
        unsigned long prev_ts = 0;
        for (int pg = 0; pg < PAGES_PER_STREAM; pg++) {
            if (!page_found[pg]) { missing++; continue; }
            if (page_ts[pg] < prev_ts) {
                printf("[tc06]   stream %d page %d: ts=%lu < prev_ts=%lu (inversion)\n",
                       s, pg, page_ts[pg], prev_ts);
                inversions++;
            }
            prev_ts = page_ts[pg];
        }
        printf("[tc06] stream %d: %d/%d pages, %d intra-stream ts inversions\n",
               s, PAGES_PER_STREAM - missing, PAGES_PER_STREAM, inversions);
        total_missing    += missing;
        total_inversions += inversions;
    }

    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(e);

    int failed = (n < 0 || total_missing > 0 || total_inversions > 0);
    printf("[tc06] %s\n", failed ? "FAIL" : "PASS");
    if (n < 0)               printf("[tc06]   table not active\n");
    if (total_missing > 0)   printf("[tc06]   %d pages missing\n", total_missing);
    if (total_inversions > 0) printf("[tc06]   %d intra-stream timestamp inversions\n", total_inversions);
    return failed;
}
