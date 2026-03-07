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

/* Sentinel + 1M-thread flood ordering test.
 *
 * Step 1: write a single sentinel page with a tiny kernel (1 block x 1 thread).
 *         Query and record its timestamp ts_sentinel.
 * Step 2: write NUM_FLOOD_PAGES with 4096 blocks x 256 threads = ~1M threads.
 *         Query again.
 *
 * Ordering invariant:
 *   Every flood entry must have ts >= ts_sentinel.
 *   The sentinel was written BEFORE the flood started, so the tracker must
 *   record flood faults with timestamps >= the sentinel's timestamp.
 *   Any flood entry with ts < ts_sentinel is an ordering violation. */

#define NUM_FLOOD_PAGES   256
#define FLOOD_BLOCKS      4096
#define FLOOD_THREADS     256
#define PAGE_SIZE         4096
#define INTS_PER_PAGE     (PAGE_SIZE / sizeof(int))
#define TOTAL_PAGES       (1 + NUM_FLOOD_PAGES)
#define MAX_ENTRIES       8192

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void write_sentinel(int *page) {
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) page[i] = 0xDEAD0000 + i;
}

/* each block handles one page in the flood region */
__global__ void flood_write(int *base, int npages) {
    int page = blockIdx.x % npages;
    int *p   = base + page * INTS_PER_PAGE;
    for (int i = threadIdx.x; i < (int)INTS_PER_PAGE; i += blockDim.x)
        p[i] = page * 777 + i;
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
    long flood_threads = (long)FLOOD_BLOCKS * FLOOD_THREADS;
    printf("[tc07] sentinel_then_flood: 1 sentinel + %d flood pages, %ld flood threads\n",
           NUM_FLOOD_PAGES, flood_threads);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    /* sentinel page at base, flood pages immediately after */
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, (size_t)TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *sentinel_page = managed;
    int *flood_base    = managed + INTS_PER_PAGE;

    unsigned long sentinel_va = (unsigned long)sentinel_page;
    unsigned long flood_va    = (unsigned long)flood_base;

    pid_t pid = getpid();
    printf("[tc07] pid=%d  sentinel=0x%lx  flood_base=0x%lx\n",
           pid, sentinel_va, flood_va);
    set_query_pid(pid);

    procfs_write(PROCFS_START, "1\n");

    /* step 1: sentinel */
    write_sentinel<<<1, 1>>>(sentinel_page);
    CUDA_CHECK(cudaDeviceSynchronize());

    entry_t e1[MAX_ENTRIES];
    int n1 = read_pages(e1, MAX_ENTRIES);

    unsigned long ts_sentinel = 0;
    for (int i = 0; i < n1; i++)
        if (e1[i].addr == sentinel_va) { ts_sentinel = e1[i].ts; break; }

    printf("[tc07] sentinel ts=%lu  (n1=%d)\n", ts_sentinel, n1);

    /* step 2: flood */
    flood_write<<<FLOOD_BLOCKS, FLOOD_THREADS>>>(flood_base, NUM_FLOOD_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc07] flood complete\n");

    entry_t *e2 = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e2) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n2 = read_pages(e2, MAX_ENTRIES);
    printf("[tc07] post-flood query: %d entries\n", n2);

    /* ordering check: every flood entry must have ts >= ts_sentinel */
    int early_violations = 0;
    int flood_missing    = 0;
    for (int p = 0; p < NUM_FLOOD_PAGES; p++) {
        unsigned long pa = flood_va + p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n2; i++) {
            if (e2[i].addr != pa) continue;
            found = 1;
            if (e2[i].ts < ts_sentinel) {
                printf("[tc07]   flood page %d: ts=%lu < ts_sentinel=%lu (ordering violation)\n",
                       p, e2[i].ts, ts_sentinel);
                early_violations++;
            }
        }
        if (!found) flood_missing++;
    }

    printf("[tc07] flood_missing=%d  ordering_violations=%d\n",
           flood_missing, early_violations);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));
    free(e2);

    int failed = (n1 < 0 || n2 < 0 || ts_sentinel == 0
                  || flood_missing > 0 || early_violations > 0);
    printf("[tc07] %s\n", failed ? "FAIL" : "PASS");
    if (n1 < 0 || n2 < 0)    printf("[tc07]   table not active\n");
    if (ts_sentinel == 0)     printf("[tc07]   sentinel page not recorded\n");
    if (flood_missing > 0)    printf("[tc07]   %d flood pages not recorded\n", flood_missing);
    if (early_violations > 0) printf("[tc07]   %d flood entries have ts < ts_sentinel\n", early_violations);
    return failed;
}
