/*
 * tc05_two_allocs_one_table.cu
 *
 * Table lifecycle test 5: two independent managed allocations share
 * a single tracking table instance.
 *
 * A single start_track initializes one table. GPU writes to both
 * allocations must both appear in that table. The table's full
 * lifecycle (init, record from two separate VA ranges, query, destroy)
 * is exercised across two disjoint regions.
 *
 * Sequence:
 *   start_track         (one table)
 *   write alloc A       (faults from VA range A)
 *   write alloc B       (faults from VA range B)
 *   query               (both ranges must appear)
 *   stop_track
 *
 * PASS: all NUM_PAGES pages from alloc A and all NUM_PAGES pages from
 *       alloc B appear in dirty_pages under the single table.
 *
 * Build: nvcc -o tc05_two_allocs_one_table tc05_two_allocs_one_table.cu
 * Run:   sudo ./tc05_two_allocs_one_table
 */

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

#define NUM_PAGES   4   /* pages per allocation */
#define PAGE_SIZE   4096
#define NUM_INTS    (NUM_PAGES * PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES 4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void gpu_write(int *data, int n)
{
    for (int i = 0; i < n; i++) data[i] = i + 1;
}

static void procfs_write(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    write(fd, val, strlen(val));
    close(fd);
}

static void set_query_pid(pid_t p)
{
    char b[32];
    snprintf(b, sizeof(b), "%d\n", p);
    procfs_write(PROCFS_QUERY, b);
}

static int read_pages(entry_t *out, int max)
{
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

static int page_tracked(entry_t *e, int n, unsigned long a)
{
    unsigned long pa = a & ~(unsigned long)(PAGE_SIZE - 1);
    for (int i = 0; i < n; i++)
        if (e[i].addr == pa) return 1;
    return 0;
}

static int count_missing(entry_t *e, int n, int *base, int npages)
{
    int miss = 0;
    for (int p = 0; p < npages; p++)
        if (!page_tracked(e, n, (unsigned long)base + p * PAGE_SIZE)) miss++;
    return miss;
}

int main(void)
{
    printf("[tc05] two_allocs_one_table\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, NUM_PAGES * PAGE_SIZE));
    memset(alloc_a, 0, NUM_PAGES * PAGE_SIZE);
    memset(alloc_b, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc05] alloc A: 0x%lx - 0x%lx\n",
           (unsigned long)alloc_a,
           (unsigned long)alloc_a + NUM_PAGES * PAGE_SIZE);
    printf("[tc05] alloc B: 0x%lx - 0x%lx\n",
           (unsigned long)alloc_b,
           (unsigned long)alloc_b + NUM_PAGES * PAGE_SIZE);

    pid_t pid = getpid();
    set_query_pid(pid);

    /* ---- single init ---- */
    procfs_write(PROCFS_START, "1\n");
    printf("[tc05] single table initialized (pid=%d)\n", pid);

    /* ---- write both allocations ---- */
    gpu_write<<<1, 1>>>(alloc_a, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] wrote alloc A\n");

    gpu_write<<<1, 1>>>(alloc_b, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] wrote alloc B\n");

    /* ---- query ---- */
    entry_t e[MAX_ENTRIES];
    int n = read_pages(e, MAX_ENTRIES);

    int miss_a = count_missing(e, n, alloc_a, NUM_PAGES);
    int miss_b = count_missing(e, n, alloc_b, NUM_PAGES);

    printf("[tc05] total entries: %d\n", n);
    printf("[tc05] alloc A: %d/%d pages present\n", NUM_PAGES - miss_a, NUM_PAGES);
    printf("[tc05] alloc B: %d/%d pages present\n", NUM_PAGES - miss_b, NUM_PAGES);

    /* ---- destroy ---- */
    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(alloc_a));
    CUDA_CHECK(cudaFree(alloc_b));

    int failed = (n < 0 || miss_a > 0 || miss_b > 0);
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (n < 0)     printf("[tc05]   procfs read failed (n=%d)\n", n);
    if (miss_a > 0) printf("[tc05]   alloc A: %d pages missing\n", miss_a);
    if (miss_b > 0) printf("[tc05]   alloc B: %d pages missing\n", miss_b);
    return failed;
}
