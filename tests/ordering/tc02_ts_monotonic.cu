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

#define NUM_PAGES   8
#define PAGE_SIZE   4096
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

__global__ void gpu_write_page(int *base, int page_idx) {
    int ipp = PAGE_SIZE / sizeof(int);
    int *p = base + page_idx * ipp;
    for (int i = 0; i < ipp; i++) p[i] = page_idx * 100 + i;
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

/* Compare entries by address for qsort. */
static int cmp_by_addr(const void *a, const void *b) {
    const entry_t *ea = (const entry_t *)a;
    const entry_t *eb = (const entry_t *)b;
    if (ea->addr < eb->addr) return -1;
    if (ea->addr > eb->addr) return  1;
    return 0;
}

int main(void) {
    printf("[tc02] ts_monotonic (%d pages, one kernel per page)\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc02] pid=%d  alloc=0x%lx\n", pid, (unsigned long)managed);
    set_query_pid(pid);

    procfs_write(PROCFS_START, "1\n");

    /* write one page at a time, syncing after each launch to preserve order */
    for (int p = 0; p < NUM_PAGES; p++) {
        gpu_write_page<<<1, 1>>>(managed, p);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("[tc02] all %d pages written sequentially\n", NUM_PAGES);

    entry_t e[MAX_ENTRIES];
    int n = read_pages(e, MAX_ENTRIES);
    printf("[tc02] dirty_pages returned %d entries\n", n);

    int missing = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        unsigned long pa = (unsigned long)managed + p * PAGE_SIZE;
        int found = 0;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found = 1; break; }
        if (!found) {
            printf("[tc02]   page %d (0x%lx) not in table\n", p, pa);
            missing++;
        }
    }

    /* sort by address to correlate with write order, then check timestamps */
    qsort(e, n, sizeof(entry_t), cmp_by_addr);

    int ts_violations = 0;
    for (int i = 1; i < n; i++) {
        if (e[i].ts < e[i-1].ts) {
            printf("[tc02]   ts inversion: entry[%d].ts=%lu < entry[%d].ts=%lu\n",
                   i, e[i].ts, i-1, e[i-1].ts);
            ts_violations++;
        }
    }

    printf("[tc02] missing=%d  ts_violations=%d\n", missing, ts_violations);
    for (int i = 0; i < n; i++)
        printf("[tc02]   [%d] addr=0x%lx ts=%lu pid=%d\n",
               i, e[i].addr, e[i].ts, e[i].pid);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));

    int failed = (n < 0 || missing > 0 || ts_violations > 0);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (n < 0)           printf("[tc02]   table not active\n");
    if (missing > 0)     printf("[tc02]   %d pages missing from table\n", missing);
    if (ts_violations > 0) printf("[tc02]   %d timestamp inversions detected\n", ts_violations);
    return failed;
}
