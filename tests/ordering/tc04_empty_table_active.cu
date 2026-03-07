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

#define NUM_PAGES   4
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

__global__ void gpu_write_all(int *data, int n) {
    for (int i = 0; i < n; i++) data[i] = i + 7;
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
    printf("[tc04] empty_table_active\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc04] pid=%d  alloc=0x%lx\n", pid, (unsigned long)managed);
    set_query_pid(pid);

    procfs_write(PROCFS_START, "1\n");

    /* query immediately — no GPU writes yet */
    entry_t pre[MAX_ENTRIES];
    int n_pre = read_pages(pre, MAX_ENTRIES);
    printf("[tc04] pre-write query: %d entries (expected 0, not -2)\n", n_pre);

    /* now write and query again */
    gpu_write_all<<<1, 256>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    entry_t post[MAX_ENTRIES];
    int n_post = read_pages(post, MAX_ENTRIES);
    printf("[tc04] post-write query: %d entries (expected %d)\n", n_post, NUM_PAGES);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));

    /* n_pre==0  → table is active but empty (correct)
     * n_pre==-2 → table not active (bug)
     * n_pre>0   → stale entries in a fresh table (bug)
     * n_post should match NUM_PAGES */
    int failed = (n_pre != 0 || n_post < NUM_PAGES);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (n_pre < 0)       printf("[tc04]   pre-write: table not active (got %d)\n", n_pre);
    if (n_pre > 0)       printf("[tc04]   pre-write: %d stale entries in fresh table\n", n_pre);
    if (n_post < NUM_PAGES) printf("[tc04]   post-write: only %d/%d pages tracked\n", n_post, NUM_PAGES);
    return failed;
}
