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

#define NUM_PAGES   100
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

__global__ void gpu_write(int *data, int n) {
    for (int i = 0; i < n; i++) data[i] = i + 1;
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

static int page_tracked(entry_t *e, int n, unsigned long a) {
    unsigned long pa = a & ~(unsigned long)(PAGE_SIZE - 1);
    for (int i = 0; i < n; i++)
        if (e[i].addr == pa) return 1;
    return 0;
}

int main(void) {
    printf("[tc01] basic_lifecycle\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc01] PID: %d\n", pid);
    set_query_pid(pid);

    procfs_write(PROCFS_START, "1\n");
    printf("[tc01] table initialized (pid=%d)\n", pid);

    gpu_write<<<1, 100>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    entry_t e[MAX_ENTRIES];
    int n = read_pages(e, MAX_ENTRIES);
    int miss = 0;
    for (int p = 0; p < NUM_PAGES; p++)
        if (!page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) miss++;
    printf("[tc01] query: %d entries, %d/%d pages missing\n", n, miss, NUM_PAGES);
    for (int i = 0; i < n; i++)
        printf("[tc01]   entry %d: addr=0x%lx ts=%lu pid=%d\n", i, e[i].addr, e[i].ts, e[i].pid);

    procfs_write(PROCFS_STOP, "1\n");
    printf("[tc01] table destroyed\n");

    entry_t e2[MAX_ENTRIES];
    int n2 = read_pages(e2, MAX_ENTRIES);
    printf("[tc01] post-destroy query returned: %d (expected -2)\n", n2);

    CUDA_CHECK(cudaFree(managed));

    int failed = (n < 0 || miss > 0 || n2 != -2);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (n < 0)    printf("[tc01]   procfs read failed (n=%d)\n", n);
    if (miss > 0) printf("[tc01]   %d pages not recorded\n", miss);
    if (n2 != -2) printf("[tc01]   expected not-active after stop, got %d\n", n2);
    return failed;
}
