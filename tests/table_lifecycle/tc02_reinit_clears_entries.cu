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

__global__ void gpu_write_range(int *data, int ps, int pe) {
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = ps; p < pe; p++)
        for (int i = 0; i < ipp; i++)
            data[p * ipp + i] = p * ipp + i + 1;
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
        if (n < max && sscanf(line, "0x%lx %lu %d", &out[n].addr, &out[n].ts, &out[n].pid) == 3)
            n++;
    }
    fclose(f);
    return n;
}

static int page_tracked(entry_t *e, int n, unsigned long a) {
    unsigned long pa = a & ~(unsigned long)(PAGE_SIZE - 1);
    for (int i = 0; i < n; i++) {
        if (e[i].addr == pa) {
            return 1;
        }
    }
    return 0;
}

int main(void) {
    printf("[tc02] reinit_clears_entries\n");

    if (geteuid() != 0) { 
        fprintf(stderr, "ERROR: must run as root\n"); return 1; 
    }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc02] process pid=%d\n", pid);
    set_query_pid(pid);
    int half = NUM_PAGES / 2;

    procfs_write(PROCFS_START, "1\n");
    printf("[tc02] initial start_track\n");

    gpu_write_range<<<1, 100>>>(managed, 0, half);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc02] wrote pages 0..%d (pre-reinit)\n", half - 1);

    procfs_write(PROCFS_START, "1\n");
    printf("[tc02] reinit (start_track again) — old entries should be cleared\n");

    gpu_write_range<<<1, 1>>>(managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc02] wrote pages %d..%d (post-reinit)\n", half, NUM_PAGES - 1);

    entry_t e[MAX_ENTRIES];
    int n = read_pages(e, MAX_ENTRIES);
    printf("[tc02] query returned %d entries\n", n);
    for (int i = 0; i < n; i++)
        printf("[tc02]   entry %d: addr=0x%lx ts=%lu pid=%d\n",
               i, e[i].addr, e[i].ts, e[i].pid);

    int ghost = 0;   /* first-half pages that survived reinit (bad) */
    int present = 0; /* second-half pages correctly recorded */
    int missing = 0; /* second-half pages not recorded (bad) */

    for (int p = 0; p < half; p++)
        if (page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) ghost++;
    for (int p = half; p < NUM_PAGES; p++) {
        if ( page_tracked(e, n, (unsigned long)managed + p * PAGE_SIZE)) present++;
        else missing++;
    }

    printf("[tc02] ghost=%d missing=%d present=%d (total entries=%d)\n",
           ghost, missing, present, n);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));

    int failed = (n < 0 || ghost > 0 || missing > 0);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (n < 0)     printf("[tc02]   procfs read failed (n=%d)\n", n);
    if (ghost > 0) printf("[tc02]   %d pre-reinit pages survived (should have been cleared)\n", ghost);
    if (missing > 0) printf("[tc02]   %d post-reinit pages not recorded\n", missing);
    return failed;
}
