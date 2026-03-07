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

__global__ void gpu_write_range(int *base, int start_page, int end_page) {
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = start_page; p < end_page; p++)
        for (int i = 0; i < ipp; i++)
            base[p * ipp + i] = p * 1000 + i;
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

static int count_in_range(entry_t *e, int n, unsigned long base, int start_page, int end_page) {
    int found = 0;
    for (int p = start_page; p < end_page; p++) {
        unsigned long pa = base + p * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void) {
    int half = NUM_PAGES / 2;
    printf("[tc03] query_nondestructive (%d pages, %d per round)\n", NUM_PAGES, half);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc03] pid=%d  alloc=0x%lx\n", pid, (unsigned long)managed);
    set_query_pid(pid);

    procfs_write(PROCFS_START, "1\n");

    /* round 1: write first half */
    gpu_write_range<<<1, 32>>>(managed, 0, half);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc03] wrote pages 0..%d (round 1)\n", half - 1);

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_pages(snap1, MAX_ENTRIES);
    int r1_in_snap1 = count_in_range(snap1, n1, (unsigned long)managed, 0, half);
    int r2_in_snap1 = count_in_range(snap1, n1, (unsigned long)managed, half, NUM_PAGES);
    printf("[tc03] snap1: %d entries  round1=%d/%d  round2=%d/%d (expect %d,0)\n",
           n1, r1_in_snap1, half, r2_in_snap1, half, half);

    /* round 2: write second half */
    gpu_write_range<<<1, 32>>>(managed, half, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc03] wrote pages %d..%d (round 2)\n", half, NUM_PAGES - 1);

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_pages(snap2, MAX_ENTRIES);
    int r1_in_snap2 = count_in_range(snap2, n2, (unsigned long)managed, 0, half);
    int r2_in_snap2 = count_in_range(snap2, n2, (unsigned long)managed, half, NUM_PAGES);
    printf("[tc03] snap2: %d entries  round1=%d/%d  round2=%d/%d (expect %d,%d)\n",
           n2, r1_in_snap2, half, r2_in_snap2, half, half, half);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));

    /* snap1 must have round-1 pages but not round-2 (not yet written)
     * snap2 must have BOTH rounds (query is non-destructive) */
    int failed = (n1 < 0 || n2 < 0
                  || r1_in_snap1 != half
                  || r2_in_snap1 != 0
                  || r1_in_snap2 != half
                  || r2_in_snap2 != half);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (n1 < 0) printf("[tc03]   snap1 failed (n1=%d)\n", n1);
    if (n2 < 0) printf("[tc03]   snap2 failed (n2=%d)\n", n2);
    if (r1_in_snap1 != half)  printf("[tc03]   snap1 missing round-1 pages (%d/%d)\n", r1_in_snap1, half);
    if (r2_in_snap1 != 0)     printf("[tc03]   snap1 has round-2 pages before they were written (%d)\n", r2_in_snap1);
    if (r1_in_snap2 != half)  printf("[tc03]   snap2 missing round-1 pages after second query (%d/%d) — query was destructive\n", r1_in_snap2, half);
    if (r2_in_snap2 != half)  printf("[tc03]   snap2 missing round-2 pages (%d/%d)\n", r2_in_snap2, half);
    return failed;
}
