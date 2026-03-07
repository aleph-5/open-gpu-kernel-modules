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

/* Two non-overlapping halves of a single allocation. */
#define NUM_PAGES     16
#define HALF_PAGES    (NUM_PAGES / 2)
#define PAGE_SIZE     4096
#define MAX_ENTRIES   4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void gpu_write_range(int *base, int start_page, int npages) {
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = 0; p < npages; p++)
        for (int i = 0; i < ipp; i++)
            base[(start_page + p) * ipp + i] = (start_page + p) * 100 + i;
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

static int count_in_range(entry_t *e, int n, unsigned long base,
                           int start_page, int npages) {
    int found = 0;
    for (int p = 0; p < npages; p++) {
        unsigned long pa = base + (start_page + p) * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void) {
    printf("[tc05] epoch_isolation (%d pages per epoch)\n", HALF_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    printf("[tc05] pid=%d  alloc=0x%lx\n", pid, (unsigned long)managed);
    set_query_pid(pid);

    /* ---- epoch 1: write first half ---- */
    procfs_write(PROCFS_START, "1\n");
    gpu_write_range<<<1, 32>>>(managed, 0, HALF_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] epoch 1: wrote pages 0..%d\n", HALF_PAGES - 1);
    procfs_write(PROCFS_STOP, "1\n");
    printf("[tc05] epoch 1: stopped\n");

    /* touch second half now — tracking is OFF, these must not appear in epoch 2 */
    gpu_write_range<<<1, 32>>>(managed, HALF_PAGES, HALF_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] inter-epoch: wrote pages %d..%d (tracking off)\n",
           HALF_PAGES, NUM_PAGES - 1);

    /* ---- epoch 2: write second half under fresh table ---- */
    procfs_write(PROCFS_START, "1\n");
    gpu_write_range<<<1, 32>>>(managed, HALF_PAGES, HALF_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] epoch 2: wrote pages %d..%d\n", HALF_PAGES, NUM_PAGES - 1);

    entry_t e[MAX_ENTRIES];
    int n = read_pages(e, MAX_ENTRIES);
    printf("[tc05] epoch 2 query: %d entries\n", n);

    int epoch1_present = count_in_range(e, n, (unsigned long)managed, 0, HALF_PAGES);
    int epoch2_present = count_in_range(e, n, (unsigned long)managed, HALF_PAGES, HALF_PAGES);
    printf("[tc05] epoch1 pages in table: %d/%d (expected 0)\n", epoch1_present, HALF_PAGES);
    printf("[tc05] epoch2 pages in table: %d/%d (expected %d)\n",
           epoch2_present, HALF_PAGES, HALF_PAGES);

    for (int i = 0; i < n; i++)
        printf("[tc05]   [%d] addr=0x%lx ts=%lu pid=%d\n",
               i, e[i].addr, e[i].ts, e[i].pid);

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));

    int failed = (n < 0 || epoch1_present != 0 || epoch2_present != HALF_PAGES);
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (n < 0)               printf("[tc05]   epoch 2 table not active\n");
    if (epoch1_present != 0) printf("[tc05]   %d epoch-1 pages bled into epoch 2\n", epoch1_present);
    if (epoch2_present != HALF_PAGES)
        printf("[tc05]   epoch 2 only captured %d/%d pages\n", epoch2_present, HALF_PAGES);
    return failed;
}
