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
#define PROCFS_RANGE  "/proc/driver/nvidia-uvm/dirty_range"

#define NUM_PAGES   8
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
    if (fd < 0) { 
        perror(path); exit(1); 
    }
    write(fd, val, strlen(val));
    close(fd);
}

static void set_query_pid(pid_t p)
{
    char b[32];
    snprintf(b, sizeof(b), "%d\n", p);
    procfs_write(PROCFS_QUERY, b);
}

static void start_track(pid_t p) {
    char b[32];
    snprintf(b, sizeof(b), "%d\n", p);
    procfs_write(PROCFS_START, b);
}

static void stop_track(pid_t p) {
    char b[32];
    snprintf(b, sizeof(b), "%d\n", p);
    procfs_write(PROCFS_STOP, b);
}

static void set_range_full(void) {
    procfs_write(PROCFS_RANGE, "0x0 0xffffffffffffffff\n");
}

static int read_pages(entry_t *out, int max)
{
    FILE *f = fopen(PROCFS_PAGES, "r");
    if (!f) return -1;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) { 
                fclose(f); return -2; 
            }
            continue;
        }
        if (n < max && sscanf(line, "0x%lx %lu %d", &out[n].addr, &out[n].ts, &out[n].pid) == 3) n++;
    }
    fclose(f);
    return n;
}

int main(void)
{
    printf("[tc03] stop_start_empty_table\n");

    if (geteuid() != 0) { 
        fprintf(stderr, "ERROR: must run as root\n"); return 1; 
    }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    set_query_pid(pid);

    /* ---- init table A ---- */
    start_track(pid);
    printf("[tc03] table A initialized\n");

    /* ---- write: populate table A ---- */
    gpu_write<<<1, 1>>>(managed, NUM_INTS);
    CUDA_CHECK(cudaDeviceSynchronize());

    entry_t e[MAX_ENTRIES];
    set_range_full();
    int n_before = read_pages(e, MAX_ENTRIES);
    printf("[tc03] table A entries before stop: %d\n", n_before);
    for (int i = 0; i < n_before; i++)
        printf("[tc03]   entry %d: addr=0x%lx ts=%lu pid=%d\n", i, e[i].addr, e[i].ts, e[i].pid);

    /* ---- destroy table A ---- */
    stop_track(pid);
    printf("[tc03] table A destroyed\n");

    /* ---- create table B (fresh) ---- */
    start_track(pid);
    printf("[tc03] table B initialized — no writes will follow\n");

    /* ---- query table B: must be 0 entries (active but empty) ---- */
    set_range_full();
    int n_after = read_pages(e, MAX_ENTRIES);
    printf("[tc03] table B entries (no writes since start): %d\n", n_after);
    

    /* ---- teardown ---- */
    stop_track(pid);
    CUDA_CHECK(cudaFree(managed));

    /*
     * PASS conditions:
     *   n_before > 0  : table A recorded at least some pages
     *   n_after == 0  : table B is empty (no faults since restart)
     */
    int failed = (n_before <= 0 || n_after != 0);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (n_before <= 0)
        printf("[tc03]   table A was empty before stop (expected >0 entries)\n");
    if (n_after == -2)
        printf("[tc03]   table B reports not-active (tracking should be active)\n");
    else if (n_after != 0)
        printf("[tc03]   table B has %d stale entries (expected 0)\n", n_after);
    return failed;
}
