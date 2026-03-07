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
#define NUM_CYCLES  6   

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void gpu_write_one_page(int *data, int page_idx) {
    int ipp = PAGE_SIZE / sizeof(int);
    int base = page_idx * ipp;
    for (int i = 0; i < ipp; i++)
        data[base + i] = page_idx * 1000 + i + 1;
}

static void procfs_write(const char *path, const char *val) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) { 
        perror(path); exit(1); 
    }
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
    printf("[tc04] multi_reinit_stress (%d cycles, %d pages)\n", NUM_CYCLES, NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }
    if (NUM_CYCLES > NUM_PAGES) { fprintf(stderr, "ERROR: NUM_CYCLES > NUM_PAGES\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    pid_t pid = getpid();
    set_query_pid(pid);

    procfs_write(PROCFS_START, "1\n");

    int total_errors = 0;

    for (int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        int cur_page  = cycle;
        int prev_page = cycle - 1;

        if (cycle > 0) {
            procfs_write(PROCFS_START, "1\n");
            printf("[tc04] cycle %d: reinit\n", cycle);
        } else {
            printf("[tc04] cycle %d: initial table\n", cycle);
        }

        gpu_write_one_page<<<1, 1>>>(managed, cur_page);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Query. */
        entry_t e[MAX_ENTRIES];
        int n = read_pages(e, MAX_ENTRIES);

        int cur_present  = page_tracked(e, n, (unsigned long)managed + cur_page  * PAGE_SIZE);
        int prev_present = (prev_page >= 0)
                         ? page_tracked(e, n, (unsigned long)managed + prev_page * PAGE_SIZE)
                         : 0;

        int cycle_err = 0;
        if (!cur_present) { printf("[tc04]   cycle %d: page[%d] missing (should be present)\n", cycle, cur_page); cycle_err++; }
        if ( prev_present) { printf("[tc04]   cycle %d: page[%d] still present (should have been cleared by reinit)\n", cycle, prev_page); cycle_err++; }

        printf("[tc04] cycle %d: page[%d]=%s page[%d]=%s  entries=%d  %s\n",
               cycle,
               cur_page,  cur_present  ? "present" : "MISSING",
               prev_page, prev_present ? "LINGERING" : "absent",
               n, cycle_err ? "FAIL" : "ok");

        total_errors += cycle_err;
    }

    procfs_write(PROCFS_STOP, "1\n");
    CUDA_CHECK(cudaFree(managed));

    printf("[tc04] %s (%d error(s) across %d cycles)\n",
           total_errors ? "FAIL" : "PASS", total_errors, NUM_CYCLES);
    return total_errors > 0 ? 1 : 0;
}
