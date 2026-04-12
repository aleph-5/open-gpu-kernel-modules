#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

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

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_one_page(int *data, int page_idx)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++)
        data[page_idx * ipp + i] = page_idx * 1000 + i + 1;
}

static int procfs_write_exact(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -errno;
    ssize_t n = write(fd, val, strlen(val));
    int saved = errno;
    close(fd);
    if (n < 0) return -saved;
    return 0;
}

static int start_track_delta(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d delta", getpid());
    return procfs_write_exact(PROCFS_START, buf);
}

static int stop_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_STOP, buf);
}

static int cutover(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_CUTOVER, buf);
}

static int read_dump(entry_t *out, int max)
{
    FILE *f = fopen(PROCFS_DUMP, "r");
    if (!f) return -errno;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (n < max && sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2)
            n++;
    }
    if (ferror(f)) { int saved = errno; fclose(f); return -saved; }
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

int main(void)
{
    printf("[tc04] multi_reinit_stress (%d cycles, %d pages, delta mode)\n",
           NUM_CYCLES, NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }
    if (NUM_CYCLES > NUM_PAGES) { fprintf(stderr, "ERROR: NUM_CYCLES > NUM_PAGES\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int total_errors = 0;

    for (int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        int cur_page  = cycle;
        int prev_page = cycle - 1;

        int rc = start_track_delta();
        if (rc) {
            fprintf(stderr, "[tc04] cycle %d: start failed: %s\n", cycle, strerror(-rc));
            total_errors++;
            break;
        }

        gpu_write_one_page<<<1, 1>>>(managed, cur_page);
        CUDA_CHECK(cudaDeviceSynchronize());

        rc = cutover();
        if (rc) {
            fprintf(stderr, "[tc04] cycle %d: cutover failed: %s\n", cycle, strerror(-rc));
            stop_track();
            total_errors++;
            break;
        }

        entry_t e[MAX_ENTRIES];
        int n = read_dump(e, MAX_ENTRIES);
        if (n < 0) {
            fprintf(stderr, "[tc04] cycle %d: dump failed: %s\n", cycle, strerror(-n));
            stop_track();
            total_errors++;
            break;
        }

        int cur_present  = page_tracked(e, n, (unsigned long)managed + cur_page  * PAGE_SIZE);
        int prev_present = (prev_page >= 0)
                         ? page_tracked(e, n, (unsigned long)managed + prev_page * PAGE_SIZE)
                         : 0;

        int cycle_err = 0;
        if (!cur_present)  { printf("[tc04]   cycle %d: page[%d] MISSING\n", cycle, cur_page);  cycle_err++; }
        if ( prev_present) { printf("[tc04]   cycle %d: page[%d] LINGERED\n", cycle, prev_page); cycle_err++; }

        printf("[tc04] cycle %d: page[%d]=%s page[%d]=%s entries=%d %s\n",
               cycle, cur_page,  cur_present  ? "present" : "MISSING",
                      prev_page, prev_present ? "LINGERING" : "absent",
               n, cycle_err ? "FAIL" : "ok");

        total_errors += cycle_err;

        rc = stop_track();
        if (rc) {
            fprintf(stderr, "[tc04] cycle %d: stop failed: %s\n", cycle, strerror(-rc));
            total_errors++;
            break;
        }
    }

    CUDA_CHECK(cudaFree(managed));

    printf("[tc04] %s (%d error(s) across %d cycles)\n",
           total_errors ? "FAIL" : "PASS", total_errors, NUM_CYCLES);
    return total_errors > 0 ? 1 : 0;
}
