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

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_range(int *base, int start_page, int npages)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = 0; p < npages; p++)
        for (int i = 0; i < ipp; i++)
            base[(start_page + p) * ipp + i] = (start_page + p) * 100 + i;
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

static int count_in_range(entry_t *e, int n, unsigned long base,
                           int start_page, int npages)
{
    int found = 0;
    for (int p = 0; p < npages; p++) {
        unsigned long pa = base + (start_page + p) * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc05] epoch_isolation (%d pages per epoch, delta mode)\n", HALF_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc05] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    /* Epoch 1: write first half. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] start1 failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, 0, HALF_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] epoch 1: wrote pages 0..%d\n", HALF_PAGES - 1);

    rc = stop_track();
    if (rc) { fprintf(stderr, "[tc05] stop1 failed: %s\n", strerror(-rc)); return 1; }
    printf("[tc05] epoch 1: stopped\n");

    /* Write second half while tracking is OFF - must not appear in epoch 2. */
    gpu_write_range<<<1, 32>>>(managed, HALF_PAGES, HALF_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] inter-epoch: wrote pages %d..%d (tracking off)\n",
           HALF_PAGES, NUM_PAGES - 1);

    /* Epoch 2: fresh start, write second half again. */
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] start2 failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, HALF_PAGES, HALF_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[tc05] epoch 2: wrote pages %d..%d\n", HALF_PAGES, NUM_PAGES - 1);

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc05] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc05] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }
    printf("[tc05] epoch 2 dump: %d entries\n", n);

    int epoch1_present = count_in_range(e, n, (unsigned long)managed, 0, HALF_PAGES);
    int epoch2_present = count_in_range(e, n, (unsigned long)managed, HALF_PAGES, HALF_PAGES);
    printf("[tc05] epoch1 pages in dump: %d/%d (expected 0)\n", epoch1_present, HALF_PAGES);
    printf("[tc05] epoch2 pages in dump: %d/%d (expected %d)\n",
           epoch2_present, HALF_PAGES, HALF_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (epoch1_present != 0 || epoch2_present != HALF_PAGES);
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (epoch1_present != 0)          printf("[tc05]   %d epoch-1 pages bled into epoch 2\n", epoch1_present);
    if (epoch2_present != HALF_PAGES) printf("[tc05]   epoch 2 captured %d/%d pages\n", epoch2_present, HALF_PAGES);
    return failed;
}
