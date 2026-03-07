/*
 * test_dirty_tracking.cu
 *
 * Tests that GPU read faults produce READ_ONLY mappings under dirty tracking.
 *
 * Run as root. Check dmesg for [DIRTY-TRACK] perm: lines after running.
 *
 * Expected output in dmesg:
 *   After "READ PHASE":  all access=1 (READ)  -> prot=1 (READ_ONLY)
 *   After "WRITE PHASE": all access=2 (WRITE) -> prot=2 (READ_WRITE)
 *
 * Build:
 *   nvcc -o test_dirty_tracking test_dirty_tracking.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

#define PROCFS_DIRTY_START "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_DIRTY_STOP  "/proc/driver/nvidia-uvm/dirty_pids_stop_track"

#define NUM_PAGES   1
#define PAGE_SIZE   4096
#define NUM_INTS    (NUM_PAGES * PAGE_SIZE / sizeof(int))

#define CUDA_CHECK(call)                                                     \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(1);                                                    \
        }                                                               \
    } while (0)

/* ---------- GPU kernels ------------------------------------------------- */

/* Read every element; prevents the compiler from optimizing away the reads. */
__global__ void kernel_read(const int *data, int n, volatile int *sink)
{
    int acc = 0;
    for (int i = 0; i < n; i++)
        acc += data[i];
    *sink = acc;
}

/* Write a distinct value to every element. */
__global__ void kernel_write(int *data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = i + 1;
}

/* ---------- Helpers ------------------------------------------------------ */

static void procfs_write(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        perror(path);
        exit(1);
    }
    if (write(fd, val, strlen(val)) < 0) {
        perror("write procfs");
        exit(1);
    }
    close(fd);
}

static void start_tracking(void)
{
    procfs_write(PROCFS_DIRTY_START, "1\n");
    printf("[test] dirty tracking started for pid %d\n", getpid());
}

static void stop_tracking(void)
{
    procfs_write(PROCFS_DIRTY_STOP, "1\n");
    printf("[test] dirty tracking stopped for pid %d\n", getpid());
}

/* Force pages back to CPU so the GPU will fault fresh on the next access. */
static void migrate_to_cpu(void *ptr, size_t bytes)
{
	cudaMemLocation cpu_loc;
	cpu_loc.type = cudaMemLocationTypeHost;  // For any CPU
	cpu_loc.id = 0;  // Ignored for Host, but set to 0

    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cpu_loc, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ---------- Main --------------------------------------------------------- */

int main(void)
{
	printf("[test] PID = %d\n", getpid());
    int  *managed = NULL;
    int  *sink    = NULL;
    int   dev;

    CUDA_CHECK(cudaGetDevice(&dev));

    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&sink,    sizeof(int)));

    /* CPU-side initialise so pages start resident on the CPU. */
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    printf("[test] --- Allocated managed at 0x%lx ---\n", managed);
    printf("[test] --- Sink at 0x%lx ---\n", sink);
    *sink = 0;
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ------------------------------------------------------------------ */
    printf("\n[test] === enabling dirty tracking ===\n");
    start_tracking();

    /* ------------------------------------------------------------------ */
    printf("[test] --- READ PHASE ---\n");
    printf("[test] migrating pages to CPU first (ensures GPU will fault)\n");
    migrate_to_cpu(managed, NUM_PAGES * PAGE_SIZE);

    printf("[test] launching read kernel (%d pages) ...\n", NUM_PAGES);
    kernel_read<<<1, 1>>>(managed, NUM_INTS, sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[test] read kernel done (sink=%d)\n", *sink);
    printf("[test] >>> expect dmesg: access=1 -> prot=1 (READ_ONLY) for all %d pages\n\n",
           NUM_PAGES);

    /* /1* ------------------------------------------------------------------ *1/ */
    /* printf("[test] --- WRITE PHASE ---\n"); */
    /* printf("[test] migrating pages to CPU again (re-arm faults)\n"); */
    /* migrate_to_cpu(managed, NUM_PAGES * PAGE_SIZE); */

    /* printf("[test] launching write kernel (%d pages) ...\n", NUM_PAGES); */
    /* kernel_write<<<1, 1>>>(managed, NUM_INTS); */
    /* CHECK(cudaDeviceSynchronize()); */
    /* printf("[test] write kernel done\n"); */
    /* printf("[test] >>> expect dmesg: access=2 -> prot=2 (READ_WRITE) for all %d pages\n\n", */
    /*        NUM_PAGES); */

    /* ------------------------------------------------------------------ */
    printf("[test] === disabling dirty tracking ===\n");
    stop_tracking();

    CUDA_CHECK(cudaFree(managed));
    CUDA_CHECK(cudaFree(sink));

    printf("[test] done — run: sudo dmesg | grep 'DIRTY-TRACK.*perm'\n");
    return 0;
}
