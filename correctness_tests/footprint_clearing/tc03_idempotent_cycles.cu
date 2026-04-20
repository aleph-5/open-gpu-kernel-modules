/*
 * tc03_idempotent_cycles
 *
 * Multiple consecutive start→stop cycles must each leave pages in the same
 * READ_WRITE state.  If the restore logic is not idempotent — e.g. it fails
 * to re-snapshot on the second start, or it over-revokes on the second stop
 * — a subsequent write pass would be slow (pages at RO, faults).
 *
 * Strategy: after calibrating T_cached and T_fault, run two start→stop
 * cycles back-to-back and assert that the write pass after each is fast.
 *
 * Exit: 0 PASS, 1 FAIL, 77 SKIP
 */
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP  "/proc/driver/nvidia-uvm/dirty_tracking_stop"

#define NUM_PAGES        2048
#define PAGE_SIZE        4096
#define NUM_INTS         ((long)(NUM_PAGES) * PAGE_SIZE / sizeof(int))
#define FAULT_RATIO_MIN  4.0f
#define RESTORED_FACTOR  8.0f

#define CUDA_CHECK(c) do {                                                    \
    cudaError_t _e = (c);                                                     \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(_e));                  \
        exit(1);                                                              \
    }                                                                         \
} while (0)

__global__ void gpu_write_all(volatile int *data, long n)
{
    long idx    = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x  * blockDim.x;
    for (long i = idx; i < n; i += stride)
        data[i] = (int)(i + 1);
}

static float timed_gpu_write(int *data)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    gpu_write_all<<<256, 256>>>((volatile int *)data, NUM_INTS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

static int procfs_write(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -errno;
    ssize_t n = write(fd, val, strlen(val));
    int saved = errno;
    close(fd);
    return (n < 0) ? -saved : 0;
}

static int start_track(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d delta", getpid());
    return procfs_write(PROCFS_START, buf);
}

static int stop_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write(PROCFS_STOP, buf);
}

int main(void)
{
    printf("[tc03] idempotent_cycles\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *data = NULL;
    CUDA_CHECK(cudaMallocManaged(&data, (size_t)NUM_PAGES * PAGE_SIZE));

    /* Warm up CUDA context. */
    gpu_write_all<<<1, 1>>>((volatile int *)data, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Pass 1: all pages fault RO→RW. */
    float T_fault = timed_gpu_write(data);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Pass 2: pages already RW — no faults. */
    float T_cached = timed_gpu_write(data);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc03]   T_fault=%.3fms  T_cached=%.3fms  ratio=%.1f\n",
           T_fault, T_cached, T_fault / T_cached);

    if (T_fault < FAULT_RATIO_MIN * T_cached) {
        printf("[tc03] SKIP: fault/cached ratio %.1f < %.1f — cannot calibrate\n",
               T_fault / T_cached, FAULT_RATIO_MIN);
        CUDA_CHECK(cudaFree(data));
        return 77;
    }

    float threshold = RESTORED_FACTOR * T_cached;
    int failed = 0;

    for (int cycle = 1; cycle <= 2; cycle++) {
        int rc = start_track();
        if (rc) { fprintf(stderr, "[tc03] cycle %d start failed: %s\n", cycle, strerror(-rc)); return 1; }
        rc = stop_track();
        if (rc) { fprintf(stderr, "[tc03] cycle %d stop failed: %s\n", cycle, strerror(-rc)); return 1; }

        float T_after = timed_gpu_write(data);
        CUDA_CHECK(cudaDeviceSynchronize());

        int cycle_failed = (T_after > threshold);
        printf("[tc03]   cycle %d: T_after=%.3fms  threshold=%.3fms  %s\n",
               cycle, T_after, threshold, cycle_failed ? "FAIL" : "ok");
        if (cycle_failed) {
            fprintf(stderr,
                    "[tc03]   cycle %d: permissions not restored: write took %.1fx T_cached (limit %.0fx)\n",
                    cycle, T_after / T_cached, RESTORED_FACTOR);
            failed = 1;
        }
    }

    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    CUDA_CHECK(cudaFree(data));
    return failed;
}
