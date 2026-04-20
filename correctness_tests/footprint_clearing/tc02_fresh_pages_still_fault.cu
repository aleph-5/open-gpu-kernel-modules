/*
 * tc02_fresh_pages_still_fault
 *
 * stop_track must not spuriously grant READ_WRITE permission to pages that
 * had no GPU mapping before tracking started.  If it did, writes to those
 * pages after stop would be fast (no faults) — indistinguishable from
 * pre-established RW pages.
 *
 * Setup:
 *   data_ref   — written twice before the test; used to obtain T_cached.
 *   data_fresh — never touched by the GPU; receives a start→stop cycle
 *                and then a timed write.
 *
 * Pass condition: T_fresh_after_stop > FAULT_RATIO_MIN/2 * T_cached
 *   (the write is still in the "faulting" regime, not the "cached" regime).
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
    printf("[tc02] fresh_pages_still_fault\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *data_ref   = NULL;
    int *data_fresh = NULL;
    CUDA_CHECK(cudaMallocManaged(&data_ref,   (size_t)NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&data_fresh, (size_t)NUM_PAGES * PAGE_SIZE));

    /* Warm up CUDA context. */
    gpu_write_all<<<1, 1>>>((volatile int *)data_ref, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Establish T_cached on data_ref (two passes; second has no faults). */
    timed_gpu_write(data_ref);
    CUDA_CHECK(cudaDeviceSynchronize());
    float T_cached = timed_gpu_write(data_ref);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Measure T_fault on a separate fresh allocation for calibration. */
    int *data_cal = NULL;
    CUDA_CHECK(cudaMallocManaged(&data_cal, (size_t)NUM_PAGES * PAGE_SIZE));
    float T_fault = timed_gpu_write(data_cal);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(data_cal));

    printf("[tc02]   T_fault=%.3fms  T_cached=%.3fms  ratio=%.1f\n",
           T_fault, T_cached, T_fault / T_cached);

    if (T_fault < FAULT_RATIO_MIN * T_cached) {
        printf("[tc02] SKIP: fault/cached ratio %.1f < %.1f — cannot calibrate\n",
               T_fault / T_cached, FAULT_RATIO_MIN);
        CUDA_CHECK(cudaFree(data_ref));
        CUDA_CHECK(cudaFree(data_fresh));
        return 77;
    }

    /* data_fresh has never been written by the GPU.
     * start→stop: the snapshot is empty, restore is a no-op.
     * Writes to data_fresh after stop must still fault. */
    int rc = start_track();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); return 1; }
    rc = stop_track();
    if (rc) { fprintf(stderr, "[tc02] stop failed: %s\n", strerror(-rc)); return 1; }

    float T_fresh_after = timed_gpu_write(data_fresh);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* If stop had spuriously granted RW, T_fresh_after would be ≈ T_cached.
     * It must remain in the faulting regime: > (FAULT_RATIO_MIN/2) * T_cached. */
    float fault_floor = (FAULT_RATIO_MIN / 2.0f) * T_cached;
    printf("[tc02]   T_fresh_after_stop=%.3fms  fault_floor=%.3fms\n",
           T_fresh_after, fault_floor);

    int failed = (T_fresh_after < fault_floor);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (failed)
        fprintf(stderr,
                "[tc02]   stop granted RW to unmapped pages: write took %.3fms "
                "(expected > %.3fms)\n",
                T_fresh_after, fault_floor);

    CUDA_CHECK(cudaFree(data_ref));
    CUDA_CHECK(cudaFree(data_fresh));
    return failed;
}
