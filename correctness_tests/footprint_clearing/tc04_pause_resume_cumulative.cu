/*
 * tc04_pause_resume_cumulative
 *
 * Pause→resume must not clobber the pre-tracking snapshot.  resume re-runs
 * the downgrade path; if it overwrites dirty_downgraded_pages with the
 * current pte_bits[GPU_WRITE] (a copy), the original snapshot is lost and
 * stop has nothing to restore.  The fix is to OR into the saved masks so
 * resume only *adds* pages — it never erases the snapshot from first start.
 *
 * Strategy:
 *   start_track  -> snapshot = all_pages (RW), pages downgraded to RO.
 *   pause_track  -> masks untouched.
 *   resume_track -> re-downgrade.  pte_bits[GPU_WRITE] is empty (everything
 *                   is already RO and we wrote nothing in between).
 *                     - copy bug: snapshot := empty.
 *                     - OR  fix : snapshot stays all_pages.
 *   stop_track   -> restore from snapshot.
 *
 *   Then time a write pass.  copy bug -> T_fault.  OR fix -> ~T_cached.
 *
 * Exit: 0 PASS, 1 FAIL, 77 SKIP (calibration ratio too low to distinguish)
 */
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START  "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP   "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_PAUSE  "/proc/driver/nvidia-uvm/dirty_tracking_pause"
#define PROCFS_RESUME "/proc/driver/nvidia-uvm/dirty_tracking_resume"

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

static int pause_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write(PROCFS_PAUSE, buf);
}

static int resume_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write(PROCFS_RESUME, buf);
}

int main(void)
{
    printf("[tc04] pause_resume_cumulative\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *data = NULL;
    CUDA_CHECK(cudaMallocManaged(&data, (size_t)NUM_PAGES * PAGE_SIZE));

    /* Warm up CUDA context. */
    gpu_write_all<<<1, 1>>>((volatile int *)data, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Pass 1: all pages fault RO->RW. */
    float T_fault = timed_gpu_write(data);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Pass 2: pages already RW -- no faults. */
    float T_cached = timed_gpu_write(data);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc04]   T_fault=%.3fms  T_cached=%.3fms  ratio=%.1f\n",
           T_fault, T_cached, T_fault / T_cached);

    if (T_fault < FAULT_RATIO_MIN * T_cached) {
        printf("[tc04] SKIP: fault/cached ratio %.1f < %.1f -- cannot calibrate\n",
               T_fault / T_cached, FAULT_RATIO_MIN);
        CUDA_CHECK(cudaFree(data));
        return 77;
    }

    /* start -> pause -> resume -> stop, no writes between. */
    int rc = start_track();
    if (rc) { fprintf(stderr, "[tc04] start failed: %s\n", strerror(-rc)); return 1; }

    rc = pause_track();
    if (rc) { fprintf(stderr, "[tc04] pause failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    rc = resume_track();
    if (rc) { fprintf(stderr, "[tc04] resume failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    rc = stop_track();
    if (rc) { fprintf(stderr, "[tc04] stop failed: %s\n", strerror(-rc)); return 1; }

    /* Pass 3: must be ~T_cached if the snapshot survived the resume re-downgrade. */
    float T_after = timed_gpu_write(data);
    CUDA_CHECK(cudaDeviceSynchronize());

    float threshold = RESTORED_FACTOR * T_cached;
    printf("[tc04]   T_after_stop=%.3fms  threshold=%.3fms (%.0fx T_cached)\n",
           T_after, threshold, RESTORED_FACTOR);

    int failed = (T_after > threshold);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (failed)
        fprintf(stderr,
                "[tc04]   snapshot lost across pause->resume: write took %.1fx T_cached (limit %.0fx)\n",
                T_after / T_cached, RESTORED_FACTOR);

    CUDA_CHECK(cudaFree(data));
    return failed;
}
