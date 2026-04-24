#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE_BYTES 4096

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(1);                                                        \
    }                                                                   \
} while (0)

__global__ void kernel_write(char *buf, long n_pages)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pages)
        buf[idx * PAGE_SIZE_BYTES] = (char)(idx & 0xFF);
}

static long ns_elapsed(struct timespec *a, struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1000000000L + (b->tv_nsec - a->tv_nsec);
}

int main(int argc, char **argv)
{
    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }
    if (!dt_sysfs_exists(DT_PROCFS_START)) {
        fprintf(stderr, "ERROR: %s not found - is the nvidia-uvm module loaded?\n",
                DT_PROCFS_START);
        return 1;
    }

    long n_pages = 100000;
    if (argc > 1)
        n_pages = atol(argv[1]);

    size_t buf_size = (size_t)n_pages * PAGE_SIZE_BYTES;
    printf("[tc01] %ld pages (%.1f MB)\n", n_pages, buf_size / 1e6);

    char *buf;
    CUDA_CHECK(cudaMallocManaged(&buf, buf_size));
    memset(buf, 0, buf_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads = 256;
    int blocks = (n_pages + threads - 1) / threads;

    // warmup: bring pages to GPU
    kernel_write<<<blocks, threads>>>(buf, n_pages);
    CUDA_CHECK(cudaDeviceSynchronize());

    // baseline: data already resident on GPU, no faults
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    kernel_write<<<blocks, threads>>>(buf, n_pages);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long baseline_ns = ns_elapsed(&t0, &t1);
    printf("[tc01] baseline  : %.3f ms\n", baseline_ns / 1e6);

    struct timespec ti0, ti1;
    clock_gettime(CLOCK_MONOTONIC, &ti0);
    int rc = dt_start("delta");
    clock_gettime(CLOCK_MONOTONIC, &ti1);
    if (rc) {
        fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc));
        return 1;
    }
    printf("[tc01] start_path: %.3f ms\n", ns_elapsed(&ti0, &ti1) / 1e6);

    struct timespec t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t2);
    kernel_write<<<blocks, threads>>>(buf, n_pages);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t3);
    long tracked_ns = ns_elapsed(&t2, &t3);

    rc = dt_cutover();
    if (rc) {
        fprintf(stderr, "[tc01] cutover failed: %s\n", strerror(-rc));
        (void)dt_stop();
        return 1;
    }

    long recorded = dt_dump_count_pages_in_range((unsigned long)buf, buf_size, PAGE_SIZE_BYTES);
    if (recorded < 0) {
        fprintf(stderr, "[tc01] dump failed: %s\n", strerror(-(int)recorded));
        (void)dt_stop();
        return 1;
    }

    rc = dt_stop();
    if (rc) {
        fprintf(stderr, "[tc01] stop failed: %s\n", strerror(-rc));
        return 1;
    }

    printf("[tc01] tracked   : %.3f ms\n", tracked_ns / 1e6);
    printf("[tc01] overhead  : %.3f ms  (%.1f%%)\n",
           (tracked_ns - baseline_ns) / 1e6,
           100.0 * (tracked_ns - baseline_ns) / baseline_ns);
    printf("[tc01] recorded  : %ld / %ld pages\n", recorded, n_pages);
    printf("[tc01] missing   : %ld pages\n", n_pages - recorded);
    if (tracked_ns > 0 && recorded > 0)
        printf("[tc01] rate      : %.0f pages/sec\n", recorded / (tracked_ns / 1e9));

    CUDA_CHECK(cudaFree(buf));
    return (recorded == n_pages) ? 0 : 1;
}
