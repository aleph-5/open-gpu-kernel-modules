#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE  4096
#define NUM_PAGES  1024
#define ALLOC_SIZE ((size_t)NUM_PAGES * PAGE_SIZE)
#define NUM_ITERS (100)

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* Read-modify-write: each thread reads a value, adds its index, writes it back */
__global__ void rmw_kernel(int *buf, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        buf[idx] = buf[idx] + idx;
}

int main(void)
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

    pid_t pid = getpid();
    printf("pid: %d\n", pid);

    int *buf;
    CUDA_CHECK(cudaMallocManaged((void **)&buf, ALLOC_SIZE));

    /* Initialize buffer on host */
    for (size_t i = 0; i < ALLOC_SIZE / sizeof(int); i++)
        buf[i] = (int)i;

    int rc = dt_start("delta");
    if (rc) {
        fprintf(stderr, "start failed: %s\n", strerror(-rc));
        return 1;
    }
    printf("tracking started\n");

    /* Launch RMW kernel workload while tracking is active. */
    int num_ints = (int)(ALLOC_SIZE / sizeof(int));
    int threads  = 256;
    int blocks   = (num_ints + threads - 1) / threads;
    for (int i = 0; i < NUM_ITERS; i++) {
        rmw_kernel<<<blocks, threads>>>(buf, num_ints);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    printf("rmw kernel done\n");

    rc = dt_cutover();
    if (rc) {
        fprintf(stderr, "cutover failed: %s\n", strerror(-rc));
        (void)dt_stop();
        return 1;
    }

    int recorded = dt_dump_count_pages_in_range((unsigned long)buf, ALLOC_SIZE, PAGE_SIZE);
    if (recorded < 0) {
        fprintf(stderr, "dump failed: %s\n", strerror(-recorded));
        (void)dt_stop();
        return 1;
    }

    rc = dt_stop();
    if (rc) {
        fprintf(stderr, "stop failed: %s\n", strerror(-rc));
        return 1;
    }
    printf("tracking stopped\n");
    printf("recorded pages: %d / %d\n", recorded, NUM_PAGES);

    CUDA_CHECK(cudaFree(buf));
    return recorded == NUM_PAGES ? 0 : 1;
}
