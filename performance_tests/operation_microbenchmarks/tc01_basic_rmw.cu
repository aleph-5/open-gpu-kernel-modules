#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define PAGE_SIZE  4096
#define NUM_PAGES  1024
#define ALLOC_SIZE ((size_t)NUM_PAGES * PAGE_SIZE)
#define NUM_ITERS (100)

#define PROCFS_START "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP  "/proc/driver/nvidia-uvm/dirty_tracking_stop"

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static void write_str_to_procfs(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    if (write(fd, val, strlen(val)) < 0) { perror("write"); exit(1); }
    close(fd);
}


/* Read-modify-write: each thread reads a value, adds its index, writes it back */
__global__ void rmw_kernel(int *buf, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        buf[idx] = buf[idx] + idx;
}

int main(void)
{
    pid_t pid = getpid();
    printf("pid: %d\n", pid);

    int *buf;
    CUDA_CHECK(cudaMallocManaged((void **)&buf, ALLOC_SIZE));

    /* Initialize buffer on host */
    for (size_t i = 0; i < ALLOC_SIZE / sizeof(int); i++)
        buf[i] = (int)i;

    /* Start dirty tracking */
    // write_str_to_procfs(PROCFS_START, "start\n");
    // printf("tracking started\n");

    /* Launch RMW kernel */
    int num_ints = (int)(ALLOC_SIZE / sizeof(int));
    int threads  = 256;
    int blocks   = (num_ints + threads - 1) / threads;
    for (int i = 0; i < NUM_ITERS; i++){
        rmw_kernel<<<blocks, threads>>>(buf, num_ints);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // printf("rmw kernel done\n");

    /* Stop dirty tracking */
    // write_str_to_procfs(PROCFS_STOP, "stop\n");
    // printf("tracking stopped\n");

    CUDA_CHECK(cudaFree(buf));
    return 0;
}
