#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define PROCFS_START  "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP   "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY  "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_PAGES  "/proc/driver/nvidia-uvm/dirty_pages"
#define PROCFS_RANGE  "/proc/driver/nvidia-uvm/dirty_range"

#define PAGE_SIZE_BYTES 4096

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
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

static void procfs_write(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    if (write(fd, val, strlen(val)) < 0) { perror(path); exit(1); }
    close(fd);
}

static long count_dirty_pages(void)
{
    FILE *f = fopen(PROCFS_PAGES, "r");
    if (!f) { perror(PROCFS_PAGES); return -1; }
    long count = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) { fclose(f); return -2; }
            continue;
        }
        count++;
    }
    fclose(f);
    return count;
}

int main(int argc, char **argv)
{
    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    long n_pages = 1000;
    if (argc > 1) n_pages = atol(argv[1]);

    size_t buf_size = (size_t)n_pages * PAGE_SIZE_BYTES;
    printf("[tc01] %ld pages (%.1f MB)\n", n_pages, buf_size / 1e6);

    char pid_str[32];
    snprintf(pid_str, sizeof(pid_str), "%d\n", getpid());

    char *buf;
    CUDA_CHECK(cudaMallocManaged(&buf, buf_size));
    memset(buf, 0, buf_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads = 256;
    int blocks  = (n_pages + threads - 1) / threads;

    procfs_write(PROCFS_QUERY, pid_str);
    procfs_write(PROCFS_RANGE, "0x0 0xffffffffffffffff\n");

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

    // tracked: start_track invalidates PTEs, next access generates faults
    struct timespec ti0, ti1;
    clock_gettime(CLOCK_MONOTONIC, &ti0);
    procfs_write(PROCFS_START, pid_str);
    clock_gettime(CLOCK_MONOTONIC, &ti1);
    printf("[tc01] invalidate: %.3f ms\n", ns_elapsed(&ti0, &ti1) / 1e6);

    struct timespec t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t2);
    kernel_write<<<blocks, threads>>>(buf, n_pages);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t3);
    long tracked_ns = ns_elapsed(&t2, &t3);

    long recorded = count_dirty_pages();
    procfs_write(PROCFS_STOP, pid_str);

    printf("[tc01] tracked   : %.3f ms\n", tracked_ns / 1e6);
    printf("[tc01] overhead  : %.3f ms  (%.1f%%)\n",
           (tracked_ns - baseline_ns) / 1e6,
           100.0 * (tracked_ns - baseline_ns) / baseline_ns);
    printf("[tc01] recorded  : %ld / %ld pages\n", recorded, n_pages);
    printf("[tc01] missing   : %ld pages\n", n_pages - recorded);
    if (tracked_ns > 0 && recorded > 0)
        printf("[tc01] rate      : %.0f pages/sec\n",
               recorded / (tracked_ns / 1e9));

    CUDA_CHECK(cudaFree(buf));
    return (recorded == n_pages) ? 0 : 1;
}

