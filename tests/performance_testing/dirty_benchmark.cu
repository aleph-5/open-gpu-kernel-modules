#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <cuda_runtime.h>

// GPU kernel: write a unique value to every page
__global__ void write_every_page(char *buf, long n_pages, int page_size) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pages) {
        buf[idx * page_size] = (char)(idx & 0xFF);
    }
}

static long ns_elapsed(struct timespec *a, struct timespec *b) {
    return (b->tv_sec - a->tv_sec) * 1000000000L + (b->tv_nsec - a->tv_nsec);
}

static void write_procfs(const char *path, const char *val) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    write(fd, val, strlen(val));
    close(fd);
}

static long count_dirty_pages(void) {
    FILE *f = fopen("/proc/driver/nvidia-uvm/dirty_pages", "r");
    if (!f) { perror("dirty_pages"); return -1; }
    long count = 0;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;   // skip header
        count++;
    }
    fclose(f);
    return count;
}

int main(int argc, char **argv) {
    long n_pages = 10000;   // default, override with argv[1]
    if (argc > 1) n_pages = atol(argv[1]);

    int page_size = 4096;
    size_t buf_size = (size_t)n_pages * page_size;

    printf("=== Dirty Tracking Benchmark: %ld pages (%.1f MB) ===\n",
           n_pages, buf_size / 1e6);

    // --- allocate managed memory ---
    char *buf;
    cudaMallocManaged(&buf, buf_size);

    // --- baseline: GPU kernel WITHOUT dirty tracking ---
    // touch once to establish mappings
    int threads = 256;
    int blocks  = (n_pages + threads - 1) / threads;
    write_every_page<<<blocks, threads>>>(buf, n_pages, page_size);
    cudaDeviceSynchronize();

    struct timespec t0, t1, t2, t3;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    write_every_page<<<blocks, threads>>>(buf, n_pages, page_size);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long baseline_ns = ns_elapsed(&t0, &t1);
    printf("[BASELINE] kernel time: %.3f ms\n", baseline_ns / 1e6);

    // --- start dirty tracking ---
    char pid_str[32];
    snprintf(pid_str, sizeof(pid_str), "%d", getpid());

    // set query pid
    write_procfs("/proc/driver/nvidia-uvm/dirty_pid_to_query", pid_str);

    // set range to cover full buffer
    char range_str[64];
    snprintf(range_str, sizeof(range_str), "%lx %lx",
             (unsigned long)buf,
             (unsigned long)(buf + buf_size));
    write_procfs("/proc/driver/nvidia-uvm/dirty_range", range_str);

    // start tracking — this also triggers GPU PTE invalidation
    clock_gettime(CLOCK_MONOTONIC, &t2);
    write_procfs("/proc/driver/nvidia-uvm/dirty_pids_start_track", pid_str);
    clock_gettime(CLOCK_MONOTONIC, &t3);

    long invalidate_ns = ns_elapsed(&t2, &t3);
    printf("[INVALIDATE] cost: %.3f ms\n", invalidate_ns / 1e6);

    // --- tracked run ---
    struct timespec t4, t5;
    clock_gettime(CLOCK_MONOTONIC, &t4);
    write_every_page<<<blocks, threads>>>(buf, n_pages, page_size);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t5);

    long tracked_ns = ns_elapsed(&t4, &t5);
    printf("[TRACKED]  kernel time: %.3f ms\n", tracked_ns / 1e6);
    printf("[OVERHEAD] %.3f ms (%.1f%%)\n",
           (tracked_ns - baseline_ns) / 1e6,
           100.0 * (tracked_ns - baseline_ns) / baseline_ns);

    // --- read results ---
    long recorded = count_dirty_pages();
    printf("[RESULTS]  dirty pages recorded: %ld / %ld\n", recorded, n_pages);
    printf("[RESULTS]  missing pages:        %ld\n", n_pages - recorded);
    printf("[RESULTS]  recording rate:       %.0f pages/sec\n",
           recorded / (tracked_ns / 1e9));

    // --- stop tracking ---
    write_procfs("/proc/driver/nvidia-uvm/dirty_pids_stop_track", pid_str);

    cudaFree(buf);
    return 0;
}
