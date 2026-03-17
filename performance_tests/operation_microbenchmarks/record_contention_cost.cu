#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>

#define PAGE_SIZE       4096
#define MAX_STREAMS     8
#define MAX_N           4096
#define ALLOC_SIZE      ((size_t)MAX_STREAMS * MAX_N * PAGE_SIZE)
#define ITERATIONS      100
#define FAULT_CHUNK     64

#define PROCFS_START     "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP      "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY_PID "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_DIRTY     "/proc/driver/nvidia-uvm/dirty_pages"
#define PROCFS_RANGE     "/proc/driver/nvidia-uvm/dirty_range"

static const int STREAM_COUNTS[] = {1, 2, 4, 8};
#define NUM_STREAM_COUNTS (int)(sizeof(STREAM_COUNTS) / sizeof(STREAM_COUNTS[0]))

static const int PAGE_COUNTS[] = {64, 256, 1024, 4096};
#define NUM_PAGE_COUNTS (int)(sizeof(PAGE_COUNTS) / sizeof(PAGE_COUNTS[0]))

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static void write_pid_to_procfs(const char *path, pid_t pid)
{
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d", (int)pid);
    int fd = open(path, O_WRONLY);
    if (fd < 0) { 
        perror(path); 
        exit(1); 
    }
    if (write(fd, buf, n) != n) { 
        perror("write procfs"); 
        close(fd); 
        exit(1); 
    }
    close(fd);
}

static void write_range_to_procfs(const char *path, unsigned long start, unsigned long end)
{
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "0x%lx 0x%lx\n", start, end);
    int fd = open(path, O_WRONLY);
    if (fd < 0) { 
        perror(path); 
        exit(1); 
    }
    if (write(fd, buf, n) != n) { 
        perror("write range procfs"); 
        close(fd); 
        exit(1); 
    }
    close(fd);
}

static void start_tracking(pid_t pid) { write_pid_to_procfs(PROCFS_START, pid); }
static void stop_tracking(pid_t pid)  { write_pid_to_procfs(PROCFS_STOP,  pid); }

static int count_recorded_pages(pid_t pid, volatile char *buf, size_t size)
{
    char line[256];
    int count = 0;

    write_pid_to_procfs(PROCFS_QUERY_PID, pid);
    write_range_to_procfs(PROCFS_RANGE, (unsigned long)buf, (unsigned long)buf + size);

    FILE *fp = fopen(PROCFS_DIRTY, "r");
    if (!fp) { 
        perror(PROCFS_DIRTY); 
        exit(1); 
    }

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') {
            if (strstr(line, "not active") || strstr(line, "invalid range")) {
                fprintf(stderr, "dirty_pages query failed: %s", line);
                fclose(fp);
                exit(1);
            }
            continue;
        }
        count++;
    }

    fclose(fp);
    return count;
}

static double elapsed_us(struct timespec *t0, struct timespec *t1)
{
    return (t1->tv_sec - t0->tv_sec) * 1e6 + (t1->tv_nsec - t0->tv_nsec) / 1e3;
}

__global__ void write_pages(volatile char *buf, int n_pages)
{
    int tid = threadIdx.x;
    int n_chunks = (n_pages + FAULT_CHUNK - 1) / FAULT_CHUNK;
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int page = chunk * FAULT_CHUNK + tid;
        if (page < n_pages) buf[(size_t)page * PAGE_SIZE] = (char)page;
        __syncthreads();
    }
}

int main(void)
{
    pid_t pid = getpid();
    printf("pid: %d\n", pid);

    volatile char *buf;
    CUDA_CHECK(cudaMallocManaged((void **)&buf, ALLOC_SIZE));

    cudaStream_t streams[MAX_STREAMS];
    for (int s = 0; s < MAX_STREAMS; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    /* warmup: touch all pages */
    write_pages<<<1, FAULT_CHUNK>>>(buf, MAX_STREAMS * MAX_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    const char *csv_path = "record_contention_cost.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) { perror(csv_path); exit(1); }
    fprintf(csv, "condition,streams,pages_per_stream,iteration,elapsed_us,recorded_pages\n");

    /* Disjoint pages: each stream touches a separate set of pages [s*n, (s+1)*n) */
    printf("----disjoint pages----\n");

    for (int si = 0; si < NUM_STREAM_COUNTS; si++) {
        int S = STREAM_COUNTS[si];
        for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
            int n = PAGE_COUNTS[pi];
            printf("disjoint: streams=%d pages_per_stream=%d ...\n", S, n);

            double sum = 0.0;
            struct timespec t0, t1;

            for (int iter = 0; iter < ITERATIONS; iter++) {
                start_tracking(pid);
                CUDA_CHECK(cudaDeviceSynchronize());

                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int s = 0; s < S; s++)
                    write_pages<<<1, FAULT_CHUNK, 0, streams[s]>>>(buf + (size_t)s * n * PAGE_SIZE, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                clock_gettime(CLOCK_MONOTONIC, &t1);

                double t = elapsed_us(&t0, &t1);

                int recorded = count_recorded_pages(pid, buf, (size_t)S * n * PAGE_SIZE);
                if (recorded != S * n)
                    fprintf(stderr, "disjoint iter %d (S=%d n=%d): expected %d got %d\n", iter, S, n, S * n, recorded);

                stop_tracking(pid);

                fprintf(csv, "disjoint,%d,%d,%d,%.3f,%d\n", S, n, iter, t, recorded);
                sum += t;
            }

            printf("  avg elapsed_us: %.3f\n", sum / ITERATIONS);
        }
    }

    /* Hot set: all streams touch the same set of pages [0, n) */
    printf("----hot set (same pages)----\n");

    for (int si = 0; si < NUM_STREAM_COUNTS; si++) {
        int S = STREAM_COUNTS[si];
        for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
            int n = PAGE_COUNTS[pi];
            printf("hot set: streams=%d pages_per_stream=%d ...\n", S, n);

            double sum = 0.0;
            struct timespec t0, t1;

            for (int iter = 0; iter < ITERATIONS; iter++) {
                start_tracking(pid);
                CUDA_CHECK(cudaDeviceSynchronize());

                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int s = 0; s < S; s++)
                    write_pages<<<1, FAULT_CHUNK, 0, streams[s]>>>(buf, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                clock_gettime(CLOCK_MONOTONIC, &t1);

                double t = elapsed_us(&t0, &t1);

                int recorded = count_recorded_pages(pid, buf, (size_t)n * PAGE_SIZE);
                if (recorded != n)
                    fprintf(stderr, "hot set iter %d (S=%d n=%d): expected %d got %d\n", iter, S, n, n, recorded);

                stop_tracking(pid);

                fprintf(csv, "hot_set,%d,%d,%d,%.3f,%d\n", S, n, iter, t, recorded);
                sum += t;
            }

            printf("  avg elapsed_us: %.3f\n", sum / ITERATIONS);
        }
    }

    fclose(csv);
    printf("per-iteration results written to %s\n", csv_path);

    for (int s = 0; s < MAX_STREAMS; s++)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    CUDA_CHECK(cudaFree((void *)buf));
    return 0;
}
