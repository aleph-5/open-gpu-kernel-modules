#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>

#define PAGE_SIZE    4096
#define MAX_PAGES    16384
#define ALLOC_SIZE   (2 * (size_t)MAX_PAGES * PAGE_SIZE)
#define ITERATIONS   100

static const int PAGE_COUNTS[] = {64, 256, 1024, 4096, 16384};
#define NUM_PAGE_COUNTS (int)(sizeof(PAGE_COUNTS) / sizeof(PAGE_COUNTS[0]))

#define PROCFS_START     "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP      "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY_PID "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_RANGE     "/proc/driver/nvidia-uvm/dirty_range"
#define PROCFS_DIRTY     "/proc/driver/nvidia-uvm/dirty_pages"

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

/* time a procfs write in microseconds using CLOCK_MONOTONIC */
static double timed_write_pid(const char *path, pid_t pid)
{
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%d", (int)pid);
    int fd = open(path, O_WRONLY);
    if (fd < 0) { 
        perror(path); 
        exit(1); 
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (write(fd, buf, n) != n) { 
        perror("write procfs timed"); 
        close(fd); 
        exit(1); 
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    close(fd);

    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
}

static void write_range_to_procfs(const char *path, unsigned long start, unsigned long end)
{
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "0x%lx 0x%lx\n", start, end);
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    if (write(fd, buf, n) != n) { perror("write range procfs"); close(fd); exit(1); }
    close(fd);
}

static int count_recorded_pages(pid_t pid, volatile char *buf, size_t size)
{
    char line[256];
    int count = 0;

    write_pid_to_procfs(PROCFS_QUERY_PID, pid);
    write_range_to_procfs(PROCFS_RANGE,
                          (unsigned long)buf,
                          (unsigned long)buf + size);

    FILE *fp = fopen(PROCFS_DIRTY, "r");
    if (!fp) { perror(PROCFS_DIRTY); exit(1); }

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

/*
 * Touch/read pages FAULT_CHUNK at a time with __syncthreads() between chunks.
 * Limits simultaneous GPU faults to FAULT_CHUNK, preventing UVM's prefetch
 * heuristic from mapping pages outside the fault recording path.
 * Launch as <<<1, FAULT_CHUNK>>>.
 */
#define FAULT_CHUNK 64

__global__ void touch_pages(volatile char *buf, int n_pages)
{
    int tid = threadIdx.x;
    int n_chunks = (n_pages + FAULT_CHUNK - 1) / FAULT_CHUNK;
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int page = chunk * FAULT_CHUNK + tid;
        if (page < n_pages)
            buf[(size_t)page * PAGE_SIZE] = (char)page;
        __syncthreads();
    }
}

__global__ void read_pages(volatile char *buf, int n_pages)
{
    int tid = threadIdx.x;
    int n_chunks = (n_pages + FAULT_CHUNK - 1) / FAULT_CHUNK;
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        int page = chunk * FAULT_CHUNK + tid;
        volatile char sink;
        if (page < n_pages)
            sink = buf[(size_t)page * PAGE_SIZE];
        __syncthreads();
    }
}

int main(void)
{
    pid_t pid = getpid();
    printf("pid: %d\n", pid);

    volatile char *buf;
    CUDA_CHECK(cudaMallocManaged((void **)&buf, ALLOC_SIZE));

    /* warmup: touch all pages once to avoid first-time CUDA overhead */
    touch_pages<<<1, FAULT_CHUNK>>>(buf, MAX_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    const char *csv_path = "table_init_cost.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) { perror(csv_path); exit(1); }
    fprintf(csv, "page_count,iteration,init_time_us\n");


    /* Measure init cost for different page counts */
    printf("----measuring init cost for page counts in {64, 256, 1024, 4096, 16384}----\n");
    for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
        int n = PAGE_COUNTS[pi];
        printf("measuring init cost: page_count=%d ...\n", n);

        double sum = 0.0;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            touch_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            /* time start_track: inits empty xarray + invalidates N GPU PTEs */
            double t = timed_write_pid(PROCFS_START, pid);

            write_pid_to_procfs(PROCFS_STOP, pid);

            fprintf(csv, "%d,%d,%.3f\n", n, iter, t);
            sum += t;
        }

        printf("  avg init_time_us: %.3f\n", sum / ITERATIONS);
    }


    /* Measure destroy cost for different page counts */
    printf("----measuring destroy cost for page counts in {64, 256, 1024, 4096, 16384}----\n");

    const char *csv_path2 = "table_destroy_cost.csv";
    FILE *csv2 = fopen(csv_path2, "w");
    if (!csv2) { perror(csv_path2); exit(1); }
    fprintf(csv2, "page_count,iteration,recorded_pages,destroy_time_us\n");

    for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
        int n = PAGE_COUNTS[pi];
        printf("measuring destroy cost: page_count=%d ...\n", n);

        double sum = 0.0;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            write_pid_to_procfs(PROCFS_START, pid);

            CUDA_CHECK(cudaDeviceSynchronize());
            touch_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            int recorded = count_recorded_pages(pid, buf, (size_t)n * PAGE_SIZE);
            if (recorded != n) {
                fprintf(stderr,
                        "iter %d: expected %d recorded pages, got %d — skipping\n",
                        iter, n, recorded);
                write_pid_to_procfs(PROCFS_STOP, pid);
                continue;
            }

            /* time stop_track: walks and frees N xarray entries */
            double t = timed_write_pid(PROCFS_STOP, pid);

            fprintf(csv2, "%d,%d,%d,%.3f\n", n, iter, recorded, t);
            sum += t;
        }

        printf("  avg destroy_time_us: %.3f\n", sum / ITERATIONS);
    }


    /*Measure reinit cost */
    volatile char *write_buf = buf;
    volatile char *read_buf  = buf + (size_t)MAX_PAGES * PAGE_SIZE;

    read_pages<<<1, FAULT_CHUNK>>>(read_buf, MAX_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("----measuring reinit cost (dirty_pages x read_pages)----\n");

    const char *csv_path3 = "table_reinit_cost.csv";
    FILE *csv3 = fopen(csv_path3, "w");
    if (!csv3) { perror(csv_path3); exit(1); }
    fprintf(csv3, "dirty_pages,read_pages,iteration,recorded_pages,reinit_time_us\n");

    for (int mi = 0; mi < NUM_PAGE_COUNTS; mi++) {
        int m = PAGE_COUNTS[mi];
        for (int ni = 0; ni < NUM_PAGE_COUNTS; ni++) {
            int n = PAGE_COUNTS[ni];
            printf("measuring reinit cost: dirty_pages=%d read_pages=%d ...\n", n, m);

            double sum = 0.0;

            for (int iter = 0; iter < ITERATIONS; iter++) {
                write_pid_to_procfs(PROCFS_START, pid);

                CUDA_CHECK(cudaDeviceSynchronize());
                touch_pages<<<1, FAULT_CHUNK>>>(write_buf, n);
                read_pages<<<1, FAULT_CHUNK>>>(read_buf, m);
                CUDA_CHECK(cudaDeviceSynchronize());

                int recorded = count_recorded_pages(pid, write_buf,
                                                    (size_t)n * PAGE_SIZE);
                if (recorded != n) {
                    fprintf(stderr,
                            "reinit iter %d (n=%d m=%d): expected %d got %d — skipping\n",
                            iter, n, m, n, recorded);
                    write_pid_to_procfs(PROCFS_STOP, pid);
                    continue;
                }

                /* time reinit: destroys N xarray entries + invalidates
                 * N write PTEs (write_buf) + M read PTEs (read_buf) */
                double t = timed_write_pid(PROCFS_START, pid);

                fprintf(csv3, "%d,%d,%d,%d,%.3f\n", n, m, iter, recorded, t);
                sum += t;
            }

            printf("  avg reinit_time_us: %.3f\n", sum / ITERATIONS);
        }
    }

    fclose(csv2);
    printf("per-iteration results written to %s\n", csv_path2);

    fclose(csv);
    printf("per-iteration results written to %s\n", csv_path);

    fclose(csv3);
    printf("per-iteration results written to %s\n", csv_path3);

    CUDA_CHECK(cudaFree((void *)buf));
    return 0;
}
