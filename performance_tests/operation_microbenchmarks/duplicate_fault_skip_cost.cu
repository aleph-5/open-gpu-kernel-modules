#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define PAGE_SIZE   4096
#define MAX_PAGES   16384
#define ITERATIONS  100

#define PROCFS_START     "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP      "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_DIRTY_PAGES "/proc/driver/nvidia-uvm/dirty_pages"
#define PROCFS_RANGE     "/proc/driver/nvidia-uvm/dirty_range"

static const int PAGE_COUNTS[] = {64, 256, 1024, 4096, 16384};
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

static void write_str_to_procfs(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    if (write(fd, val, strlen(val)) < 0) { perror("write"); exit(1); }
    close(fd);
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

static void start_tracking(void) { write_str_to_procfs(PROCFS_START, "start\n"); }
static void stop_tracking(void) { write_str_to_procfs(PROCFS_STOP, "stop\n"); }

static int count_recorded_pages(volatile char *buf, size_t size)
{
    char line[256];
    int count = 0;
    write_range_to_procfs(PROCFS_RANGE,
                          (unsigned long)buf,
                          (unsigned long)buf + size);

    FILE *fp = fopen(PROCFS_DIRTY_PAGES, "r");
    if (!fp) { perror(PROCFS_DIRTY_PAGES); exit(1); }

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

#define FAULT_CHUNK 64

__global__ void write_pages(volatile char *buf, int n_pages)
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

int main(void)
{
    pid_t pid = getpid();
    printf("pid: %d\n", pid);

    volatile char *buf;
    CUDA_CHECK(cudaMallocManaged((void **)&buf, (size_t)MAX_PAGES * PAGE_SIZE));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* warmup */
    write_pages<<<1, FAULT_CHUNK>>>(buf, MAX_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    const char *csv_path = "duplicate_fault_skip_cost.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) { perror(csv_path); exit(1); }
    fprintf(csv, "page_count,iteration,first_fault_ms,duplicate_fault_ms,first_recorded,duplicate_recorded\n");

    for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
        int n = PAGE_COUNTS[pi];
        printf("page_count=%d ...\n", n);

        double sum_first = 0.0, sum_dup = 0.0;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            start_tracking();
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(ev_start));
            write_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float first_ms;
            CUDA_CHECK(cudaEventElapsedTime(&first_ms, ev_start, ev_stop));

            int first_recorded = count_recorded_pages(buf, (size_t)n * PAGE_SIZE);
            if (first_recorded != n)
                fprintf(stderr, "iter %d (n=%d): expected %d after first fault, got %d\n", iter, n, n, first_recorded);

            /* CPU write forces UVM to migrate pages back to CPU */
            memset((void *)buf, 0, (size_t)n * PAGE_SIZE);
            

            /* duplicate fault: same N pages fault again, already in table */
            CUDA_CHECK(cudaEventRecord(ev_start));
            write_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float dup_ms;
            CUDA_CHECK(cudaEventElapsedTime(&dup_ms, ev_start, ev_stop));

            int dup_recorded = count_recorded_pages(buf, (size_t)n * PAGE_SIZE);
            if (dup_recorded != first_recorded)
                fprintf(stderr,
                        "iter %d (n=%d): recorded count changed after duplicate fault: %d -> %d\n",
                        iter, n, first_recorded, dup_recorded);

            stop_tracking();

            fprintf(csv, "%d,%d,%.6f,%.6f,%d,%d\n",
                    n, iter, first_ms, dup_ms, first_recorded, dup_recorded);
            sum_first += first_ms;
            sum_dup   += dup_ms;
        }

        printf("  avg first_fault_ms:     %.6f\n", sum_first / ITERATIONS);
        printf("  avg duplicate_fault_ms: %.6f\n", sum_dup   / ITERATIONS);
    }

    fclose(csv);
    printf("per-iteration results written to %s\n", csv_path);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree((void *)buf));
    return 0;
}
