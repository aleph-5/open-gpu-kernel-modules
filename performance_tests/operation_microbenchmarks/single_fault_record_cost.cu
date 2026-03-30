#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define PAGE_SIZE   4096
#define NUM_PAGES   4096
#define ITERATIONS  100
#define ALLOC_SIZE  ((size_t)NUM_PAGES * PAGE_SIZE)

#define PROCFS_START "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP  "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_DIRTY_PAGES "/proc/driver/nvidia-uvm/dirty_pages"
#define PROCFS_RANGE "/proc/driver/nvidia-uvm/dirty_range"

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
    FILE *fp;
    write_range_to_procfs(PROCFS_RANGE,
                          (unsigned long)buf,
                          (unsigned long)buf + size);

    fp = fopen(PROCFS_DIRTY_PAGES, "r");
    if (!fp) {
        perror(PROCFS_DIRTY_PAGES);
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

/* Single thread writes one byte to the first byte of each page */
__global__ void write_pages(volatile char *buf, int num_pages)
{
    for (int i = 0; i < num_pages; i++)
        buf[(size_t)i * PAGE_SIZE] = (char)i;
}

int main(void)
{
    pid_t pid = getpid();
    printf("The pid is : %d\n", pid); 
    volatile char *buf;

    CUDA_CHECK(cudaMallocManaged((void **)&buf, ALLOC_SIZE));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // warmup
    write_pages<<<1,1>>>(buf, NUM_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    float times_no_track[ITERATIONS];
    float times_with_track[ITERATIONS];
    int recorded_pages[ITERATIONS];

    const char *csv_path = "single_fault_record_cost.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror(csv_path);
        exit(1);
    }
    fprintf(csv, "run,no_tracking_ms,with_tracking_ms,recorded_pages\n");

    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset((void *)buf, 0, ALLOC_SIZE);
        CUDA_CHECK(cudaEventRecord(ev_start));
        write_pages<<<1, 1>>>(buf, NUM_PAGES);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&times_no_track[iter], ev_start, ev_stop));

        memset((void *)buf, 0, ALLOC_SIZE);
        start_tracking();

        CUDA_CHECK(cudaEventRecord(ev_start));
        write_pages<<<1, 1>>>(buf, NUM_PAGES);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&times_with_track[iter], ev_start, ev_stop));

        recorded_pages[iter] = count_recorded_pages(buf, ALLOC_SIZE);
        if (recorded_pages[iter] != NUM_PAGES) {
            fprintf(stderr,
                    "iteration %d: expected %d recorded pages, got %d\n",
                    iter,
                    NUM_PAGES,
                    recorded_pages[iter]);
        }
        
        stop_tracking();

        fprintf(csv,
                "%d,%.6f,%.6f,%d\n",
                iter,
                times_no_track[iter],
                times_with_track[iter],
                recorded_pages[iter]);
    }
    fclose(csv);
    printf("per-iteration results written to %s\n", csv_path);

    double sum_no = 0.0, sum_with = 0.0;
    for (int i = 0; i < ITERATIONS; i++) {
        sum_no   += times_no_track[i];
        sum_with += times_with_track[i];
    }
    double avg_no   = sum_no   / ITERATIONS;
    double avg_with = sum_with / ITERATIONS;

    printf("average_no_tracking_ms,  %.6f\n", avg_no);
    printf("average_with_tracking_ms,%.6f\n", avg_with);
    printf("overhead_ms,             %.6f\n", avg_with - avg_no);
    printf("overhead_per_fault_us,   %.6f\n", (avg_with - avg_no) * 1000.0 / NUM_PAGES);

    for (int i = 0; i < ITERATIONS; i++) {
        if (recorded_pages[i] != NUM_PAGES) {
            printf("warning: iteration %d recorded %d/%d pages\n",
                   i,
                   recorded_pages[i],
                   NUM_PAGES);
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree((void *)buf));
    return 0;
}
