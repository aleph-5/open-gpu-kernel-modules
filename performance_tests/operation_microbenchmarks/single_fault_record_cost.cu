#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE   4096
#define NUM_PAGES   4096
#define ITERATIONS  100
#define ALLOC_SIZE  ((size_t)NUM_PAGES * PAGE_SIZE)

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static int snapshot_and_count_recorded_pages(volatile char *buf, size_t size)
{
    int rc = dt_cutover();
    if (rc) {
        fprintf(stderr, "cutover failed: %s\n", strerror(-rc));
        exit(1);
    }

    int n = dt_dump_count_pages_in_range((unsigned long)buf, size, PAGE_SIZE);
    if (n < 0) {
        fprintf(stderr, "dump failed: %s\n", strerror(-n));
        exit(1);
    }
    return n;
}

/* Single thread writes one byte to the first byte of each page */
__global__ void write_pages(volatile char *buf, int num_pages)
{
    for (int i = 0; i < num_pages; i++)
        buf[(size_t)i * PAGE_SIZE] = (char)i;
}

int main(void)
{
    printf("pid: %d\n", getpid());

    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }
    if (!dt_sysfs_exists(DT_PROCFS_START)) {
        fprintf(stderr, "ERROR: %s not found - is the nvidia-uvm module loaded?\n",
                DT_PROCFS_START);
        return 1;
    }

    volatile char *buf;
    CUDA_CHECK(cudaMallocManaged((void **)&buf, ALLOC_SIZE));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // warmup
    write_pages<<<1, 1>>>(buf, NUM_PAGES);
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
        {
            int rc = dt_start("delta");
            if (rc) {
                fprintf(stderr, "start failed: %s\n", strerror(-rc));
                return 1;
            }
        }

        CUDA_CHECK(cudaEventRecord(ev_start));
        write_pages<<<1, 1>>>(buf, NUM_PAGES);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&times_with_track[iter], ev_start, ev_stop));

        recorded_pages[iter] = snapshot_and_count_recorded_pages(buf, ALLOC_SIZE);
        if (recorded_pages[iter] != NUM_PAGES) {
            fprintf(stderr,
                    "iteration %d: expected %d recorded pages, got %d\n",
                    iter,
                    NUM_PAGES,
                    recorded_pages[iter]);
        }

        {
            int rc = dt_stop();
            if (rc) {
                fprintf(stderr, "stop failed: %s\n", strerror(-rc));
                return 1;
            }
        }

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
        sum_no += times_no_track[i];
        sum_with += times_with_track[i];
    }
    double avg_no = sum_no / ITERATIONS;
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
