#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE   4096
#define MAX_PAGES   16384
#define ITERATIONS  100

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
    CUDA_CHECK(cudaMallocManaged((void **)&buf, (size_t)MAX_PAGES * PAGE_SIZE));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* warmup */
    write_pages<<<1, FAULT_CHUNK>>>(buf, MAX_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    const char *csv_path = "tc02_duplicate_fault_skip_cost.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror(csv_path);
        exit(1);
    }
    fprintf(csv,
            "page_count,iteration,first_fault_ms,duplicate_fault_ms,first_recorded,duplicate_recorded\n");

    for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
        int n = PAGE_COUNTS[pi];
        printf("page_count=%d ...\n", n);

        double sum_first = 0.0, sum_dup = 0.0;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            int rc = dt_start("delta");
            if (rc) {
                fprintf(stderr, "start failed: %s\n", strerror(-rc));
                return 1;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(ev_start));
            write_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float first_ms;
            CUDA_CHECK(cudaEventElapsedTime(&first_ms, ev_start, ev_stop));

            /* CPU write forces UVM to migrate pages back to CPU */
            memset((void *)buf, 0, (size_t)n * PAGE_SIZE);

            /* duplicate fault: same N pages fault again, already in table */
            CUDA_CHECK(cudaEventRecord(ev_start));
            write_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float dup_ms;
            CUDA_CHECK(cudaEventElapsedTime(&dup_ms, ev_start, ev_stop));

            int recorded = snapshot_and_count_recorded_pages(buf, (size_t)n * PAGE_SIZE);
            if (recorded != n)
                fprintf(stderr,
                        "iter %d (n=%d): expected %d recorded pages, got %d\n",
                        iter, n, n, recorded);

            rc = dt_stop();
            if (rc) {
                fprintf(stderr, "stop failed: %s\n", strerror(-rc));
                return 1;
            }

            fprintf(csv, "%d,%d,%.6f,%.6f,%d,%d\n",
                    n, iter, first_ms, dup_ms, recorded, recorded);
            sum_first += first_ms;
            sum_dup += dup_ms;
        }

        printf("  avg first_fault_ms:     %.6f\n", sum_first / ITERATIONS);
        printf("  avg duplicate_fault_ms: %.6f\n", sum_dup / ITERATIONS);
    }

    fclose(csv);
    printf("per-iteration results written to %s\n", csv_path);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree((void *)buf));
    return 0;
}
