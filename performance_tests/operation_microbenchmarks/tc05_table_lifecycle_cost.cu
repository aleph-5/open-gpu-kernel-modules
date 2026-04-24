#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE    4096
#define MAX_PAGES    16384
#define ALLOC_SIZE   (2 * (size_t)MAX_PAGES * PAGE_SIZE)
#define ITERATIONS   100

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

/* time dt_start("delta") in microseconds using CLOCK_MONOTONIC */
static double timed_start_delta(void)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = dt_start("delta");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc) {
        fprintf(stderr, "start failed: %s\n", strerror(-rc));
        exit(1);
    }
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
}

/* time dt_stop() in microseconds using CLOCK_MONOTONIC */
static double timed_stop(void)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = dt_stop();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc) {
        fprintf(stderr, "stop failed: %s\n", strerror(-rc));
        exit(1);
    }
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
}

/* time dt_stop()+dt_start("delta") in microseconds */
static double timed_reinit_delta(void)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int rc = dt_stop();
    if (rc) {
        fprintf(stderr, "stop failed: %s\n", strerror(-rc));
        exit(1);
    }

    rc = dt_start("delta");
    if (rc) {
        fprintf(stderr, "start failed: %s\n", strerror(-rc));
        exit(1);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
}

/*
 * Touch/read pages FAULT_CHUNK at a time with __syncthreads() between chunks.
 * Limits simultaneous GPU faults to FAULT_CHUNK.
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

    /* warmup: touch all pages once to avoid first-time CUDA overhead */
    touch_pages<<<1, FAULT_CHUNK>>>(buf, MAX_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ------------------------------------------------------------------
     * Init cost
     * ------------------------------------------------------------------ */
    const char *csv_path = "tc05_table_init_cost.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror(csv_path);
        exit(1);
    }
    fprintf(csv, "page_count,iteration,init_time_us\n");

    printf("----measuring init cost for page counts in {64, 256, 1024, 4096, 16384}----\n");
    for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
        int n = PAGE_COUNTS[pi];
        printf("measuring init cost: page_count=%d ...\n", n);

        double sum = 0.0;
        for (int iter = 0; iter < ITERATIONS; iter++) {
            touch_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            double t = timed_start_delta();

            int rc = dt_stop();
            if (rc) {
                fprintf(stderr, "stop failed: %s\n", strerror(-rc));
                return 1;
            }

            fprintf(csv, "%d,%d,%.3f\n", n, iter, t);
            sum += t;
        }

        printf("  avg init_time_us: %.3f\n", sum / ITERATIONS);
    }

    fclose(csv);
    printf("per-iteration results written to %s\n", csv_path);

    /* ------------------------------------------------------------------
     * Destroy cost
     * ------------------------------------------------------------------ */
    printf("----measuring destroy cost for page counts in {64, 256, 1024, 4096, 16384}----\n");

    const char *csv_path2 = "tc05_table_destroy_cost.csv";
    FILE *csv2 = fopen(csv_path2, "w");
    if (!csv2) {
        perror(csv_path2);
        exit(1);
    }
    fprintf(csv2, "page_count,iteration,recorded_pages,destroy_time_us\n");

    for (int pi = 0; pi < NUM_PAGE_COUNTS; pi++) {
        int n = PAGE_COUNTS[pi];
        printf("measuring destroy cost: page_count=%d ...\n", n);

        double sum = 0.0;
        for (int iter = 0; iter < ITERATIONS; iter++) {
            int rc = dt_start("delta");
            if (rc) {
                fprintf(stderr, "start failed: %s\n", strerror(-rc));
                return 1;
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            touch_pages<<<1, FAULT_CHUNK>>>(buf, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            /* time stop() which destroys the dirty table */
            double t = timed_stop();

            fprintf(csv2, "%d,%d,%d,%.3f\n", n, iter, n, t);
            sum += t;
        }

        printf("  avg destroy_time_us: %.3f\n", sum / ITERATIONS);
    }

    fclose(csv2);
    printf("per-iteration results written to %s\n", csv_path2);

    /* ------------------------------------------------------------------
     * Reinit cost (stop+start)
     * ------------------------------------------------------------------ */
    volatile char *write_buf = buf;
    volatile char *read_buf = buf + (size_t)MAX_PAGES * PAGE_SIZE;

    read_pages<<<1, FAULT_CHUNK>>>(read_buf, MAX_PAGES);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("----measuring reinit cost (dirty_pages x read_pages)----\n");

    const char *csv_path3 = "tc05_table_reinit_cost.csv";
    FILE *csv3 = fopen(csv_path3, "w");
    if (!csv3) {
        perror(csv_path3);
        exit(1);
    }
    fprintf(csv3, "dirty_pages,read_pages,iteration,recorded_pages,reinit_time_us\n");

    for (int mi = 0; mi < NUM_PAGE_COUNTS; mi++) {
        int m = PAGE_COUNTS[mi];
        for (int ni = 0; ni < NUM_PAGE_COUNTS; ni++) {
            int n = PAGE_COUNTS[ni];
            printf("measuring reinit cost: dirty_pages=%d read_pages=%d ...\n", n, m);

            int rc = dt_start("delta");
            if (rc) {
                fprintf(stderr, "start failed: %s\n", strerror(-rc));
                return 1;
            }

            double sum = 0.0;
            for (int iter = 0; iter < ITERATIONS; iter++) {
                CUDA_CHECK(cudaDeviceSynchronize());
                touch_pages<<<1, FAULT_CHUNK>>>(write_buf, n);
                read_pages<<<1, FAULT_CHUNK>>>(read_buf, m);
                CUDA_CHECK(cudaDeviceSynchronize());

                /* time reinit: stop + start path (includes barrier/drain/downgrade) */
                double t = timed_reinit_delta();

                fprintf(csv3, "%d,%d,%d,%d,%.3f\n", n, m, iter, n, t);
                sum += t;
            }

            /* cleanup: timed_reinit_delta leaves tracking started */
            rc = dt_stop();
            if (rc) {
                fprintf(stderr, "stop failed: %s\n", strerror(-rc));
                return 1;
            }

            printf("  avg reinit_time_us: %.3f\n", sum / ITERATIONS);
        }
    }

    fclose(csv3);
    printf("per-iteration results written to %s\n", csv_path3);

    CUDA_CHECK(cudaFree((void *)buf));
    return 0;
}
