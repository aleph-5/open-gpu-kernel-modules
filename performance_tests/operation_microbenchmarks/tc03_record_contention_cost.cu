#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE       4096
#define MAX_STREAMS     8
#define MAX_N           4096
#define ALLOC_SIZE      ((size_t)MAX_STREAMS * MAX_N * PAGE_SIZE)
#define ITERATIONS      100
#define FAULT_CHUNK     64

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
    CUDA_CHECK(cudaMallocManaged((void **)&buf, ALLOC_SIZE));

    cudaStream_t streams[MAX_STREAMS];
    for (int s = 0; s < MAX_STREAMS; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    /* warmup: touch all pages */
    write_pages<<<1, FAULT_CHUNK>>>(buf, MAX_STREAMS * MAX_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    const char *csv_path = "tc03_record_contention_cost.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror(csv_path);
        exit(1);
    }
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
                int rc = dt_start("delta");
                if (rc) {
                    fprintf(stderr, "start failed: %s\n", strerror(-rc));
                    return 1;
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int s = 0; s < S; s++)
                    write_pages<<<1, FAULT_CHUNK, 0, streams[s]>>>(
                        buf + (size_t)s * n * PAGE_SIZE, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                clock_gettime(CLOCK_MONOTONIC, &t1);

                double t = elapsed_us(&t0, &t1);

                int recorded = snapshot_and_count_recorded_pages(buf, (size_t)S * n * PAGE_SIZE);
                if (recorded != S * n)
                    fprintf(stderr,
                            "disjoint iter %d (S=%d n=%d): expected %d got %d\n",
                            iter, S, n, S * n, recorded);

                rc = dt_stop();
                if (rc) {
                    fprintf(stderr, "stop failed: %s\n", strerror(-rc));
                    return 1;
                }

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
                int rc = dt_start("delta");
                if (rc) {
                    fprintf(stderr, "start failed: %s\n", strerror(-rc));
                    return 1;
                }
                CUDA_CHECK(cudaDeviceSynchronize());

                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int s = 0; s < S; s++)
                    write_pages<<<1, FAULT_CHUNK, 0, streams[s]>>>(buf, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                clock_gettime(CLOCK_MONOTONIC, &t1);

                double t = elapsed_us(&t0, &t1);

                int recorded = snapshot_and_count_recorded_pages(buf, (size_t)n * PAGE_SIZE);
                if (recorded != n)
                    fprintf(stderr,
                            "hot set iter %d (S=%d n=%d): expected %d got %d\n",
                            iter, S, n, n, recorded);

                rc = dt_stop();
                if (rc) {
                    fprintf(stderr, "stop failed: %s\n", strerror(-rc));
                    return 1;
                }

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
