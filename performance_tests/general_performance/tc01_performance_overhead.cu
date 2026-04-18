#include <cuda_runtime.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE 4096

static const int PAGE_COUNTS[] = {8, 64, 256, 512, 1024};
static const int ITER_COUNTS[] = {10, 50, 200};
#define N_PAGE_COUNTS ((int)(sizeof(PAGE_COUNTS) / sizeof(PAGE_COUNTS[0])))
#define N_ITER_COUNTS ((int)(sizeof(ITER_COUNTS) / sizeof(ITER_COUNTS[0])))

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));               \
        exit(1);                                                           \
    }                                                                      \
} while (0)

__global__ void kernel_read(const int *data, int n, volatile int *sink)
{
    int acc = 0;
    for (int i = 0; i < n; i++)
        acc += data[i];
    *sink = acc;
}

__global__ void kernel_write(int *data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = i + 1;
}

__global__ void kernel_mixed(int *data, int n, volatile int *sink)
{
    int acc = 0;
    for (int i = 0; i < n / 2; i++)
        acc += data[i];
    *sink = acc;
    for (int i = n / 2; i < n; i++)
        data[i] = i + 1;
}

typedef enum { WL_READ = 0, WL_WRITE, WL_MIXED } workload_t;
#define N_WORKLOADS 3
static const workload_t WORKLOADS[N_WORKLOADS] = {WL_READ, WL_WRITE, WL_MIXED};

static const char *wl_name(workload_t wl)
{
    switch (wl) {
        case WL_READ:
            return "READ";
        case WL_WRITE:
            return "WRITE";
        case WL_MIXED:
            return "MIXED";
    }
    return "?";
}

static void launch_kernel(workload_t wl, int *managed, int num_ints, int *sink_dev)
{
    switch (wl) {
        case WL_READ:
            kernel_read<<<1, 1>>>(managed, num_ints, (volatile int *)sink_dev);
            break;
        case WL_WRITE:
            kernel_write<<<1, 1>>>(managed, num_ints);
            break;
        case WL_MIXED:
            kernel_mixed<<<1, 1>>>(managed, num_ints, (volatile int *)sink_dev);
            break;
    }
}

static void tracking_on(void)
{
    int rc = dt_start("delta");
    if (rc) {
        fprintf(stderr, "start failed: %s\n", strerror(-rc));
        exit(1);
    }
}

static void tracking_off(void)
{
    int rc = dt_stop();
    if (rc) {
        fprintf(stderr, "stop failed: %s\n", strerror(-rc));
        exit(1);
    }
}

static void reset_table(void)
{
    int rc = dt_stop();
    if (rc) {
        fprintf(stderr, "stop (reset) failed: %s\n", strerror(-rc));
        exit(1);
    }
    rc = dt_start("delta");
    if (rc) {
        fprintf(stderr, "start (reset) failed: %s\n", strerror(-rc));
        exit(1);
    }
}

static double wall_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

typedef struct {
    double kernel_avg_ms;
    double kernel_std_ms;
    double wall_avg_ms;
    double wall_std_ms;
} bench_t;

static bench_t stats(double *k, double *w, int n)
{
    bench_t r = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        r.kernel_avg_ms += k[i];
        r.wall_avg_ms += w[i];
    }
    r.kernel_avg_ms /= n;
    r.wall_avg_ms /= n;
    for (int i = 0; i < n; i++) {
        r.kernel_std_ms += (k[i] - r.kernel_avg_ms) * (k[i] - r.kernel_avg_ms);
        r.wall_std_ms += (w[i] - r.wall_avg_ms) * (w[i] - r.wall_avg_ms);
    }
    r.kernel_std_ms = sqrt(r.kernel_std_ms / n);
    r.wall_std_ms = sqrt(r.wall_std_ms / n);
    return r;
}

static bench_t run_bench(int *managed, int *sink_dev,
                         int num_pages, int num_iters, workload_t wl,
                         int tracking)
{
    int num_ints = num_pages * PAGE_SIZE / (int)sizeof(int);

    double *k_times = (double *)malloc(num_iters * sizeof(double));
    double *w_times = (double *)malloc(num_iters * sizeof(double));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    if (tracking) {
        tracking_on();
    } else {
        launch_kernel(wl, managed, num_ints, sink_dev);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < num_iters; i++) {
        if (tracking)
            reset_table();

        double t0 = wall_ms();
        CUDA_CHECK(cudaEventRecord(ev_start, 0));

        launch_kernel(wl, managed, num_ints, sink_dev);

        CUDA_CHECK(cudaEventRecord(ev_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        double t1 = wall_ms();

        float km;
        CUDA_CHECK(cudaEventElapsedTime(&km, ev_start, ev_stop));
        k_times[i] = km;
        w_times[i] = t1 - t0;
    }

    if (tracking)
        tracking_off();

    bench_t result = stats(k_times, w_times, num_iters);
    free(k_times);
    free(w_times);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return result;
}

int main(void)
{
    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }
    if (!dt_sysfs_exists(DT_PROCFS_START)) {
        fprintf(stderr, "ERROR: %s not found - is the nvidia-uvm module loaded?\n",
                DT_PROCFS_START);
        return 1;
    }

    pid_t my_pid = getpid();
    int max_pages = PAGE_COUNTS[N_PAGE_COUNTS - 1];

    int *managed = NULL;
    int *sink_dev = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)max_pages * PAGE_SIZE));
    CUDA_CHECK(cudaMalloc(&sink_dev, sizeof(int)));
    memset(managed, 0, (size_t)max_pages * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    fprintf(stderr, "[tc02] pid=%d  managed=0x%lx  max_pages=%d\n",
            my_pid, (unsigned long)managed, max_pages);

    int total_rows = N_WORKLOADS * N_PAGE_COUNTS * N_ITER_COUNTS;
    int row = 0;

    typedef struct {
        workload_t wl;
        int pages, iters;
        bench_t off, on;
    } row_t;

    row_t *rows = (row_t *)malloc(total_rows * sizeof(row_t));

    for (int wi = 0; wi < N_WORKLOADS; wi++) {
        workload_t wl = WORKLOADS[wi];
        for (int pi = 0; pi < N_PAGE_COUNTS; pi++) {
            for (int ii = 0; ii < N_ITER_COUNTS; ii++) {
                int pages = PAGE_COUNTS[pi];
                int iters = ITER_COUNTS[ii];

                fprintf(stderr, "[tc02] %-5s  pages=%-4d  iters=%-3d  ...\n",
                        wl_name(wl), pages, iters);

                bench_t off = run_bench(managed, sink_dev, pages, iters, wl, 0);
                bench_t on = run_bench(managed, sink_dev, pages, iters, wl, 1);

                rows[row++] = (row_t){wl, pages, iters, off, on};
            }
        }
    }

    fprintf(stderr, "\n+------------+-------+-------+--------------+--------------+-----------+\n");
    fprintf(stderr, "| %-10s | %5s | %5s | %12s | %12s | %9s |\n",
            "WL", "pages", "iters", "wall_OFF ms", "wall_ON ms", "overhead%");
    fprintf(stderr, "+------------+-------+-------+--------------+--------------+-----------+\n");

    for (int i = 0; i < row; i++) {
        row_t *r = &rows[i];
        double ovhd_pct =
            (r->off.wall_avg_ms > 0.0)
                ? 100.0 * (r->on.wall_avg_ms - r->off.wall_avg_ms) / r->off.wall_avg_ms
                : 0.0;
        fprintf(stderr, "| %-10s | %5d | %5d | %12.4f | %12.4f | %+8.1f%% |\n",
                wl_name(r->wl), r->pages, r->iters,
                r->off.wall_avg_ms, r->on.wall_avg_ms, ovhd_pct);
    }
    fprintf(stderr, "+------------+-------+-------+--------------+--------------+-----------+\n");
    free(rows);
    CUDA_CHECK(cudaFree(managed));
    CUDA_CHECK(cudaFree(sink_dev));
    return 0;
}
