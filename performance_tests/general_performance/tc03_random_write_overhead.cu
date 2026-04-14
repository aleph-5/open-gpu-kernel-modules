/*
 * tc03_random_write_overhead.cu
 *
 * Like tc02 but writes touch pages in a random permutation order instead of
 * sequential order.  In tc02, sequential writes always insert at the tail of
 * the vector backend's sorted array, so memmove is never called and the O(n)
 * shift cost is invisible.  Here, random page ordering forces insertions at
 * arbitrary positions, exercising the full memmove path on every insert and
 * giving a true measure of the vector backend's insertion overhead.
 *
 * Each iteration generates a fresh Fisher-Yates shuffle so the ordering varies
 * across samples, avoiding any warm-path bias.
 *
 * Output: CSV to stdout  (same schema as tc02, workload is always RAND_WRITE)
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../common/dirty_tracking_procfs.h"

#define PAGE_SIZE  4096
#define PAGE_INTS  ((int)(PAGE_SIZE / sizeof(int)))

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

/*
 * Writes one int to the first word of each page, visiting pages in the order
 * given by page_order[].  This makes page faults arrive in a random order,
 * so vector-backend inserts land at arbitrary positions in the sorted array.
 */
__global__ void kernel_write_random(int *data, const int *page_order, int num_pages)
{
    for (int i = 0; i < num_pages; i++) {
        int p = page_order[i];
        data[(long)p * PAGE_INTS] = p + 1;
    }
}

/* ---------- helpers ---------------------------------------------------- */

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

/*
 * In-place Fisher-Yates shuffle of arr[0..n-1].
 * Called once per benchmark iteration so each sample sees a different order.
 */
static void shuffle(int *arr, int n)
{
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* ---------- stats ------------------------------------------------------- */

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

/* ---------- benchmark -------------------------------------------------- */

/*
 * page_order_host: host-side scratch buffer for the shuffle (length >= num_pages)
 * page_order_dev:  device allocation for the shuffled order  (length >= num_pages)
 */
static bench_t run_bench(int *managed,
                         int *page_order_host,
                         int *page_order_dev,
                         int num_pages, int num_iters,
                         int tracking)
{
    double *k_times = (double *)malloc(num_iters * sizeof(double));
    double *w_times = (double *)malloc(num_iters * sizeof(double));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* Initialize the identity permutation once; shuffle() mutates it. */
    for (int i = 0; i < num_pages; i++)
        page_order_host[i] = i;

    if (tracking) {
        tracking_on();
    } else {
        /* Warmup: bring pages to the GPU so baseline measures pure exec time. */
        shuffle(page_order_host, num_pages);
        CUDA_CHECK(cudaMemcpy(page_order_dev, page_order_host,
                              num_pages * sizeof(int), cudaMemcpyHostToDevice));
        kernel_write_random<<<1, 1>>>(managed, page_order_dev, num_pages);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < num_iters; i++) {
        /* Generate a new random page order for this iteration. */
        shuffle(page_order_host, num_pages);
        CUDA_CHECK(cudaMemcpy(page_order_dev, page_order_host,
                              num_pages * sizeof(int), cudaMemcpyHostToDevice));

        if (tracking)
            reset_table();

        double t0 = wall_ms();
        CUDA_CHECK(cudaEventRecord(ev_start, 0));

        kernel_write_random<<<1, 1>>>(managed, page_order_dev, num_pages);

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

/* ---------- main ------------------------------------------------------- */

int main(void)
{
    if (geteuid() != 0) {
        fprintf(stderr, "ERROR: must run as root\n");
        return 1;
    }
    if (!dt_sysfs_exists(DT_PROCFS_START)) {
        fprintf(stderr,
                "ERROR: %s not found - is the nvidia-uvm module loaded?\n",
                DT_PROCFS_START);
        return 1;
    }

    srand((unsigned int)time(NULL));

    pid_t my_pid = getpid();
    int max_pages = PAGE_COUNTS[N_PAGE_COUNTS - 1];

    int *managed = NULL;
    int *page_order_dev = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, (size_t)max_pages * PAGE_SIZE));
    CUDA_CHECK(cudaMalloc(&page_order_dev, max_pages * (int)sizeof(int)));
    memset(managed, 0, (size_t)max_pages * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Host scratch for the shuffle; sized to max_pages. */
    int *page_order_host = (int *)malloc(max_pages * sizeof(int));
    if (!page_order_host) {
        perror("malloc");
        return 1;
    }

    fprintf(stderr,
            "[tc03] pid=%d  managed=0x%lx  max_pages=%d  (random write order)\n",
            my_pid, (unsigned long)managed, max_pages);

    int total_rows = N_PAGE_COUNTS * N_ITER_COUNTS;
    typedef struct {
        int pages, iters;
        bench_t off, on;
    } row_t;

    row_t *rows = (row_t *)malloc(total_rows * sizeof(row_t));
    int row = 0;

    for (int pi = 0; pi < N_PAGE_COUNTS; pi++) {
        for (int ii = 0; ii < N_ITER_COUNTS; ii++) {
            int pages = PAGE_COUNTS[pi];
            int iters = ITER_COUNTS[ii];

            fprintf(stderr, "[tc03] RAND_WRITE  pages=%-4d  iters=%-3d  ...\n",
                    pages, iters);

            bench_t off = run_bench(managed, page_order_host, page_order_dev, pages, iters, 0);
            bench_t on = run_bench(managed, page_order_host, page_order_dev, pages, iters, 1);

            rows[row++] = (row_t){pages, iters, off, on};
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
                "RAND_WRITE", r->pages, r->iters,
                r->off.wall_avg_ms, r->on.wall_avg_ms, ovhd_pct);
    }
    fprintf(stderr, "+------------+-------+-------+--------------+--------------+-----------+\n");
    free(rows);
    free(page_order_host);
    CUDA_CHECK(cudaFree(page_order_dev));
    CUDA_CHECK(cudaFree(managed));
    return 0;
}
