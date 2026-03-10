/*
 * tc02_ts_sleep_coherence.cu — latency tests
 *
 * Each dirty_pages entry carries a `ts` field whose unit is not documented
 * here — it could be nanoseconds, jiffies, or a driver-internal counter.
 * Regardless of unit, the ts values must faithfully reflect real elapsed
 * time between events.  This test verifies that property without assuming
 * the unit.
 *
 * Protocol:
 *   1. Start tracking.
 *   2. Write BATCH_PAGES pages to alloc_a (batch A), then synchronize.
 *      Record wall-clock instant t_a.
 *   3. Sleep SLEEP_NS nanoseconds (wall clock).
 *   4. Write BATCH_PAGES pages to alloc_b (batch B), then synchronize.
 *      Record wall-clock instant t_b.
 *   5. Read all dirty_pages entries; separate into sets A and B by address.
 *   6. Compute ts_A_max and ts_B_min.
 *   7. Infer the ts rate from the overall span:
 *        rate  = (ts_B_max − ts_A_min) / (t_b − t_a_kernel_start)
 *        expected_gap = SLEEP_NS × rate
 *   8. Compute actual_gap = ts_B_min − ts_A_max.
 *
 * Pass criteria:
 *   1. All 2×BATCH_PAGES entries are present.
 *   2. ts_A_max < ts_B_min  — every batch-B entry is strictly newer than
 *      every batch-A entry.  A 100 ms sleep makes this non-trivial: if ts
 *      values are coalesced or stamped at read time rather than fault time,
 *      the separation will collapse.
 *   3. actual_gap >= RATIO_THRESHOLD × expected_gap  — the ts gap must be
 *      at least RATIO_THRESHOLD (50 %) of what the wall-clock sleep predicts.
 *      This catches implementations that compress timestamps or use a clock
 *      with coarse resolution (e.g., HZ=100 jiffies = 10 ms ticks would
 *      still pass; a broken counter that never advances would fail).
 *
 * Why this is hard to pass:
 *   - Criterion 2 fails if the fault handler assigns ts from a cached
 *     snapshot rather than a fresh clock read per fault.
 *   - Criterion 3 fails if ts resolution is coarser than ~50 ms, or if
 *     timestamps are not tied to the real-time clock at all.
 *   - Together they rule out both "all entries get the same ts" and
 *     "ts advances but too slowly" failure modes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <cuda_runtime.h>

#define PROCFS_START  "/proc/driver/nvidia-uvm/dirty_pids_start_track"
#define PROCFS_STOP   "/proc/driver/nvidia-uvm/dirty_pids_stop_track"
#define PROCFS_QUERY  "/proc/driver/nvidia-uvm/dirty_pid_to_query"
#define PROCFS_PAGES  "/proc/driver/nvidia-uvm/dirty_pages"

#define BATCH_PAGES     32
#define PAGE_SIZE       4096
#define INTS_PER_PAGE   (PAGE_SIZE / sizeof(int))
#define MAX_ENTRIES     4096

#define SLEEP_NS        100000000L  /* 100 ms deliberate gap between batches */
#define RATIO_THRESHOLD 0.5         /* actual_gap must be >= 50% of expected  */
#define POLL_TIMEOUT_S  5

#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while (0)

typedef struct { unsigned long addr, ts; int pid; } entry_t;

__global__ void write_batch(int *data, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        data[i] = i + 7;
}

static void procfs_write(const char *path, const char *val) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) { perror(path); exit(1); }
    write(fd, val, strlen(val));
    close(fd);
}

static void set_query_pid(pid_t p) {
    char b[32];
    snprintf(b, sizeof(b), "%d\n", p);
    procfs_write(PROCFS_QUERY, b);
}

static int read_dirty(entry_t *out, int max) {
    FILE *f = fopen(PROCFS_PAGES, "r");
    if (!f) return -1;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (strstr(line, "not active")) { fclose(f); return -2; }
            continue;
        }
        if (n < max &&
            sscanf(line, "0x%lx %lu %d", &out[n].addr, &out[n].ts, &out[n].pid) == 3)
            n++;
    }
    fclose(f);
    return n;
}

static long ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static void sleep_ns(long ns) {
    struct timespec req = { ns / 1000000000L, ns % 1000000000L };
    nanosleep(&req, NULL);
}

int main(void) {
    printf("[tc02] ts_sleep_coherence — %d pages/batch, sleep=%ldms, threshold=%.0f%%\n",
           BATCH_PAGES, SLEEP_NS / 1000000L, RATIO_THRESHOLD * 100.0);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, (size_t)BATCH_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, (size_t)BATCH_PAGES * PAGE_SIZE));
    memset(alloc_a, 0, (size_t)BATCH_PAGES * PAGE_SIZE);
    memset(alloc_b, 0, (size_t)BATCH_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base_a = (unsigned long)alloc_a;
    unsigned long base_b = (unsigned long)alloc_b;
    pid_t pid = getpid();
    printf("[tc02] pid=%d  alloc_a=0x%lx  alloc_b=0x%lx\n", pid, base_a, base_b);
    set_query_pid(pid);
    procfs_write(PROCFS_START, "1\n");

    /* --- Batch A --------------------------------------------------------- */
    long t_wall_start = ns_now();
    write_batch<<<1, 256>>>(alloc_a, BATCH_PAGES * (int)INTS_PER_PAGE);
    CUDA_CHECK(cudaDeviceSynchronize());
    long t_a_done = ns_now();
    printf("[tc02] batch A synced at wall +%ldms\n",
           (t_a_done - t_wall_start) / 1000000L);

    /* --- Deliberate gap -------------------------------------------------- */
    sleep_ns(SLEEP_NS);

    /* --- Batch B --------------------------------------------------------- */
    write_batch<<<1, 256>>>(alloc_b, BATCH_PAGES * (int)INTS_PER_PAGE);
    CUDA_CHECK(cudaDeviceSynchronize());
    long t_b_done = ns_now();
    printf("[tc02] batch B synced at wall +%ldms\n",
           (t_b_done - t_wall_start) / 1000000L);

    /* --- Poll until both batches are fully visible ----------------------- */
    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    if (!e) { fprintf(stderr, "malloc failed\n"); return 1; }

    int n = 0;
    long t_poll_deadline = ns_now() + (long)POLL_TIMEOUT_S * 1000000000L;
    while (1) {
        n = read_dirty(e, MAX_ENTRIES);
        if (n >= 2 * BATCH_PAGES) break;
        if (ns_now() >= t_poll_deadline) break;
    }
    long t_read_done = ns_now();
    printf("[tc02] read %d/%d entries after poll\n", n, 2 * BATCH_PAGES);

    procfs_write(PROCFS_STOP, "1\n");

    /* --- Separate entries by allocation address -------------------------- */
    unsigned long ts_a_min = ~0UL, ts_a_max = 0;
    unsigned long ts_b_min = ~0UL, ts_b_max = 0;
    int a_count = 0, b_count = 0;

    for (int i = 0; i < n; i++) {
        unsigned long base = BATCH_PAGES * (unsigned long)PAGE_SIZE;
        if (e[i].addr >= base_a && e[i].addr < base_a + base) {
            if (e[i].ts < ts_a_min) ts_a_min = e[i].ts;
            if (e[i].ts > ts_a_max) ts_a_max = e[i].ts;
            a_count++;
        } else if (e[i].addr >= base_b && e[i].addr < base_b + base) {
            if (e[i].ts < ts_b_min) ts_b_min = e[i].ts;
            if (e[i].ts > ts_b_max) ts_b_max = e[i].ts;
            b_count++;
        }
    }

    printf("[tc02] batch A: %d pages  ts=[%lu, %lu]\n", a_count, ts_a_min, ts_a_max);
    printf("[tc02] batch B: %d pages  ts=[%lu, %lu]\n", b_count, ts_b_min, ts_b_max);

    CUDA_CHECK(cudaFree(alloc_a));
    CUDA_CHECK(cudaFree(alloc_b));
    free(e);

    int failed = 0;

    /* Check 1: completeness */
    if (a_count != BATCH_PAGES || b_count != BATCH_PAGES) {
        printf("[tc02] FAIL: incomplete entries — a=%d/%d  b=%d/%d\n",
               a_count, BATCH_PAGES, b_count, BATCH_PAGES);
        failed = 1;
        goto done;
    }

    /* Check 2: strict ts ordering across the sleep boundary */
    if (ts_a_max >= ts_b_min) {
        printf("[tc02] FAIL: ts ordering violated — ts_a_max=%lu >= ts_b_min=%lu\n"
               "             (every batch-B entry must be strictly newer than batch-A)\n",
               ts_a_max, ts_b_min);
        failed = 1;
    }

    /* Check 3: ts gap proportional to the wall-clock sleep (unit-independent) */
    {
        unsigned long ts_total_span  = ts_b_max - ts_a_min;
        long          wall_total_ns  = t_read_done - t_wall_start;

        if (ts_total_span == 0) {
            printf("[tc02] FAIL: ts_total_span is 0 — timestamps have no resolution\n");
            failed = 1;
        } else {
            /*
             * Estimate the ts gap we expect to see across the sleep:
             *   expected_gap = SLEEP_NS * (ts_total_span / wall_total_ns)
             * Use double arithmetic; both spans can be large but fit in double.
             */
            double rate          = (double)ts_total_span / (double)wall_total_ns;
            double expected_gap  = (double)SLEEP_NS * rate;
            unsigned long actual_gap = (ts_b_min > ts_a_max) ? (ts_b_min - ts_a_max) : 0;
            double ratio         = (double)actual_gap / expected_gap;

            printf("[tc02] ts_total_span=%lu  wall_total_ns=%ld\n",
                   ts_total_span, wall_total_ns);
            printf("[tc02] expected_gap=%.0f  actual_gap=%lu  ratio=%.3f (need >= %.2f)\n",
                   expected_gap, actual_gap, ratio, RATIO_THRESHOLD);

            if (ratio < RATIO_THRESHOLD) {
                printf("[tc02] FAIL: ts gap too small (%.3f < %.2f) — timestamps do not\n"
                       "             reflect real elapsed time across the %ldms sleep\n",
                       ratio, RATIO_THRESHOLD, SLEEP_NS / 1000000L);
                failed = 1;
            }
        }
    }

done:
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
