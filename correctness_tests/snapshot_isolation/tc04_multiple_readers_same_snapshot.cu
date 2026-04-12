/*
 * tc04_multiple_readers_same_snapshot.cu
 *
 * In delta mode, the dump procfs file reads AND DESTROYS the snapshot.
 * A second read of the same dump file (after the first reader consumes it)
 * must return 0 entries (empty — already drained).
 *
 * Additionally, two threads opening the dump simultaneously (races at open
 * time) must together see each page exactly once (the first opener gets the
 * snapshot, the second gets empty).
 *
 * Flow:
 *   start(delta) → write pages → cutover
 *   thread-1: open dump, read fully → record count1
 *   thread-2: open dump, read fully → record count2
 *   PASS: count1 + count2 == NUM_PAGES (each page seen exactly once total)
 *         (one thread gets all pages, the other gets 0 — order not guaranteed)
 *
 *   Second check (serial): open dump again after first read → must return 0.
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

#define NUM_PAGES   16
#define PAGE_SIZE   4096
#define MAX_ENTRIES 4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_range(int *base, int num_pages, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = 0; p < num_pages; p++)
        for (int i = 0; i < ipp; i++) base[p * ipp + i] = tag * 1000 + p * 100 + i;
}

static int procfs_write_exact(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -errno;
    ssize_t n = write(fd, val, strlen(val));
    int saved = errno;
    close(fd);
    if (n < 0) return -saved;
    return 0;
}

static int start_track_delta(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d delta", getpid());
    return procfs_write_exact(PROCFS_START, buf);
}

static int stop_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_STOP, buf);
}

static int cutover(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_CUTOVER, buf);
}

static int read_dump(entry_t *out, int max)
{
    FILE *f = fopen(PROCFS_DUMP, "r");
    if (!f) return -errno;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (n < max && sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2)
            n++;
    }
    if (ferror(f)) { int saved = errno; fclose(f); return -saved; }
    fclose(f);
    return n;
}

typedef struct {
    int count;
    int done;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} reader_ctx_t;

static reader_ctx_t g_ctx;

static void *reader_thread(void *arg)
{
    (void)arg;
    entry_t *e = (entry_t *)malloc(MAX_ENTRIES * sizeof(entry_t));
    int n = (e != NULL) ? read_dump(e, MAX_ENTRIES) : -ENOMEM;
    free(e);

    pthread_mutex_lock(&g_ctx.lock);
    if (n > 0) g_ctx.count += n;
    g_ctx.done++;
    pthread_cond_signal(&g_ctx.cond);
    pthread_mutex_unlock(&g_ctx.lock);
    return (void *)(intptr_t)(n > 0 ? n : 0);
}

int main(void)
{
    printf("[tc04] multiple_readers_same_snapshot: %d pages\n", NUM_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc04] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc04] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, NUM_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    /* Two threads race to read the same snapshot. */
    pthread_mutex_init(&g_ctx.lock, NULL);
    pthread_cond_init(&g_ctx.cond, NULL);
    g_ctx.count = 0;
    g_ctx.done  = 0;

    pthread_t t1, t2;
    pthread_create(&t1, NULL, reader_thread, NULL);
    pthread_create(&t2, NULL, reader_thread, NULL);

    pthread_mutex_lock(&g_ctx.lock);
    while (g_ctx.done < 2)
        pthread_cond_wait(&g_ctx.cond, &g_ctx.lock);
    int total_seen = g_ctx.count;
    pthread_mutex_unlock(&g_ctx.lock);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("[tc04] two concurrent readers saw total %d entries (want %d)\n",
           total_seen, NUM_PAGES);

    /* Serial second read — snapshot must already be drained. */
    entry_t e2[MAX_ENTRIES];
    int n_serial = read_dump(e2, MAX_ENTRIES);
    printf("[tc04] serial second read: %d entries (want 0)\n",
           n_serial < 0 ? -1 : n_serial);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (total_seen != NUM_PAGES || n_serial != 0);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (total_seen != NUM_PAGES)
        printf("[tc04]   two readers saw %d total (want %d — each page exactly once)\n",
               total_seen, NUM_PAGES);
    if (n_serial != 0)
        printf("[tc04]   serial re-read got %d entries (want 0 — snapshot must drain)\n", n_serial);
    return failed;
}
