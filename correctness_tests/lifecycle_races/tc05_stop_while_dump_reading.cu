/*
 * tc05_stop_while_dump_reading.cu
 *
 * Race: one thread reads from the dump procfs file (slow, byte-by-byte) while
 * the main thread calls stop(). The driver must not deadlock or crash.
 *
 * After the race, the main thread starts a fresh session, writes a page,
 * does cutover+dump, and verifies the page is recorded (driver is functional).
 *
 * Flow:
 *   start (delta) → write many pages → cutover
 *   spawn reader thread that opens dump and reads slowly (with usleep between reads)
 *   main: stop() while reader is mid-read
 *   join reader thread (it may get a partial read or EOF — both are acceptable)
 *   start → write page → cutover → dump → verify page present → stop
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

#define NUM_PAGES   64
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
        for (int i = 0; i < ipp; i++)
            base[p * ipp + i] = tag * 1000 + p * 100 + i;
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

/* Slow reader thread: opens dump and reads byte-by-byte with delay. */
static void *slow_reader(void *arg)
{
    (void)arg;
    int fd = open(PROCFS_DUMP, O_RDONLY);
    if (fd < 0) return (void *)(intptr_t)-1;

    char ch;
    int total = 0;
    while (read(fd, &ch, 1) == 1) {
        total++;
        usleep(500); /* 0.5ms delay per byte — exaggerates the race window */
    }
    close(fd);
    printf("[tc05] slow reader finished: %d bytes read\n", total);
    return (void *)(intptr_t)total;
}

int main(void)
{
    printf("[tc05] stop_while_dump_reading (lifecycle race)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc05] pid=%d alloc=0x%lx\n", getpid(), (unsigned long)managed);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 64>>>(managed, NUM_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc05] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    /* Spawn slow reader. */
    pthread_t reader;
    if (pthread_create(&reader, NULL, slow_reader, NULL) != 0) {
        perror("pthread_create");
        stop_track();
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }

    /* Brief yield so reader can open the file and start reading. */
    usleep(5000); /* 5ms */

    /* Race: stop while reader is mid-read. */
    rc = stop_track();
    if (rc) {
        fprintf(stderr, "[tc05] stop failed: %s\n", strerror(-rc));
        pthread_join(reader, NULL);
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }
    printf("[tc05] stop() returned OK while dump was being read\n");

    pthread_join(reader, NULL);

    /* Verify driver is still functional. */
    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] re-start failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_range<<<1, 1>>>(managed, 1, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc05] final cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc05] final dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(managed)); return 1; }

    unsigned long pa = (unsigned long)managed;
    int found = 0;
    for (int i = 0; i < n; i++) if (e[i].addr == pa) { found = 1; break; }
    printf("[tc05] final session: n=%d found=%d (want found=1)\n", n, found);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = !found;
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (!found) printf("[tc05]   page not recorded in post-race session\n");
    return failed;
}
