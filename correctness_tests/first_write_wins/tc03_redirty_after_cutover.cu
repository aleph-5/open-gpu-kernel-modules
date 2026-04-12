/*
 * tc03_redirty_after_cutover.cu
 *
 * Write P → cutover → dump1 (records ts1).
 * Write P again → cutover → dump2 (records ts2).
 * In delta mode, cutover revokes PTEs so P will re-fault in epoch 2.
 * The new epoch's record is fresh: ts2 > ts1 (first-write-wins resets at
 * each cutover boundary in delta mode — there is no carry-over).
 *
 * This is distinct from tc01 (within a single epoch) and from cumulative-mode
 * behaviour: in delta mode each epoch is independent, so P appears in each
 * epoch's dump with a fresh, monotonically-increasing timestamp.
 *
 * PASS conditions:
 *   dump1: P present, ts1 > 0
 *   dump2: P present, ts2 > 0, ts2 >= ts1
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

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

__global__ void gpu_write_page(int *page, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int i = 0; i < ipp; i++) page[i] = tag * 1000 + i;
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

int main(void)
{
    printf("[tc03] redirty_after_cutover (delta mode, two epochs)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long pa_P = (unsigned long)managed;
    printf("[tc03] pid=%d P=0x%lx\n", getpid(), pa_P);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start failed: %s\n", strerror(-rc)); return 1; }

    /* Epoch 1: write P. */
    gpu_write_page<<<1, 1>>>(managed, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc03] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }

    unsigned long ts1 = 0;
    for (int i = 0; i < n1; i++)
        if (snap1[i].addr == pa_P) { ts1 = snap1[i].ts; break; }
    printf("[tc03] epoch1: n=%d, P ts1=%lu\n", n1, ts1);

    /* Epoch 2: write P again. Cutover from epoch 1 revoked PTEs so P will
     * re-fault and be recorded fresh with ts2 >= ts1. */
    gpu_write_page<<<1, 1>>>(managed, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc03] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    unsigned long ts2 = 0;
    int p2_count = 0;
    for (int i = 0; i < n2; i++) {
        if (snap2[i].addr == pa_P) { p2_count++; ts2 = snap2[i].ts; }
    }
    printf("[tc03] epoch2: n=%d, P count=%d ts2=%lu (want count=1, ts2>=%lu)\n",
           n2, p2_count, ts2, ts1);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (ts1 == 0 || p2_count != 1 || ts2 == 0 || ts2 < ts1);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (ts1 == 0)       printf("[tc03]   P not recorded in epoch 1\n");
    if (p2_count == 0)  printf("[tc03]   P missing from epoch 2 dump\n");
    if (p2_count > 1)   printf("[tc03]   P appears %d times in epoch 2 (expected 1)\n", p2_count);
    if (ts2 < ts1)      printf("[tc03]   ts regression: ts2=%lu < ts1=%lu\n", ts2, ts1);
    return failed;
}
