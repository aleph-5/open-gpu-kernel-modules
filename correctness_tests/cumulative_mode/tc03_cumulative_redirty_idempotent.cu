/*
 * tc03_cumulative_redirty_idempotent.cu
 *
 * Write page P in epoch 1 (cutover → dump1). Write P again in epoch 2
 * (cutover → dump2). In cumulative mode, dump2 must show P exactly once
 * (first-write-wins across epochs: P was first seen in epoch 1 and its
 * epoch-1 timestamp is preserved in the cumulative set). In delta mode,
 * P would appear in dump1 only (epoch 1 delta) and in dump2 only (epoch 2
 * delta) but not across – this test exercises the cumulative invariant.
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

#define NUM_PAGES   4
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

__global__ void gpu_write_page(int *base, int page_idx, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    int *p = base + page_idx * ipp;
    for (int i = 0; i < ipp; i++) p[i] = tag * 1000 + i;
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

static int start_track_cumulative(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d cumulative", getpid());
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
    printf("[tc03] cumulative_redirty_idempotent\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, NUM_PAGES * PAGE_SIZE));
    memset(managed, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    unsigned long pa_P = base; /* page 0 is P */
    printf("[tc03] pid=%d alloc=0x%lx, P=0x%lx\n", getpid(), base, pa_P);

    int rc = start_track_cumulative();
    if (rc) { fprintf(stderr, "[tc03] start failed: %s\n", strerror(-rc)); return 1; }

    /* Epoch 1: write P. */
    gpu_write_page<<<1, 1>>>(managed, 0, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    if (n1 < 0) { fprintf(stderr, "[tc03] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }

    unsigned long ts1 = 0;
    for (int i = 0; i < n1; i++)
        if (snap1[i].addr == pa_P) { ts1 = snap1[i].ts; break; }
    printf("[tc03] epoch1 dump: n=%d, P ts1=%lu\n", n1, ts1);

    /* Epoch 2: write P again (will be a no-op for first-write-wins on live ds,
     * but we need to force a fault: P's PTE was set up; we rely on PTEs being
     * revoked at cutover via the barrier. If P does not re-fault, that's fine
     * since first-write-wins means the cumulative entry still has ts1. */
    gpu_write_page<<<1, 1>>>(managed, 0, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc03] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    /* Count occurrences of P in snap2. */
    int p_count = 0;
    unsigned long ts2_p = 0;
    for (int i = 0; i < n2; i++) {
        if (snap2[i].addr == pa_P) {
            p_count++;
            ts2_p = snap2[i].ts;
        }
    }
    printf("[tc03] epoch2 dump: n=%d, P count=%d ts=%lu (expect count=1)\n",
           n2, p_count, ts2_p);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    /* P must appear exactly once (idempotent). ts1 must be non-zero (P recorded). */
    int failed = (ts1 == 0 || p_count != 1);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (ts1 == 0)      printf("[tc03]   P not recorded in epoch 1\n");
    if (p_count == 0)  printf("[tc03]   P missing from cumulative dump2\n");
    if (p_count > 1)   printf("[tc03]   P appears %d times in cumulative (expected 1)\n", p_count);
    return failed;
}
