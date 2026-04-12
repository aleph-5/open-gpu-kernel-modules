/*
 * tc02_timestamp_frozen_at_cutover.cu
 *
 * Once a page is captured in a snapshot (via cutover), its timestamp must be
 * frozen — subsequent writes to the same page (in the next epoch) must NOT
 * retroactively change the timestamp stored in the snapshot.
 *
 * In delta mode:
 *   Epoch 1: write P → cutover → record ts1 from dump1.
 *   Epoch 2: write P again → cutover → record ts2 from dump2.
 *   The dump is destructive (delta mode), so ts1 and ts2 are from separate
 *   snapshots. ts2 must be >= ts1 (no regression), and ts1 must be non-zero.
 *
 * In cumulative mode (additional check):
 *   Write P → cutover → ts1.  Write P again → cutover → cumulative dump.
 *   P must appear exactly once in cumulative dump with ts = ts1 (preserved,
 *   not overwritten by the second write).
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

static int start_track(const char *mode)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d %s", getpid(), mode);
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
    printf("[tc02] timestamp_frozen_at_cutover\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long pa_P = (unsigned long)managed;
    printf("[tc02] pid=%d P=0x%lx\n", getpid(), pa_P);

    /* --- Delta mode check --- */
    int rc = start_track("delta");
    if (rc) { fprintf(stderr, "[tc02] start-delta failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_page<<<1, 1>>>(managed, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover1 failed\n"); stop_track(); return 1; }

    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump(snap1, MAX_ENTRIES);
    unsigned long ts1 = 0;
    for (int i = 0; i < n1; i++) if (snap1[i].addr == pa_P) { ts1 = snap1[i].ts; break; }
    printf("[tc02] delta epoch1: ts1=%lu\n", ts1);

    gpu_write_page<<<1, 1>>>(managed, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover2 failed\n"); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    unsigned long ts2 = 0;
    for (int i = 0; i < n2; i++) if (snap2[i].addr == pa_P) { ts2 = snap2[i].ts; break; }
    printf("[tc02] delta epoch2: ts2=%lu (want >= ts1=%lu)\n", ts2, ts1);

    stop_track();

    /* --- Cumulative mode check --- */
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = start_track("cumulative");
    if (rc) { fprintf(stderr, "[tc02] start-cumulative failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_page<<<1, 1>>>(managed, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cum cutover1 failed\n"); stop_track(); return 1; }

    entry_t cum1[MAX_ENTRIES];
    int nc1 = read_dump(cum1, MAX_ENTRIES);
    unsigned long cum_ts1 = 0;
    for (int i = 0; i < nc1; i++) if (cum1[i].addr == pa_P) { cum_ts1 = cum1[i].ts; break; }
    printf("[tc02] cumulative epoch1: ts=%lu\n", cum_ts1);

    gpu_write_page<<<1, 1>>>(managed, 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cum cutover2 failed\n"); stop_track(); return 1; }

    entry_t cum2[MAX_ENTRIES];
    int nc2 = read_dump(cum2, MAX_ENTRIES);
    int p_count = 0;
    unsigned long cum_ts2 = 0;
    for (int i = 0; i < nc2; i++) if (cum2[i].addr == pa_P) { p_count++; cum_ts2 = cum2[i].ts; }
    printf("[tc02] cumulative epoch2: count=%d ts=%lu (want count=1, ts==cum_ts1=%lu)\n",
           p_count, cum_ts2, cum_ts1);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (ts1 == 0 || ts2 < ts1 || cum_ts1 == 0 || p_count != 1 || cum_ts2 != cum_ts1);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (ts1 == 0)            printf("[tc02]   delta: P not recorded in epoch 1\n");
    if (ts2 < ts1)           printf("[tc02]   delta: ts regression ts2=%lu < ts1=%lu\n", ts2, ts1);
    if (cum_ts1 == 0)        printf("[tc02]   cumulative: P not recorded\n");
    if (p_count != 1)        printf("[tc02]   cumulative: P count=%d (want 1)\n", p_count);
    if (cum_ts2 != cum_ts1)  printf("[tc02]   cumulative: ts changed ts2=%lu ts1=%lu (frozen broken)\n",
                                     cum_ts2, cum_ts1);
    return failed;
}
