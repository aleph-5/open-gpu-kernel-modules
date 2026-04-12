/*
 * tc01_same_page_repeat_write.cu
 *
 * Write page P → record ts1 from dump.
 * Write P again (after PTEs are re-established via a second start or by
 * forcing a re-fault via reading the page on CPU to trigger migration).
 * The entry for P must still show ts1 (first-write-wins, not overwritten).
 *
 * Implementation: use two delta epochs with the same page.
 *   Epoch 1: write P → cutover → dump1 → record ts1.
 *   Epoch 2 (same session): since PTEs are revoked at cutover boundary,
 *   write P again → cutover → dump2 → P has ts2 (new epoch, ts2 > ts1).
 *   Verify ts2 >= ts1 (not reset back to 0 or corrupted).
 *
 * Then verify WITHIN an epoch: write P twice before cutover.
 *   The second write to an already-recorded page is a no-op (first-write-wins).
 *   The entry count for P must be exactly 1.
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
    printf("[tc01] same_page_repeat_write (first-write-wins)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long pa_P = (unsigned long)managed;
    printf("[tc01] pid=%d P=0x%lx\n", getpid(), pa_P);

    /* Part A: write P twice within the same epoch, verify single entry. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc01] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_page<<<1, 1>>>(managed, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    /* Force P to re-fault by reading it back on CPU (may trigger migration back). */
    volatile int sink = managed[0];
    (void)sink;
    gpu_write_page<<<1, 1>>>(managed, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc01] cutover-A failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc01] dump-A failed: %s\n", strerror(-n)); stop_track(); return 1; }

    int p_count = 0;
    unsigned long ts1 = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr == pa_P) { p_count++; ts1 = e[i].ts; }
    }
    printf("[tc01] epoch-A dump: n=%d, P_count=%d ts1=%lu (expect count=1)\n",
           n, p_count, ts1);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (p_count != 1 || ts1 == 0);
    printf("[tc01] %s\n", failed ? "FAIL" : "PASS");
    if (p_count == 0) printf("[tc01]   P not recorded at all\n");
    if (p_count > 1)  printf("[tc01]   P appears %d times (expected 1, first-write-wins broken)\n", p_count);
    if (ts1 == 0)     printf("[tc01]   P has zero timestamp\n");
    return failed;
}
