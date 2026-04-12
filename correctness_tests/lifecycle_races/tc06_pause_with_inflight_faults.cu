/*
 * tc06_pause_with_inflight_faults.cu
 *
 * Race: pause() is called while a GPU kernel is actively generating UVM faults.
 * Pages faulted before pause() must be recorded; pages faulted after pause()
 * must NOT be recorded. The resume() call re-enables recording.
 *
 * Because we cannot perfectly synchronize GPU faults with pause(), this test
 * uses a conservative split: two separate kernels with a sync point between
 * them to control which pages are definitely pre-pause and post-pause.
 *
 * Flow:
 *   start (delta)
 *   write phase-A pages → sync → pause()
 *   write phase-B pages → sync → resume()
 *   cutover → dump
 *   PASS: phase-A pages present, phase-B pages absent
 *         (OR: total count == phase-A count — conservative assertion)
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
#define PROCFS_PAUSE   "/proc/driver/nvidia-uvm/dirty_tracking_pause"
#define PROCFS_RESUME  "/proc/driver/nvidia-uvm/dirty_tracking_resume"

#define PHASE_A_PAGES 8
#define PHASE_B_PAGES 8
#define TOTAL_PAGES   (PHASE_A_PAGES + PHASE_B_PAGES)
#define PAGE_SIZE     4096
#define MAX_ENTRIES   4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_range(int *base, int start_page, int end_page, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = start_page; p < end_page; p++)
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

static int pause_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_PAUSE, buf);
}

static int resume_track(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_RESUME, buf);
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

static int count_in_range(entry_t *e, int n, unsigned long base, int start_p, int end_p)
{
    int found = 0;
    for (int p = start_p; p < end_p; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc06] pause_with_inflight_faults: A=%d B=%d pages\n",
           PHASE_A_PAGES, PHASE_B_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc06] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc06] start failed: %s\n", strerror(-rc)); return 1; }

    /* Phase A: write before pause. */
    gpu_write_range<<<1, 32>>>(managed, 0, PHASE_A_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize()); /* ensure faults resolved before pause */

    rc = pause_track();
    if (rc) { fprintf(stderr, "[tc06] pause failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    /* Phase B: write while paused — these faults must NOT be recorded. */
    gpu_write_range<<<1, 32>>>(managed, PHASE_A_PAGES, TOTAL_PAGES, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = resume_track();
    if (rc) { fprintf(stderr, "[tc06] resume failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc06] cutover failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc06] dump failed: %s\n", strerror(-n)); stop_track(); return 1; }

    int a_found = count_in_range(e, n, base, 0, PHASE_A_PAGES);
    int b_found = count_in_range(e, n, base, PHASE_A_PAGES, TOTAL_PAGES);
    printf("[tc06] dump: n=%d phase-A=%d/%d phase-B=%d/%d (want A=%d B=0)\n",
           n, a_found, PHASE_A_PAGES, b_found, PHASE_B_PAGES, PHASE_A_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (a_found != PHASE_A_PAGES || b_found != 0);
    printf("[tc06] %s\n", failed ? "FAIL" : "PASS");
    if (a_found != PHASE_A_PAGES)
        printf("[tc06]   only %d/%d phase-A pages recorded (should all be recorded)\n",
               a_found, PHASE_A_PAGES);
    if (b_found != 0)
        printf("[tc06]   %d phase-B pages recorded while paused (should be 0)\n", b_found);
    return failed;
}
