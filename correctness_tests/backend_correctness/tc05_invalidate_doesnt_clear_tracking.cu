/*
 * tc05_invalidate_clears_tracking.cu
 *
 * When a managed allocation is freed (cudaFree), UVM sends an invalidate
 * notification to the dirty tracker. The dirty entry for the freed allocation
 * must PERSIST in the tracker — invalidate does not remove records.
 *
 * Flow:
 *   start(delta)
 *   alloc A → write A → sync
 *   alloc B → write B → sync
 *   cudaFree(A) → invalidate notification sent for A's pages
 *   cutover → dump
 *   PASS: both A's and B's pages present in dump
 *         (A's entries survived the invalidate)
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

#define PAGES_EACH  16
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

__global__ void gpu_write_all(int *base, int num_pages, int tag)
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

static int count_in_alloc(entry_t *e, int n, unsigned long base, int num_pages)
{
    int found = 0;
    for (int p = 0; p < num_pages; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++) if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc05] invalidate_persists_entry: %d pages each\n", PAGES_EACH);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *alloc_a = NULL, *alloc_b = NULL;
    CUDA_CHECK(cudaMallocManaged(&alloc_a, PAGES_EACH * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&alloc_b, PAGES_EACH * PAGE_SIZE));
    memset(alloc_a, 0, PAGES_EACH * PAGE_SIZE);
    memset(alloc_b, 0, PAGES_EACH * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base_a = (unsigned long)alloc_a;
    unsigned long base_b = (unsigned long)alloc_b;
    printf("[tc05] pid=%d A=0x%lx B=0x%lx\n", getpid(), base_a, base_b);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc05] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_all<<<1, 32>>>(alloc_a, PAGES_EACH, 1);
    gpu_write_all<<<1, 32>>>(alloc_b, PAGES_EACH, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Free alloc A — UVM sends an invalidate for A's pages. The dirty entries
     * must survive: invalidate does not remove records from the tracker. */
    CUDA_CHECK(cudaFree(alloc_a));
    alloc_a = NULL;

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc05] cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(alloc_b)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc05] dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(alloc_b)); return 1; }

    int a_found = count_in_alloc(e, n, base_a, PAGES_EACH);
    int b_found = count_in_alloc(e, n, base_b, PAGES_EACH);
    printf("[tc05] dump: n=%d A=%d/%d B=%d/%d (want A=%d B=%d)\n",
           n, a_found, PAGES_EACH, b_found, PAGES_EACH, PAGES_EACH, PAGES_EACH);

    stop_track();
    CUDA_CHECK(cudaFree(alloc_b));

    int failed = (a_found != PAGES_EACH || b_found != PAGES_EACH);
    printf("[tc05] %s\n", failed ? "FAIL" : "PASS");
    if (a_found != PAGES_EACH) printf("[tc05]   A: only %d/%d pages in dump (invalidate wrongly cleared entries)\n", a_found, PAGES_EACH);
    if (b_found != PAGES_EACH) printf("[tc05]   B: only %d/%d pages in dump\n", b_found, PAGES_EACH);
    return failed;
}
