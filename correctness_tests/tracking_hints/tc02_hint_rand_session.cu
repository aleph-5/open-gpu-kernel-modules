/*
 * tc02_hint_rand_session.cu
 *
 * Verify that selecting WRITE_RAND via the procfs hint (which selects the
 * nested-bitmap backend, uvm_dirty_ds_nested_bitmap_ops) yields a correct
 * dump for a random sparse write workload.
 *
 * Random sparse writes are the workload bitmap-style backends are intended
 * to handle well, so this is the natural correctness check for the rand hint.
 *
 * Flow:
 *   write hint=WRITE_RAND → start(delta) → write a deterministic random subset
 *     of pages → cutover → dump → assert exactly the written set is present,
 *     no extras, no duplicates, addresses sorted.
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
#define PROCFS_HINT    "/proc/driver/nvidia-uvm/dirty_tracking_hint"

#define TOTAL_PAGES 512
#define WRITTEN_MAX 200
#define PAGE_SIZE   4096
#define MAX_ENTRIES 4096
#define SEED        0xC0FFEE

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

__global__ void gpu_write_indices(int *base, const int *indices, int n)
{
    int t = blockIdx.x;
    if (t >= n) return;
    int p = indices[t];
    int ipp = PAGE_SIZE / sizeof(int);
    base[p * ipp] = p;
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
    char buf[32]; snprintf(buf, sizeof(buf), "%d delta", getpid());
    return procfs_write_exact(PROCFS_START, buf);
}
static int stop_track(void)
{
    char buf[16]; snprintf(buf, sizeof(buf), "%d", getpid());
    return procfs_write_exact(PROCFS_STOP, buf);
}
static int cutover(void)
{
    char buf[16]; snprintf(buf, sizeof(buf), "%d", getpid());
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
    printf("[tc02] hint_rand_session: %d pages total, ~%d randomly written, hint=WRITE_RAND\n",
           TOTAL_PAGES, WRITTEN_MAX);
    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int rc = procfs_write_exact(PROCFS_HINT, "WRITE_RAND");
    if (rc) { fprintf(stderr, "[tc02] hint write failed: %s\n", strerror(-rc)); return 1; }

    /* Build a deterministic random subset (no duplicates) by shuffle-sample. */
    srand(SEED);
    int perm[TOTAL_PAGES];
    for (int i = 0; i < TOTAL_PAGES; i++) perm[i] = i;
    for (int i = TOTAL_PAGES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    int written_n = WRITTEN_MAX;
    int *expected = perm;  /* first written_n entries, unsorted page indices */

    size_t alloc_bytes = (size_t)TOTAL_PAGES * PAGE_SIZE;
    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, alloc_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_indices = NULL;
    CUDA_CHECK(cudaMallocManaged(&d_indices, written_n * sizeof(int)));
    memcpy(d_indices, expected, written_n * sizeof(int));

    unsigned long base = (unsigned long)managed;
    printf("[tc02] pid=%d alloc=0x%lx written_n=%d\n", getpid(), base, written_n);

    rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc02] start failed: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(d_indices)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_indices<<<written_n, 1>>>(managed, d_indices, written_n);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc02] cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(d_indices)); CUDA_CHECK(cudaFree(managed)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc02] dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(d_indices)); CUDA_CHECK(cudaFree(managed)); return 1; }
    printf("[tc02] dump: %d entries (want %d)\n", n, written_n);

    /* Validity checks. */
    int out_of_range = 0, unsorted = 0, duplicates = 0;
    for (int i = 0; i < n; i++) {
        if (e[i].addr < base || e[i].addr >= base + alloc_bytes) out_of_range++;
        if (i > 0) {
            if (e[i].addr < e[i - 1].addr) unsorted++;
            if (e[i].addr == e[i - 1].addr) duplicates++;
        }
    }

    /* Per-page expected/got membership. */
    char expected_set[TOTAL_PAGES] = {0};
    for (int i = 0; i < written_n; i++) expected_set[expected[i]] = 1;

    char got_set[TOTAL_PAGES] = {0};
    int unexpected = 0;
    for (int i = 0; i < n; i++) {
        unsigned long off = e[i].addr - base;
        unsigned long page_idx = off / PAGE_SIZE;
        if (page_idx < TOTAL_PAGES) got_set[page_idx] = 1;
        if (page_idx >= TOTAL_PAGES || !expected_set[page_idx]) unexpected++;
    }
    int missing = 0;
    for (int p = 0; p < TOTAL_PAGES; p++)
        if (expected_set[p] && !got_set[p]) missing++;

    stop_track();
    procfs_write_exact(PROCFS_HINT, "WRITE_SEQ"); /* restore default */
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(managed));

    int failed = (n != written_n || out_of_range || unsorted || duplicates || missing || unexpected);
    printf("[tc02] %s\n", failed ? "FAIL" : "PASS");
    if (n != written_n) printf("[tc02]   count: got %d want %d\n", n, written_n);
    if (out_of_range)   printf("[tc02]   %d out-of-range entries\n", out_of_range);
    if (unsorted)       printf("[tc02]   %d unsorted neighbour pairs\n", unsorted);
    if (duplicates)     printf("[tc02]   %d duplicate addresses\n", duplicates);
    if (missing)        printf("[tc02]   %d expected pages missing\n", missing);
    if (unexpected)     printf("[tc02]   %d unexpected pages in dump\n", unexpected);
    return failed;
}
