/*
 * tc03_concurrent_insert_during_dump.cu
 *
 * A GPU kernel writes new pages while the dump file is being read by a
 * background thread. The dump was created by a prior cutover, so any
 * pages written concurrently are in the NEXT epoch's live data structure,
 * not in the snapshot being read.
 *
 * Safety invariant: the dump must complete without crash, deadlock, or
 * memory corruption. The concurrent inserts must not corrupt the snapshot.
 *
 * Correctness invariant: the dump must return exactly the pages from the
 * epoch that was cutover (not mixed with new pages written after cutover).
 *
 * Flow:
 *   start(delta)
 *   write phase-A pages → cutover
 *   [snapshot now fixed; phase-A pages in snapshot]
 *   spawn background GPU thread writing phase-B pages continuously
 *   foreground thread: open dump and read slowly (with delays)
 *   join GPU thread; cutover2 → dump2
 *   PASS:
 *     dump1 contains only phase-A pages
 *     dump2 contains only phase-B pages (delta)
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

#define PHASE_PAGES  8
#define TOTAL_PAGES  (PHASE_PAGES * 2)
#define PAGE_SIZE    4096
#define MAX_ENTRIES  4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

typedef struct { unsigned long addr, ts; } entry_t;

typedef struct {
    int *base;
    int start_page;
    int end_page;
    int tag;
} write_args_t;

__global__ void gpu_write_range(int *base, int start_page, int end_page, int tag)
{
    int ipp = PAGE_SIZE / sizeof(int);
    for (int p = start_page; p < end_page; p++)
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

static int read_dump_slow(entry_t *out, int max)
{
    int fd = open(PROCFS_DUMP, O_RDONLY);
    if (fd < 0) return -errno;

    char rawbuf[MAX_ENTRIES * 64];
    int total = 0;
    char ch;
    while (read(fd, &ch, 1) == 1) {
        if (total < (int)sizeof(rawbuf) - 1) rawbuf[total++] = ch;
        usleep(200); /* slow reader: 200µs per byte */
    }
    close(fd);
    rawbuf[total] = '\0';

    int n = 0;
    char *line = rawbuf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        unsigned long addr, ts;
        if (line[0] != '#' && sscanf(line, "0x%lx %lu", &addr, &ts) == 2 && n < max) {
            out[n].addr = addr; out[n].ts = ts; n++;
        }
        line = nl ? nl + 1 : NULL;
    }
    return n;
}

static int read_dump(entry_t *out, int max)
{
    FILE *f = fopen(PROCFS_DUMP, "r");
    if (!f) return -errno;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (n < max && sscanf(line, "0x%lx %lu", &out[n].addr, &out[n].ts) == 2) n++;
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
        for (int i = 0; i < n; i++) if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

/* Background thread: write phase-B pages on the GPU. */
static write_args_t g_write_args;
static void *gpu_write_thread(void *arg)
{
    write_args_t *wa = (write_args_t *)arg;
    gpu_write_range<<<1, 32>>>(wa->base, wa->start_page, wa->end_page, wa->tag);
    cudaDeviceSynchronize();
    return NULL;
}

int main(void)
{
    printf("[tc03] concurrent_insert_during_dump: A=%d B=%d pages\n",
           PHASE_PAGES, PHASE_PAGES);

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, TOTAL_PAGES * PAGE_SIZE));
    memset(managed, 0, TOTAL_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long base = (unsigned long)managed;
    printf("[tc03] pid=%d alloc=0x%lx\n", getpid(), base);

    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03] start failed: %s\n", strerror(-rc)); return 1; }

    gpu_write_range<<<1, 32>>>(managed, 0, PHASE_PAGES, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover1 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    /* Spawn background GPU writer for phase-B (writes into live ds, not snapshot). */
    g_write_args.base       = managed;
    g_write_args.start_page = PHASE_PAGES;
    g_write_args.end_page   = TOTAL_PAGES;
    g_write_args.tag        = 2;

    pthread_t writer;
    pthread_create(&writer, NULL, gpu_write_thread, &g_write_args);

    /* Foreground: read dump slowly (concurrent with GPU inserts). */
    entry_t snap1[MAX_ENTRIES];
    int n1 = read_dump_slow(snap1, MAX_ENTRIES);

    pthread_join(writer, NULL);

    if (n1 < 0) { fprintf(stderr, "[tc03] dump1 failed: %s\n", strerror(-n1)); stop_track(); return 1; }

    int a_in_1 = count_in_range(snap1, n1, base, 0, PHASE_PAGES);
    int b_in_1 = count_in_range(snap1, n1, base, PHASE_PAGES, TOTAL_PAGES);
    printf("[tc03] dump1 (concurrent with inserts): n=%d A=%d/%d B=%d (want A=%d B=0)\n",
           n1, a_in_1, PHASE_PAGES, b_in_1, PHASE_PAGES);

    /* Drain epoch 2. */
    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03] cutover2 failed: %s\n", strerror(-rc)); stop_track(); return 1; }

    entry_t snap2[MAX_ENTRIES];
    int n2 = read_dump(snap2, MAX_ENTRIES);
    if (n2 < 0) { fprintf(stderr, "[tc03] dump2 failed: %s\n", strerror(-n2)); stop_track(); return 1; }

    int b_in_2 = count_in_range(snap2, n2, base, PHASE_PAGES, TOTAL_PAGES);
    int a_in_2 = count_in_range(snap2, n2, base, 0, PHASE_PAGES);
    printf("[tc03] dump2: n=%d B=%d/%d A=%d (want B=%d A=0)\n",
           n2, b_in_2, PHASE_PAGES, a_in_2, PHASE_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (a_in_1 != PHASE_PAGES || b_in_1 != 0 || b_in_2 != PHASE_PAGES || a_in_2 != 0);
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    if (a_in_1 != PHASE_PAGES) printf("[tc03]   dump1: only %d/%d phase-A pages\n", a_in_1, PHASE_PAGES);
    if (b_in_1 != 0)           printf("[tc03]   dump1: %d concurrent-insert pages in snapshot\n", b_in_1);
    if (b_in_2 != PHASE_PAGES) printf("[tc03]   dump2: only %d/%d phase-B pages\n", b_in_2, PHASE_PAGES);
    if (a_in_2 != 0)           printf("[tc03]   dump2: %d phase-A pages leaked\n", a_in_2);
    return failed;
}
