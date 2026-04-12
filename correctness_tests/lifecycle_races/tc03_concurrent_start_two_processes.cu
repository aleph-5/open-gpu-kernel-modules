/*
 * tc03_concurrent_start_two_processes.cu
 *
 * Two processes (parent and child via fork) each independently start their
 * own tracking session and write distinct pages. Each process reads its own
 * dump. The tracking is per-pid, so each process must see only its own pages
 * and the sessions must not interfere.
 *
 * Flow:
 *   parent allocates two non-overlapping managed arrays (one for parent, one for child)
 *   fork()
 *   parent: start(delta) → write parent_pages → cutover → dump → verify parent pages only
 *   child:  start(delta) → write child_pages  → cutover → dump → verify child pages only
 *   parent waits for child; both PASS → overall PASS
 *
 * Note: each process runs cudaInit implicitly. The child inherits the CUDA
 * context initialization but the managed allocations are shared memory; each
 * process writes to its own allocation range.
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

#define NUM_PAGES   8
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

static int count_in_range(entry_t *e, int n, unsigned long base, int num_pages)
{
    int found = 0;
    for (int p = 0; p < num_pages; p++) {
        unsigned long pa = base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { found++; break; }
    }
    return found;
}

int main(void)
{
    printf("[tc03] concurrent_start_two_processes\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    /* Allocate two regions before fork so both processes can reference them. */
    int *parent_mem = NULL, *child_mem = NULL;
    CUDA_CHECK(cudaMallocManaged(&parent_mem, NUM_PAGES * PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged(&child_mem,  NUM_PAGES * PAGE_SIZE));
    memset(parent_mem, 0, NUM_PAGES * PAGE_SIZE);
    memset(child_mem,  0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long parent_base = (unsigned long)parent_mem;
    unsigned long child_base  = (unsigned long)child_mem;
    printf("[tc03] parent_base=0x%lx child_base=0x%lx\n", parent_base, child_base);

    pid_t child_pid = fork();
    if (child_pid < 0) {
        perror("fork");
        CUDA_CHECK(cudaFree(parent_mem));
        CUDA_CHECK(cudaFree(child_mem));
        return 1;
    }

    if (child_pid == 0) {
        /* Child process. */
        int rc = start_track_delta();
        if (rc) { fprintf(stderr, "[tc03/child] start failed: %s\n", strerror(-rc)); exit(1); }

        gpu_write_range<<<1, 32>>>(child_mem, NUM_PAGES, 200);
        cudaDeviceSynchronize();

        rc = cutover();
        if (rc) { fprintf(stderr, "[tc03/child] cutover failed: %s\n", strerror(-rc)); stop_track(); exit(1); }

        entry_t e[MAX_ENTRIES];
        int n = read_dump(e, MAX_ENTRIES);
        if (n < 0) { fprintf(stderr, "[tc03/child] dump failed: %s\n", strerror(-n)); stop_track(); exit(1); }

        int child_found  = count_in_range(e, n, child_base,  NUM_PAGES);
        int parent_found = count_in_range(e, n, parent_base, NUM_PAGES);
        printf("[tc03/child] dump: n=%d child_pages=%d/%d parent_pages=%d (want %d,0)\n",
               n, child_found, NUM_PAGES, parent_found, NUM_PAGES);

        stop_track();
        int fail = (child_found != NUM_PAGES || parent_found != 0);
        printf("[tc03/child] %s\n", fail ? "FAIL" : "PASS");
        exit(fail);
    }

    /* Parent process. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc03/parent] start failed: %s\n", strerror(-rc)); waitpid(child_pid, NULL, 0); return 1; }

    gpu_write_range<<<1, 32>>>(parent_mem, NUM_PAGES, 100);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc03/parent] cutover failed: %s\n", strerror(-rc)); stop_track(); waitpid(child_pid, NULL, 0); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc03/parent] dump failed: %s\n", strerror(-n)); stop_track(); waitpid(child_pid, NULL, 0); return 1; }

    int parent_found = count_in_range(e, n, parent_base, NUM_PAGES);
    int child_found  = count_in_range(e, n, child_base,  NUM_PAGES);
    printf("[tc03/parent] dump: n=%d parent_pages=%d/%d child_pages=%d (want %d,0)\n",
           n, parent_found, NUM_PAGES, child_found, NUM_PAGES);

    stop_track();

    int child_status = 0;
    waitpid(child_pid, &child_status, 0);
    int child_exit = WIFEXITED(child_status) ? WEXITSTATUS(child_status) : 1;

    CUDA_CHECK(cudaFree(parent_mem));
    CUDA_CHECK(cudaFree(child_mem));

    int parent_fail = (parent_found != NUM_PAGES || child_found != 0);
    int failed = parent_fail || child_exit;
    printf("[tc03] parent %s, child %s\n",
           parent_fail ? "FAIL" : "PASS", child_exit ? "FAIL" : "PASS");
    printf("[tc03] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
