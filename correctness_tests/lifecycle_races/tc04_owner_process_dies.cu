/*
 * tc04_owner_process_dies.cu
 *
 * Safety test: the process that owns a tracking session dies (SIGKILL) while
 * tracking is active. A new process must be able to start its own tracking
 * session afterwards without errors or stale state from the dead owner.
 *
 * Flow:
 *   fork()
 *   child: start(delta) → write pages → SIGKILL self (no stop)
 *   parent: wait for child to die
 *   parent: start(delta) → write own pages → cutover → dump
 *   PASS: dump contains parent pages; no crash or EBUSY from child's stale state.
 *
 * The driver must clean up a session when the owning process's VA space is
 * destroyed (process exit / SIGKILL).
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
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

int main(void)
{
    printf("[tc04] owner_process_dies (no stop before exit)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *parent_mem = NULL;
    CUDA_CHECK(cudaMallocManaged(&parent_mem, NUM_PAGES * PAGE_SIZE));
    memset(parent_mem, 0, NUM_PAGES * PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long parent_base = (unsigned long)parent_mem;
    printf("[tc04] pid=%d parent_base=0x%lx\n", getpid(), parent_base);

    pid_t child_pid = fork();
    if (child_pid < 0) { perror("fork"); CUDA_CHECK(cudaFree(parent_mem)); return 1; }

    if (child_pid == 0) {
        /* Child: start tracking, write pages, then die without stopping. */
        int *child_mem = NULL;
        cudaError_t ce = cudaMallocManaged(&child_mem, NUM_PAGES * PAGE_SIZE);
        if (ce != cudaSuccess) { fprintf(stderr, "[tc04/child] alloc failed\n"); exit(1); }
        memset(child_mem, 0, NUM_PAGES * PAGE_SIZE);
        cudaDeviceSynchronize();

        int rc = start_track_delta();
        if (rc) { fprintf(stderr, "[tc04/child] start failed: %s\n", strerror(-rc)); exit(1); }

        gpu_write_range<<<1, 32>>>(child_mem, NUM_PAGES, 200);
        cudaDeviceSynchronize();

        printf("[tc04/child] tracking active, sending SIGKILL to self (pid=%d)\n", getpid());
        fflush(stdout);
        /* Die without calling stop_track or cudaFree. */
        kill(getpid(), SIGKILL);
        /* Should not reach here. */
        exit(0);
    }

    /* Parent waits for child to die. */
    int status;
    waitpid(child_pid, &status, 0);
    if (WIFSIGNALED(status))
        printf("[tc04] child killed by signal %d (expected)\n", WTERMSIG(status));
    else
        printf("[tc04] child exited with code %d\n", WEXITSTATUS(status));

    /* Small pause to allow driver cleanup of dead process. */
    usleep(100000); /* 100ms */

    /* Parent starts its own session — must not get EBUSY or other error from
     * the child's stale state. */
    int rc = start_track_delta();
    if (rc) {
        fprintf(stderr, "[tc04] parent start failed after child death: %s\n", strerror(-rc));
        CUDA_CHECK(cudaFree(parent_mem));
        return 1;
    }

    gpu_write_range<<<1, 32>>>(parent_mem, NUM_PAGES, 100);
    CUDA_CHECK(cudaDeviceSynchronize());

    rc = cutover();
    if (rc) { fprintf(stderr, "[tc04] cutover failed: %s\n", strerror(-rc)); stop_track(); CUDA_CHECK(cudaFree(parent_mem)); return 1; }

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    if (n < 0) { fprintf(stderr, "[tc04] dump failed: %s\n", strerror(-n)); stop_track(); CUDA_CHECK(cudaFree(parent_mem)); return 1; }

    int parent_found = 0;
    for (int p = 0; p < NUM_PAGES; p++) {
        unsigned long pa = parent_base + (unsigned long)p * PAGE_SIZE;
        for (int i = 0; i < n; i++)
            if (e[i].addr == pa) { parent_found++; break; }
    }
    printf("[tc04] parent dump: n=%d parent_pages=%d/%d (want %d)\n",
           n, parent_found, NUM_PAGES, NUM_PAGES);

    stop_track();
    CUDA_CHECK(cudaFree(parent_mem));

    int failed = (parent_found != NUM_PAGES);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (parent_found != NUM_PAGES)
        printf("[tc04]   only %d/%d parent pages recorded after child death\n",
               parent_found, NUM_PAGES);
    return failed;
}
