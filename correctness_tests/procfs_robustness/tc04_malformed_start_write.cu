/*
 * tc04_malformed_start_write.cu
 *
 * Writing malformed or invalid values to dirty_tracking_start must return
 * errors (EINVAL or similar) and must not corrupt driver state.
 *
 * Test cases for malformed input:
 *   1. Empty string
 *   2. Non-numeric PID ("abc delta")
 *   3. Valid PID but missing mode ("12345")  [old API format]
 *   4. Valid PID + invalid mode ("12345 blah")
 *   5. Valid PID + valid mode but with extra garbage ("12345 delta EXTRA")
 *
 * After all bad writes, a valid start+write+cutover+dump session must work.
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

__global__ void gpu_write_page(int *page) { page[0] = 42; }

static int procfs_write_buf(const char *path, const char *val, size_t len)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -errno;
    ssize_t n = write(fd, val, len);
    int saved = errno;
    close(fd);
    if (n < 0) return -saved;
    return 0;
}

static int procfs_write_exact(const char *path, const char *val)
{
    return procfs_write_buf(path, val, strlen(val));
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

static void check_bad_write(const char *desc, const char *val, size_t len)
{
    int rc = procfs_write_buf(PROCFS_START, val, len);
    printf("[tc04]   %-40s rc=%-6d %s\n", desc,
           rc, rc ? "(rejected, good)" : "(accepted unexpectedly — FAIL)");
}

int main(void)
{
    printf("[tc04] malformed_start_write (procfs robustness)\n");

    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc04] pid=%d\n", getpid());
    printf("[tc04] bad-input probes (all should be rejected):\n");

    int all_rejected = 1;

    /* 1. Empty string. */
    {
        int rc = procfs_write_buf(PROCFS_START, "", 0);
        printf("[tc04]   %-40s rc=%-6d %s\n", "empty string", rc,
               rc ? "(rejected, good)" : "(accepted unexpectedly — FAIL)");
        if (rc == 0) all_rejected = 0;
    }

    /* 2. Non-numeric PID. */
    {
        int rc = procfs_write_exact(PROCFS_START, "abc delta");
        printf("[tc04]   %-40s rc=%-6d %s\n", "\"abc delta\"", rc,
               rc ? "(rejected, good)" : "(accepted unexpectedly — FAIL)");
        if (rc == 0) all_rejected = 0;
    }

    /* 3. Valid PID but missing mode (old API format). */
    {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", getpid());
        int rc = procfs_write_exact(PROCFS_START, buf);
        printf("[tc04]   %-40s rc=%-6d %s\n", "\"<pid>\" only (no mode)", rc,
               rc ? "(rejected, good)" : "(accepted unexpectedly — FAIL)");
        if (rc == 0) { all_rejected = 0; stop_track(); } /* clean up if it somehow started */
    }

    /* 4. Valid PID + invalid mode. */
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d blah", getpid());
        int rc = procfs_write_exact(PROCFS_START, buf);
        printf("[tc04]   %-40s rc=%-6d %s\n", "\"<pid> blah\"", rc,
               rc ? "(rejected, good)" : "(accepted unexpectedly — FAIL)");
        if (rc == 0) { all_rejected = 0; stop_track(); }
    }

    /* 5. Valid PID + valid mode + extra garbage. */
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d delta EXTRA_GARBAGE", getpid());
        int rc = procfs_write_exact(PROCFS_START, buf);
        printf("[tc04]   %-40s rc=%-6d %s\n", "\"<pid> delta EXTRA\"", rc,
               rc ? "(rejected, good)" : "(accepted unexpectedly — FAIL)");
        if (rc == 0) { all_rejected = 0; stop_track(); }
    }

    /* After all bad writes, driver must still work. */
    int rc = start_track_delta();
    if (rc) { fprintf(stderr, "[tc04] valid start failed after bad writes: %s\n", strerror(-rc)); CUDA_CHECK(cudaFree(managed)); return 1; }

    gpu_write_page<<<1, 1>>>(managed);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();
    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    int found = 0;
    unsigned long pa = (unsigned long)managed;
    for (int i = 0; i < n; i++) if (e[i].addr == pa) { found = 1; break; }
    printf("[tc04] valid session after bad writes: n=%d found=%d (want found=1)\n", n < 0 ? -1 : n, found);
    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (!all_rejected || !found);
    printf("[tc04] %s\n", failed ? "FAIL" : "PASS");
    if (!all_rejected) printf("[tc04]   one or more bad inputs were accepted\n");
    if (!found)        printf("[tc04]   valid session broken after bad writes\n");
    return failed;
}
