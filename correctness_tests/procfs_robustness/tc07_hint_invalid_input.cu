/*
 * tc07_hint_invalid_input.cu
 *
 * Writing malformed or unrecognized values to dirty_tracking_hint must be
 * rejected with -EINVAL and must not corrupt driver state. Only the exact
 * tokens "WRITE_SEQ" and "WRITE_RAND" (with optional trailing whitespace) are
 * accepted.
 *
 * Probes (all expected to fail):
 *   1. Empty string
 *   2. Lowercase ("write_seq")
 *   3. Garbage ("FOO")
 *   4. Partial prefix ("WRITE_")
 *   5. Two valid tokens ("WRITE_SEQ WRITE_RAND")
 *
 * After all bad writes, a full start+write+cutover+dump session must work,
 * confirming the procfs hint handler did not wedge the driver.
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

#define PAGE_SIZE   4096
#define MAX_ENTRIES 4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
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

int main(void)
{
    printf("[tc07] hint_invalid_input (procfs robustness)\n");
    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[tc07] pid=%d\n", getpid());
    printf("[tc07] bad-input probes (all should be rejected with -EINVAL):\n");

    int all_rejected = 1;

    struct { const char *desc; const char *val; size_t len; } probes[] = {
        { "empty string",            "",                       0  },
        { "\"write_seq\" (lowercase)", "write_seq",             0  },
        { "\"FOO\"",                  "FOO",                    0  },
        { "\"WRITE_\" (prefix only)", "WRITE_",                 0  },
        { "\"WRITE_SEQ WRITE_RAND\"", "WRITE_SEQ WRITE_RAND",   0  },
    };
    int nprobes = sizeof(probes) / sizeof(probes[0]);
    for (int i = 0; i < nprobes; i++) {
        size_t len = probes[i].len ? probes[i].len : strlen(probes[i].val);
        int rc = procfs_write_buf(PROCFS_HINT, probes[i].val, len);
        int rejected = (rc == -EINVAL);
        printf("[tc07]   %-30s rc=%-6d %s\n", probes[i].desc, rc,
               rejected ? "(rejected with EINVAL, good)"
                        : (rc ? "(rejected, but not EINVAL)"
                              : "(accepted unexpectedly — FAIL)"));
        if (!rejected) all_rejected = 0;
    }

    /* After bad writes, driver must still work. Pin to a known-good hint. */
    int rc = procfs_write_exact(PROCFS_HINT, "WRITE_SEQ");
    if (rc) {
        fprintf(stderr, "[tc07] valid hint write failed after bad writes: %s\n", strerror(-rc));
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }

    rc = start_track_delta();
    if (rc) {
        fprintf(stderr, "[tc07] valid start failed after bad writes: %s\n", strerror(-rc));
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }

    gpu_write_page<<<1, 1>>>(managed);
    CUDA_CHECK(cudaDeviceSynchronize());
    cutover();

    entry_t e[MAX_ENTRIES];
    int n = read_dump(e, MAX_ENTRIES);
    int found = 0;
    unsigned long pa = (unsigned long)managed;
    for (int i = 0; i < n; i++) if (e[i].addr == pa) { found = 1; break; }
    printf("[tc07] valid session after bad writes: n=%d found=%d (want found=1)\n",
           n < 0 ? -1 : n, found);
    stop_track();
    CUDA_CHECK(cudaFree(managed));

    int failed = (!all_rejected || !found);
    printf("[tc07] %s\n", failed ? "FAIL" : "PASS");
    if (!all_rejected) printf("[tc07]   one or more bad inputs were accepted (or rejected with wrong errno)\n");
    if (!found)        printf("[tc07]   valid session broken after bad writes\n");
    return failed;
}
