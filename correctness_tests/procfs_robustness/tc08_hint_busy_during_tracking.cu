/*
 * tc08_hint_busy_during_tracking.cu
 *
 * The dirty_tracking_hint procfs file selects which dirty-ds backend ops table
 * is used at init time. Swapping the ops table while a backend is initialized
 * would orphan the data already accumulated, so the kernel must reject the
 * write with -EBUSY whenever any of the ds slots (live / snapshot / cumulative)
 * is non-empty.
 *
 * Flow:
 *   1. Write a valid hint (WRITE_SEQ) before tracking → must succeed.
 *   2. Start tracking.
 *   3. Attempt to switch hint to WRITE_RAND → must fail with -EBUSY.
 *   4. Attempt to switch hint to WRITE_SEQ (same as current) → must also
 *      fail with -EBUSY (the check is structural, not value-based).
 *   5. Stop tracking.
 *   6. After stop, switching is allowed again → write WRITE_RAND succeeds.
 *   7. Cleanup: restore hint to WRITE_SEQ.
 */

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROCFS_START "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define PROCFS_STOP  "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define PROCFS_HINT  "/proc/driver/nvidia-uvm/dirty_tracking_hint"

#define PAGE_SIZE 4096

#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

__global__ void gpu_write_page(int *page) { page[0] = 1; }

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

int main(void)
{
    printf("[tc08] hint_busy_during_tracking (procfs robustness)\n");
    if (geteuid() != 0) { fprintf(stderr, "ERROR: must run as root\n"); return 1; }

    int *managed = NULL;
    CUDA_CHECK(cudaMallocManaged(&managed, PAGE_SIZE));
    memset(managed, 0, PAGE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    int failed = 0;

    /* 1. Valid hint write before tracking → must succeed. */
    int rc = procfs_write_exact(PROCFS_HINT, "WRITE_SEQ");
    printf("[tc08]   pre-start hint=WRITE_SEQ            rc=%d %s\n",
           rc, rc == 0 ? "(accepted, good)" : "(FAIL: should accept)");
    if (rc != 0) failed = 1;

    /* 2. Start tracking. */
    rc = start_track_delta();
    if (rc) {
        fprintf(stderr, "[tc08] start failed: %s\n", strerror(-rc));
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }

    /* Touch a page so the live ds is non-empty too (defensive — the EBUSY
     * check fires on priv != NULL, which is true the moment start completes). */
    gpu_write_page<<<1, 1>>>(managed);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 3. Switch hint to WRITE_RAND while tracking → must fail with -EBUSY. */
    rc = procfs_write_exact(PROCFS_HINT, "WRITE_RAND");
    int busy_rand = (rc == -EBUSY);
    printf("[tc08]   mid-tracking hint=WRITE_RAND        rc=%-6d %s\n",
           rc, busy_rand ? "(rejected EBUSY, good)"
                         : (rc ? "(rejected, but not EBUSY — FAIL)"
                               : "(accepted — FAIL)"));
    if (!busy_rand) failed = 1;

    /* 4. Switch to same value (WRITE_SEQ) → must also fail with -EBUSY. */
    rc = procfs_write_exact(PROCFS_HINT, "WRITE_SEQ");
    int busy_seq = (rc == -EBUSY);
    printf("[tc08]   mid-tracking hint=WRITE_SEQ (same)  rc=%-6d %s\n",
           rc, busy_seq ? "(rejected EBUSY, good)"
                        : (rc ? "(rejected, but not EBUSY — FAIL)"
                              : "(accepted — FAIL)"));
    if (!busy_seq) failed = 1;

    /* 5. Stop tracking. */
    rc = stop_track();
    if (rc) {
        fprintf(stderr, "[tc08] stop failed: %s\n", strerror(-rc));
        CUDA_CHECK(cudaFree(managed));
        return 1;
    }

    /* 6. Switch is allowed again after stop. */
    rc = procfs_write_exact(PROCFS_HINT, "WRITE_RAND");
    printf("[tc08]   post-stop  hint=WRITE_RAND          rc=%-6d %s\n",
           rc, rc == 0 ? "(accepted, good)" : "(FAIL: should accept)");
    if (rc != 0) failed = 1;

    /* 7. Restore default so the next test in the suite starts clean. */
    procfs_write_exact(PROCFS_HINT, "WRITE_SEQ");

    CUDA_CHECK(cudaFree(managed));
    printf("[tc08] %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
