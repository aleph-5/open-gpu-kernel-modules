/*
 * tracking_preload.c
 *
 * LD_PRELOAD shim that starts dirty tracking at the right moment:
 * AFTER the CUDA context is open (so the UVM va_space exists in the kernel),
 * but BEFORE the benchmark's first kernel launch.
 *
 * Why not a constructor?
 *   __attribute__((constructor)) fires before main(), before any CUDA API call.
 *   The UVM driver's procfs start handler calls uvm_dirty_invalidate_fn() which
 *   needs a va_space for the given PID.  If no va_space exists yet the write
 *   returns -EFAULT and tracking is never started — silent failure.
 *
 * Fix: intercept cudaMallocManaged.  By the time that call returns the CUDA
 * context (and its UVM va_space) is guaranteed to exist.  We start tracking
 * exactly once, on the first successful cudaMallocManaged.  dt_start blocks
 * until the kernel barrier completes (all currently GPU-mapped pages
 * downgraded to RO), so when control returns to the benchmark, the
 * downgrade is already done.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -ldl -o tracking_preload.so tracking_preload.c
 *
 * Use:
 *   LD_PRELOAD=./tracking_preload.so ./benchmark [args]
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DT_PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define DT_PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define DT_PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define DT_PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

/* Dump directory — one file per run, named by PID. */
#define DUMP_DIR "/tmp/tracking_preload_dumps"

static pthread_once_t g_start_once = PTHREAD_ONCE_INIT;
static int g_tracking_started = 0;

static void procfs_write_or_warn(const char *path, const char *buf)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        fprintf(stderr, "[tracking_preload] open %s: %s\n", path, strerror(errno));
        return;
    }
    if (write(fd, buf, strlen(buf)) < 0)
        fprintf(stderr, "[tracking_preload] write %s: %s\n", path, strerror(errno));
    else
        fprintf(stderr, "[tracking_preload] wrote '%s' -> %s\n", buf, path);
    close(fd);
}

/*
 * Issue cutover then read and print the dump.
 * Prints up to DUMP_PRINT_MAX individual page lines, then a total count.
 */
static void do_query_dump(void)
{
    char pid_buf[16];
    snprintf(pid_buf, sizeof(pid_buf), "%d", (int)getpid());

    /* cutover: atomically snapshot the live table */
    int fd = open(DT_PROCFS_CUTOVER, O_WRONLY);
    if (fd < 0) {
        fprintf(stderr, "[tracking_preload] open cutover: %s\n", strerror(errno));
        return;
    }
    if (write(fd, pid_buf, strlen(pid_buf)) < 0) {
        fprintf(stderr, "[tracking_preload] cutover write failed: %s\n", strerror(errno));
        close(fd);
        return;
    }
    close(fd);

    /* dump: read the frozen snapshot */
    fd = open(DT_PROCFS_DUMP, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[tracking_preload] open dump: %s\n", strerror(errno));
        return;
    }

    /* Write dump to a per-PID file so it isn't lost in terminal noise. */
    mkdir(DUMP_DIR, 0755);
    char dump_path[128];
    snprintf(dump_path, sizeof(dump_path), DUMP_DIR "/pid_%d.txt", (int)getpid());
    FILE *out = fopen(dump_path, "w");
    if (!out) {
        fprintf(stderr, "[tracking_preload] cannot open dump file %s: %s\n",
                dump_path, strerror(errno));
        close(fd);
        return;
    }

    char chunk[65536];
    long page_count = 0;
    ssize_t n;
    /* Count page lines (those starting with '0x') while copying to file. */
    char line[256];
    int line_pos = 0;

    while ((n = read(fd, chunk, sizeof(chunk))) > 0) {
        fwrite(chunk, 1, n, out);
        for (ssize_t i = 0; i < n; i++) {
            char c = chunk[i];
            if (c == '\n' || line_pos == (int)sizeof(line) - 1) {
                line[line_pos] = '\0';
                line_pos = 0;
                if (line[0] == '0')
                    page_count++;
            } else {
                line[line_pos++] = c;
            }
        }
    }
    if (line_pos > 0 && line[0] == '0')
        page_count++;

    fclose(out);
    close(fd);

    fprintf(stderr, "[tracking_preload] dirty pages recorded: %ld  (full dump -> %s)\n",
            page_count, dump_path);
}

static void do_start_tracking(void)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d delta", (int)getpid());
    /* This write blocks until the kernel barrier (downgrade) completes. */
    int fd = open(DT_PROCFS_START, O_WRONLY);
    if (fd < 0) {
        fprintf(stderr, "[tracking_preload] open %s: %s\n", DT_PROCFS_START, strerror(errno));
        return;
    }
    ssize_t n = write(fd, buf, strlen(buf));
    close(fd);
    if (n < 0) {
        fprintf(stderr, "[tracking_preload] dt_start failed: %s\n", strerror(errno));
        return;
    }
    fprintf(stderr, "[tracking_preload] tracking started (pid=%d), barrier done\n", (int)getpid());
    g_tracking_started = 1;
}

/* Intercept cudaMallocManaged — first call after this returns means va_space exists.
 * cudaError_t is an int enum; cudaSuccess == 0. Avoid depending on CUDA headers. */
int cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    typedef int (*fn_t)(void **, size_t, unsigned int);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cudaMallocManaged");

    int err = real_fn(devPtr, size, flags);

    if (err == 0 /* cudaSuccess */)
        pthread_once(&g_start_once, do_start_tracking);

    return err;
}

__attribute__((destructor))
static void tracking_stop(void)
{
    if (!g_tracking_started)
        return;
    do_query_dump();
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", (int)getpid());
    procfs_write_or_warn(DT_PROCFS_STOP, buf);
}
