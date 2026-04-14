#ifndef PERFORMANCE_TESTS_DIRTY_TRACKING_PROCFS_H
#define PERFORMANCE_TESTS_DIRTY_TRACKING_PROCFS_H

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DT_PROCFS_START   "/proc/driver/nvidia-uvm/dirty_tracking_start"
#define DT_PROCFS_STOP    "/proc/driver/nvidia-uvm/dirty_tracking_stop"
#define DT_PROCFS_CUTOVER "/proc/driver/nvidia-uvm/dirty_tracking_query_cutover"
#define DT_PROCFS_DUMP    "/proc/driver/nvidia-uvm/dirty_tracking_query_dump"

static inline int dt_sysfs_exists(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

// Returns 0 on success, or -errno on failure.
static inline int dt_procfs_write_exact(const char *path, const char *val)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0)
        return -errno;

    ssize_t n = write(fd, val, strlen(val));
    int saved = errno;
    close(fd);

    if (n < 0)
        return -saved;
    return 0;
}

static inline void dt_procfs_write_or_die(const char *path, const char *val)
{
    int rc = dt_procfs_write_exact(path, val);
    if (rc) {
        fprintf(stderr, "procfs write failed: %s: %s\n", path, strerror(-rc));
        exit(1);
    }
}

// Start dirty tracking for this process (mode: "delta" or "cumulative").
static inline int dt_start(const char *mode)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%d %s", getpid(), mode);
    return dt_procfs_write_exact(DT_PROCFS_START, buf);
}

static inline int dt_stop(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return dt_procfs_write_exact(DT_PROCFS_STOP, buf);
}

static inline int dt_cutover(void)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", getpid());
    return dt_procfs_write_exact(DT_PROCFS_CUTOVER, buf);
}

// Reads DT_PROCFS_DUMP and counts entries whose address falls within
// [base, base+size_bytes), and is aligned to page_size.
//
// Returns:
//   >=0 : count of matching entries
//   <0  : -errno on failure
static inline int dt_dump_count_pages_in_range(unsigned long base,
                                               size_t size_bytes,
                                               unsigned long page_size)
{
    FILE *f = fopen(DT_PROCFS_DUMP, "r");
    if (!f)
        return -errno;

    const unsigned long end = base + (unsigned long)size_bytes;
    int count = 0;
    char line[256];
    unsigned long addr = 0;
    unsigned long ts = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#')
            continue;
        if (sscanf(line, "0x%lx %lu", &addr, &ts) != 2)
            continue;
        if (addr >= base && addr < end && ((addr - base) % page_size) == 0)
            count++;
    }

    if (ferror(f)) {
        int saved = errno;
        fclose(f);
        return -saved;
    }

    fclose(f);
    return count;
}

#endif // PERFORMANCE_TESTS_DIRTY_TRACKING_PROCFS_H
