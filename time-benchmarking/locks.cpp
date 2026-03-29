// locks.cpp
// Reads lock timing stats from /proc/nvidia-uvm/lock_stats and prints a
// breakdown of how much of the write-fault handling time is spent waiting
// on pid_to_page_table_lock.
//
// Usage:
//   # (optional) reset counters before running your workload
//   echo reset > /proc/nvidia-uvm/lock_stats_reset
//
//   # run your CUDA workload here
//
//   # then report
//   ./locks

#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char *STATS_PATH = "/proc/driver/nvidia-uvm/lock_stats";
static const char *RESET_PATH = "/proc/driver/nvidia-uvm/lock_stats_reset";

static void reset_counters() {
    FILE *f = fopen(RESET_PATH, "w");
    if (!f) {
        perror("open lock_stats_reset");
        exit(1);
    }
    fprintf(f, "reset\n");
    fclose(f);
    printf("Counters reset.\n");
}

static void print_stats() {
    FILE *f = fopen(STATS_PATH, "r");
    if (!f) {
        perror("open lock_stats");
        fprintf(stderr, "Is the nvidia-uvm module loaded?\n");
        exit(1);
    }

    long long fault_count       = 0;
    long long lock_wait_ns      = 0;
    long long total_fn_ns       = 0;
    long long lock_overhead_pct = 0;

    char key[64];
    long long val;
    while (fscanf(f, "%63s %lld\n", key, &val) == 2) {
        if      (strcmp(key, "fault_count")       == 0) fault_count       = val;
        else if (strcmp(key, "lock_wait_ns")      == 0) lock_wait_ns      = val;
        else if (strcmp(key, "total_fn_ns")       == 0) total_fn_ns       = val;
        else if (strcmp(key, "lock_overhead_pct") == 0) lock_overhead_pct = val;
    }
    fclose(f);

    long long work_ns = total_fn_ns - lock_wait_ns;

    printf("\n");
    printf("=== pid_to_page_table_lock overhead (uvm_dirty_page_table_record) ===\n");
    printf("\n");
    printf("  Write faults recorded  : %lld\n", fault_count);
    printf("\n");
    printf("  Total fn time          : %10.3f ms  (%.1f ns/fault)\n",
           total_fn_ns   / 1e6,
           fault_count > 0 ? (double)total_fn_ns   / fault_count : 0.0);
    printf("  Lock wait time         : %10.3f ms  (%.1f ns/fault)\n",
           lock_wait_ns  / 1e6,
           fault_count > 0 ? (double)lock_wait_ns  / fault_count : 0.0);
    printf("  Work inside lock       : %10.3f ms  (%.1f ns/fault)\n",
           work_ns       / 1e6,
           fault_count > 0 ? (double)work_ns        / fault_count : 0.0);
    printf("\n");
    printf("  Lock overhead          : %lld%%\n", lock_overhead_pct);
    printf("\n");

    if (lock_overhead_pct > 50)
        printf("  >> Lock contention is the dominant cost. Consider a finer-grained\n"
               "     lock (e.g. per-pid mutex) or a lock-free structure (seqlock/RCU).\n");
    else
        printf("  >> Lock wait is not the dominant cost. The bottleneck is likely\n"
               "     inside the lock (kmalloc, xa_cmpxchg, xa_load).\n");

    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc == 2 && strcmp(argv[1], "--reset") == 0) {
        reset_counters();
        return 0;
    }
    print_stats();
    return 0;
}
