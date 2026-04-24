/*******************************************************************************
    Copyright (c) 2013-2023 NVIDIA Corporation

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.
*******************************************************************************/

#include "uvm_common.h"
// EDIT BY KUSHAGRA
#include "uvm_dirty_ds.h"
// END OF EDIT
#include "uvm_linux.h"
#include "uvm_forward_decl.h"

// EDIT BY ARUSH
#if defined(CONFIG_PROC_FS)
#include "uvm_procfs.h"
#include "nv-procfs.h"
#endif
// END OF EDIT

// EDIT BY ARUSH
// Registered from uvm_va_space.c at module init; NULL until then.
NV_STATUS (*uvm_dirty_invalidate_fn)(void) = NULL;
NV_STATUS (*uvm_dirty_activate_now_fn)(void) = NULL;
NV_STATUS (*uvm_dirty_barrier_end_fn)(void) = NULL;
// END OF EDIT

// EDIT BY ADITI KHANDELIA
NV_STATUS (*uvm_dirty_query_barrier_begin_fn)(void) = NULL;
NV_STATUS (*uvm_dirty_query_barrier_end_fn)(void) = NULL;
// END OF EDIT

// EDIT BY VIDHI JAIN
// called at stop to restore READ_WRITE or ATOMIC permission to pages we downgraded at start
void (*uvm_dirty_restore_fn)(void) = NULL;
// END OF EDIT

// EDIT BY ADITI KHANDELIA
// EDIT BY KUSHAGRA: replaced mutex with rwsem — hot-path ops (insert/lookup)
// take the read side (concurrent), lifecycle ops (init/destroy) take the write side.
static DECLARE_RWSEM(uvm_dirty_ds_rwsem);
static DEFINE_MUTEX(uvm_dirty_lifecycle_lock);
static bool g_uvm_dirty_tracking_active;

// EDIT BY KUSHAGRA
static struct uvm_dirty_ds g_dirty_ds = {
    .ops  = &uvm_dirty_ds_vector_ops, // VIDHI: Modified for Testing
    .priv = NULL,
};
// EDIT BY ADITI KHANDELIA
static struct uvm_dirty_ds g_dirty_ds_snapshot = {
    .ops  = &uvm_dirty_ds_vector_ops, 
    .priv = NULL,
};
static struct uvm_dirty_ds g_dirty_ds_cumulative = {
    .ops  = &uvm_dirty_ds_vector_ops, 
    .priv = NULL,
};
static unsigned int g_dirty_epoch = 0; 
static unsigned int g_dirty_snapshot_epoch = 0;
static bool g_snapshot_pending = false;
static pid_t g_dirty_owner_tgid = -1;
static enum uvm_dirty_tracking_querying_state g_dirty_tracking_querying_state = UVM_DIRTY_TRACKING_QUERY_DELTA;
// END OF EDIT
// END OF EDIT


bool uvm_dirty_tracking_started(void)
{ 
    down_read(&uvm_dirty_ds_rwsem); 
    bool started = g_dirty_ds.priv != NULL;
    up_read(&uvm_dirty_ds_rwsem);
    return started;
}

pid_t uvm_owner_tgid_with_dirty_tracking_started(void)
{
    pid_t owner_tgid;

    down_read(&uvm_dirty_ds_rwsem);
    owner_tgid = g_dirty_owner_tgid;
    up_read(&uvm_dirty_ds_rwsem);

    return owner_tgid;
}

bool uvm_dirty_tracking_active(void)
{
    return READ_ONCE(g_uvm_dirty_tracking_active);
}

NV_STATUS uvm_dirty_set_tracking_active(bool active)
{
    WRITE_ONCE(g_uvm_dirty_tracking_active, active);
    return NV_OK;
}

pid_t uvm_owner_tgid_with_dirty_tracking_active(void)
{
    pid_t owner_tgid;

    down_read(&uvm_dirty_ds_rwsem);
    if (!READ_ONCE(g_uvm_dirty_tracking_active)) {
        up_read(&uvm_dirty_ds_rwsem);
        return -1;
    }
    owner_tgid = g_dirty_owner_tgid;
    up_read(&uvm_dirty_ds_rwsem);

    return owner_tgid;
}

NV_STATUS uvm_dirty_page_table_init(pid_t pid)
{
    NV_STATUS status;

    down_write(&uvm_dirty_ds_rwsem);

    if (g_dirty_ds.priv != NULL) {
        up_write(&uvm_dirty_ds_rwsem);
        return NV_ERR_IN_USE;
    }

    status = uvm_dirty_ds_init(&g_dirty_ds);
    if (status != NV_OK) {
        up_write(&uvm_dirty_ds_rwsem);
        return status;
    }

    if (g_dirty_tracking_querying_state == UVM_DIRTY_TRACKING_QUERY_CUMULATIVE) {
        status = uvm_dirty_ds_init(&g_dirty_ds_cumulative);
        if (status != NV_OK) {
            uvm_dirty_ds_destroy(&g_dirty_ds);
            up_write(&uvm_dirty_ds_rwsem);
            return status;
        }
    }

    g_dirty_owner_tgid = pid;

    up_write(&uvm_dirty_ds_rwsem);
    return NV_OK;
}

NV_STATUS uvm_dirty_page_table_destroy(bool locked) {
    if (!locked) down_write(&uvm_dirty_ds_rwsem);

    WRITE_ONCE(g_uvm_dirty_tracking_active, false);

    if (g_dirty_ds.priv == NULL) {
        if (!locked) up_write(&uvm_dirty_ds_rwsem);
        return NV_ERR_GENERIC;
    }

    uvm_dirty_ds_destroy(&g_dirty_ds);

    if (g_dirty_ds_snapshot.priv != NULL) {
        uvm_dirty_ds_destroy(&g_dirty_ds_snapshot);
    }

    if (g_dirty_ds_cumulative.priv != NULL) {
        uvm_dirty_ds_destroy(&g_dirty_ds_cumulative);
    }

    g_dirty_tracking_querying_state = UVM_DIRTY_TRACKING_QUERY_DELTA;
    g_dirty_owner_tgid = -1;
    g_snapshot_pending = false;
    g_dirty_epoch = 0;
    g_dirty_snapshot_epoch = 0;

    if (!locked) up_write(&uvm_dirty_ds_rwsem);
    return NV_OK;
}

NV_STATUS uvm_dirty_page_table_record(unsigned long page_number,
    unsigned long timestamp) { // (EDIT BY VIDHI JAIN)
    NV_STATUS status;
    ktime_t t0; // EDIT BY VIDHI JAIN

    t0 = ktime_get(); // EDIT BY VIDHI JAIN
    down_read(&uvm_dirty_ds_rwsem);

    // EDIT BY VIDHI JAIN
    if (unlikely(g_dirty_ds.stats.enabled))
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)),
                     &g_dirty_ds.stats.insert_lock_wait_ns);

    if (g_dirty_ds.priv == NULL) {
        up_read(&uvm_dirty_ds_rwsem);
        return NV_ERR_NO_MEMORY;
    }

    status = uvm_dirty_ds_insert(&g_dirty_ds, page_number, timestamp);

    up_read(&uvm_dirty_ds_rwsem);
    return status;
}

// EDIT BY ADITI KHANDELIA
static NV_STATUS uvm_dirty_new_snapshot(struct uvm_dirty_ds* dest) {
    struct uvm_dirty_ds new_live = {
        .ops  = g_dirty_ds.ops,
        .priv = NULL,
        .stats = {0},
    };

    if (g_dirty_ds.priv == NULL) {
        return NV_ERR_GENERIC;
    }

    new_live.stats = g_dirty_ds.stats; // EDIT BY SANKALP MITTAL


    if (uvm_dirty_ds_init(&new_live) != NV_OK) {
        return NV_ERR_GENERIC;
    }

    *dest = new_live;
    return NV_OK;
}

struct uvm_dirty_merge_ctx {
    struct uvm_dirty_ds *dst;
    NV_STATUS status;
};

static void uvm_dirty_merge_snapshot_page(unsigned long page_num,
    unsigned long timestamp,
    void *ctx) {
    struct uvm_dirty_merge_ctx *merge_ctx = ctx;
    if (merge_ctx->status != NV_OK) {
        return;
    }
    merge_ctx->status = uvm_dirty_ds_insert(merge_ctx->dst, page_num, timestamp);
}

static NV_STATUS uvm_dirty_merge_snapshot_into_cumulative_locked(void) {
    struct uvm_dirty_merge_ctx ctx = {
        .dst = &g_dirty_ds_cumulative,
        .status = NV_OK,
    };

    if (g_dirty_ds_snapshot.priv == NULL) {
        return NV_OK;
    }

    if (g_dirty_ds_cumulative.priv == NULL) {
        return NV_ERR_GENERIC;
    }

    uvm_dirty_ds_for_each_in_range(&g_dirty_ds_snapshot,
        0,
        ~0UL,
        uvm_dirty_merge_snapshot_page,
        &ctx);

    return ctx.status;
}

NV_STATUS uvm_dirty_page_table_destroy_lifecycle_locked(void) {
    mutex_lock(&uvm_dirty_lifecycle_lock);
    down_write(&uvm_dirty_ds_rwsem);

    if (g_dirty_ds.priv == NULL) {
        up_write(&uvm_dirty_ds_rwsem);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return NV_ERR_GENERIC;
    }

    WRITE_ONCE(g_uvm_dirty_tracking_active, false);

    // Auto-finalize: promote the live dirty set into the snapshot so the dump
    // remains readable after the process exits (the query barrier path requires
    // the va_space to still be in the global list, which won't be true once the
    // caller returns from here).
    if (g_dirty_tracking_querying_state == UVM_DIRTY_TRACKING_QUERY_CUMULATIVE) {
        // Flush any pending snapshot into cumulative, then merge the live set.
        if (g_snapshot_pending)
            uvm_dirty_merge_snapshot_into_cumulative_locked();
        if (g_dirty_ds_snapshot.priv != NULL)
            uvm_dirty_ds_destroy(&g_dirty_ds_snapshot);

        struct uvm_dirty_merge_ctx live_ctx = {
            .dst    = &g_dirty_ds_cumulative,
            .status = NV_OK,
        };
        uvm_dirty_ds_for_each_in_range(&g_dirty_ds, 0, ~0UL,
                                        uvm_dirty_merge_snapshot_page, &live_ctx);

        // Move the fully-merged cumulative set into the snapshot slot so the
        // dump reader (which runs in DELTA mode after reset below) finds it.
        g_dirty_ds_snapshot = g_dirty_ds_cumulative;
        g_dirty_ds_cumulative.priv = NULL;
        uvm_dirty_ds_destroy(&g_dirty_ds);
    } else {
        // Delta mode: just hand the live set off to the snapshot slot.
        if (g_snapshot_pending && g_dirty_ds_snapshot.priv != NULL)
            uvm_dirty_ds_destroy(&g_dirty_ds_snapshot);
        g_dirty_ds_snapshot = g_dirty_ds;   // transfer ownership
    }

    g_dirty_ds.priv = NULL;
    g_snapshot_pending = true;

    // Reset all tracking state except the snapshot (consumed later by the dump).
    g_dirty_tracking_querying_state = UVM_DIRTY_TRACKING_QUERY_DELTA;
    g_dirty_owner_tgid = -1;
    g_dirty_epoch = 0;
    g_dirty_snapshot_epoch = 0;

    up_write(&uvm_dirty_ds_rwsem);
    mutex_unlock(&uvm_dirty_lifecycle_lock);
    return NV_OK;
}
// END OF EDIT

/* original lookup (no semaphore timing)
struct dirty_page_info* uvm_dirty_page_table_lookup(unsigned long page_number,
    bool locked) {
    struct dirty_page_info *info;

    if (!locked) down_read(&uvm_dirty_ds_rwsem);

    if (g_dirty_ds.priv == NULL) {
        if (!locked) up_read(&uvm_dirty_ds_rwsem);
        return NULL;
    }

    info = uvm_dirty_ds_lookup(&g_dirty_ds, page_number);

    if (!locked) up_read(&uvm_dirty_ds_rwsem);
    return info;
}
*/

// EDIT BY VIDHI JAIN
struct dirty_page_info* uvm_dirty_page_table_lookup(unsigned long page_number,
    bool locked) {
    struct dirty_page_info *info;
    ktime_t t0;

    if (!locked) {
        t0 = ktime_get();
        down_read(&uvm_dirty_ds_rwsem);
        if (unlikely(g_dirty_ds.stats.enabled))
            atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)),
                         &g_dirty_ds.stats.lookup_lock_wait_ns);
    }

    if (g_dirty_ds.priv == NULL) {
        if (!locked) up_read(&uvm_dirty_ds_rwsem);
        return NULL;
    }

    info = uvm_dirty_ds_lookup(&g_dirty_ds, page_number);

    if (!locked) up_read(&uvm_dirty_ds_rwsem);
    return info;
}
// END OF EDIT

// EDIT BY ARUSH - procfs query interface
// EDIT BY KUSHAGRA
static void procfs_print_page(unsigned long page_num, unsigned long timestamp,
                               void *ctx)
{
    seq_printf((struct seq_file *)ctx, "0x%lx %lu\n",
               page_num << PAGE_SHIFT, timestamp);
}
// END OF EDIT

// EDIT BY ADITI KHANDELIA
static int nv_procfs_read_dirty_tracking_query_dump(struct seq_file *s, 
    void *v) {

    mutex_lock(&uvm_dirty_lifecycle_lock);
    NV_STATUS status;
    bool ds_read_locked = false;
    int ret = 0;

    if (!g_snapshot_pending || g_dirty_ds_snapshot.priv == NULL) {
        ret = -EFAULT;
        goto out;
    }

    down_read(&uvm_dirty_ds_rwsem);
    ds_read_locked = true;

    if (g_dirty_tracking_querying_state == UVM_DIRTY_TRACKING_QUERY_CUMULATIVE) {
        status = uvm_dirty_merge_snapshot_into_cumulative_locked();
        if (status != NV_OK) {
            ret = -nv_status_to_errno(status);
            goto out;
        }
    }

    struct uvm_dirty_ds *query_ds =
        (g_dirty_tracking_querying_state == UVM_DIRTY_TRACKING_QUERY_CUMULATIVE) ?
            &g_dirty_ds_cumulative :
            &g_dirty_ds_snapshot;

    ktime_t t0 = ktime_get();
    if (unlikely(query_ds->stats.enabled)) {
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)), &query_ds->stats.for_each_lock_wait_ns);
    }

    if (query_ds->priv == NULL) {
        seq_printf(s, "# dirty tracking not active\n");
        goto out;
    }

    seq_printf(s, "# page_address_hex timestamp_ns\n");

    uvm_dirty_ds_for_each_in_range(query_ds, 0, ~0UL, procfs_print_page, s);
    up_read(&uvm_dirty_ds_rwsem);
    ds_read_locked = false;

    // EDIT BY KUSHAGRA
    /*
        Only consume the snapshot when the output fit in the seq_file buffer.
        If seq_has_overflowed(), seq_file will discard the output, grow the
        buffer, and call this function again.  Leaving g_snapshot_pending true
        and the snapshot intact lets that retry succeed.
    */
    if (!seq_has_overflowed(s)) {
        uvm_dirty_ds_destroy(&g_dirty_ds_snapshot);
        g_snapshot_pending = false;
    }
    // END OF EDIT
    goto out;

out:
    if (ds_read_locked) {
        up_read(&uvm_dirty_ds_rwsem);
    }
    mutex_unlock(&uvm_dirty_lifecycle_lock);
    return ret;
}

static int nv_procfs_read_dirty_tracking_query_dump_entry(struct seq_file *s, void *v)
{
    return nv_procfs_read_dirty_tracking_query_dump(s, v);
}

UVM_DEFINE_SINGLE_PROCFS_FILE(dirty_tracking_query_dump_entry);

static ssize_t dirty_tracking_query_cutover(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {

    if (!uvm_dirty_tracking_started()) {
        return -EFAULT;
    }

    char kbuf[16];
    size_t to_copy = min(count, sizeof(kbuf) - 1);
    if (copy_from_user(kbuf, buf, to_copy)) {
        return -EFAULT;
    }
    kbuf[to_copy] = '\0';
    pid_t target_pid = 0;
    if (sscanf(kbuf, "%d", &target_pid) != 1 || target_pid <= 0) {
        return -EINVAL;
    }
    if (target_pid != uvm_owner_tgid_with_dirty_tracking_started()) {
        return -ESRCH;
    }

    NV_STATUS status;
    bool query_barrier_held = false;
    bool ds_write_locked = false;
    int ret = 0;

    if (!uvm_dirty_query_barrier_begin_fn || !uvm_dirty_query_barrier_end_fn) {
        return -EFAULT;
    }

    mutex_lock(&uvm_dirty_lifecycle_lock);

    if (g_snapshot_pending) {
        ret = -EBUSY;
        goto out;
    }

    struct uvm_dirty_ds new_live;
    status = uvm_dirty_new_snapshot(&new_live);
    if (status != NV_OK) {
        ret = -nv_status_to_errno(status);
        goto out;
    }

    struct uvm_dirty_ds old_snapshot = g_dirty_ds_snapshot; 

    down_read(&uvm_dirty_ds_rwsem);
    if (g_dirty_ds.priv == NULL) {
        up_read(&uvm_dirty_ds_rwsem);
        uvm_dirty_ds_destroy(&new_live);
        ret = -EFAULT;
        goto out;
    }
    up_read(&uvm_dirty_ds_rwsem);

    status = uvm_dirty_query_barrier_begin_fn();
    if (status != NV_OK) {
        ret = -nv_status_to_errno(status);
        uvm_dirty_ds_destroy(&new_live);
        goto out;
    }
    query_barrier_held = true;

    down_write(&uvm_dirty_ds_rwsem);
    ds_write_locked = true;
    g_dirty_ds_snapshot = g_dirty_ds;
    g_dirty_ds = new_live;
    g_dirty_epoch += 1;
    g_dirty_snapshot_epoch = g_dirty_epoch;
    g_snapshot_pending = true;
    up_write(&uvm_dirty_ds_rwsem);
    ds_write_locked = false;

    uvm_dirty_query_barrier_end_fn();
    query_barrier_held = false;

    if (old_snapshot.priv != NULL) {
        uvm_dirty_ds_destroy(&old_snapshot);
    }

    goto out;

out:
    if (ds_write_locked)
        up_write(&uvm_dirty_ds_rwsem);

    if (query_barrier_held)
        uvm_dirty_query_barrier_end_fn();

    mutex_unlock(&uvm_dirty_lifecycle_lock);
    return ret == 0 ? count : ret;
}

static const struct proc_ops dirty_tracking_query_cutover_fops = {
    .proc_write = dirty_tracking_query_cutover,
};

static ssize_t dirty_tracking_resume(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {
    
    mutex_lock(&uvm_dirty_lifecycle_lock);

    char kbuf[16];
    size_t to_copy = min(count, sizeof(kbuf) - 1);
    if (copy_from_user(kbuf, buf, to_copy)) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    kbuf[to_copy] = '\0';

    if (!uvm_dirty_tracking_started()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    if (uvm_dirty_tracking_active()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    pid_t target_pid = 0;
    if (sscanf(kbuf, "%d", &target_pid) != 1 || target_pid <= 0) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EINVAL;
    }
    if (target_pid != uvm_owner_tgid_with_dirty_tracking_started()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -ESRCH;
    }

    NV_STATUS status;

    if (!uvm_dirty_invalidate_fn || !uvm_dirty_activate_now_fn || !uvm_dirty_barrier_end_fn) {
        pr_err("uvm_dirty: one or more function pointers not registered\n");
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    // EDIT BY SANKALP MITTAL
    if (unlikely(g_dirty_ds.stats.enabled)) {
        ktime_t t_cb = ktime_get();
        status = uvm_dirty_invalidate_fn();
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t_cb)), &g_dirty_ds.stats.invalidate_time_ns);
        atomic_long_inc(&g_dirty_ds.stats.invalidate_count);
    } else {
        status = uvm_dirty_invalidate_fn();
    }
    // END OF EDIT
    if (status != NV_OK) {
        pr_err("uvm_dirty: invalidate fn failed for pid %d (status %d)\n", target_pid, status);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    // EDIT BY SANKALP MITTAL
    if (unlikely(g_dirty_ds.stats.enabled)) {
        ktime_t t_cb = ktime_get();
        status = uvm_dirty_activate_now_fn();
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t_cb)), &g_dirty_ds.stats.activate_time_ns);
        atomic_long_inc(&g_dirty_ds.stats.activate_count);
    } else {
        status = uvm_dirty_activate_now_fn();
    }
    // END OF EDIT
    if (status != NV_OK) {
        pr_err("uvm_dirty: activate fn failed for pid %d (status %d)\n", target_pid, status);
        uvm_dirty_barrier_end_fn();
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    // EDIT BY SANKALP MITTAL
    if (unlikely(g_dirty_ds.stats.enabled)) {
        ktime_t t_cb = ktime_get();
        status = uvm_dirty_barrier_end_fn();
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t_cb)), &g_dirty_ds.stats.barrier_end_time_ns);
        atomic_long_inc(&g_dirty_ds.stats.barrier_end_count);
    } else {
        status = uvm_dirty_barrier_end_fn();
    }
    // END OF EDIT
    if (status != NV_OK) {
        pr_err("uvm_dirty: barrier_end fn failed for pid %d (status %d)\n", target_pid, status);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    mutex_unlock(&uvm_dirty_lifecycle_lock);
    return count;
}

static const struct proc_ops dirty_tracking_resume_fops = {
    .proc_write = dirty_tracking_resume,
};

static ssize_t dirty_tracking_pause(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {

    mutex_lock(&uvm_dirty_lifecycle_lock);

    char kbuf[16];
    size_t to_copy = min(count, sizeof(kbuf) - 1);
    if (copy_from_user(kbuf, buf, to_copy)) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    kbuf[to_copy] = '\0';

    if (!uvm_dirty_tracking_started()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    if (!uvm_dirty_tracking_active()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    pid_t target_pid = 0;
    if (sscanf(kbuf, "%d", &target_pid) != 1 || target_pid <= 0) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EINVAL;
    }
    if (target_pid != uvm_owner_tgid_with_dirty_tracking_active()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -ESRCH;
    }

    if (!uvm_dirty_query_barrier_begin_fn || !uvm_dirty_query_barrier_end_fn) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    NV_STATUS status = uvm_dirty_query_barrier_begin_fn();
    if (status != NV_OK) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -nv_status_to_errno(status);
    }
    
    WRITE_ONCE(g_uvm_dirty_tracking_active, false);

    status = uvm_dirty_query_barrier_end_fn();
    if (status != NV_OK) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -nv_status_to_errno(status);
    }

    mutex_unlock(&uvm_dirty_lifecycle_lock);
    return count;
}

static const struct proc_ops dirty_tracking_pause_fops = {
    .proc_write = dirty_tracking_pause,
};

static ssize_t dirty_tracking_start(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {

    mutex_lock(&uvm_dirty_lifecycle_lock);

    // input should be of the form <pid> <delta/cumulative>
    char kbuf[32];
    size_t to_copy = min(count, sizeof(kbuf) - 1);
    if (copy_from_user(kbuf, buf, to_copy)) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    kbuf[to_copy] = '\0';

    pid_t target_pid = 0;
    char querying_state_str[16];
    // EDIT BY SANKALP MITTAL
    int consumed = 0;
    if (sscanf(kbuf, "%d %15s%n", &target_pid, querying_state_str, &consumed) != 2 || target_pid <= 0) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EINVAL;
    }
    /* Reject trailing garbage after the mode keyword. */
    for (int i = consumed; kbuf[i] != '\0'; i++) {
        if (kbuf[i] != ' ' && kbuf[i] != '\t' && kbuf[i] != '\n' && kbuf[i] != '\r') {
            mutex_unlock(&uvm_dirty_lifecycle_lock);
            return -EINVAL;
        }
    }
    // END OF EDIT

    if (uvm_dirty_tracking_started()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EBUSY;
    }

    if (strcmp(querying_state_str, "delta") == 0) {
        g_dirty_tracking_querying_state = UVM_DIRTY_TRACKING_QUERY_DELTA;
    } else if (strcmp(querying_state_str, "cumulative") == 0) {
        g_dirty_tracking_querying_state = UVM_DIRTY_TRACKING_QUERY_CUMULATIVE;
    } else {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EINVAL;
    }

    NV_STATUS status;
    status = uvm_dirty_page_table_init(target_pid);
    if (status != NV_OK) {
        pr_err("uvm_dirty: page table init failed for pid %d (status %d)\n", target_pid, status);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    if (!uvm_dirty_invalidate_fn || !uvm_dirty_activate_now_fn || !uvm_dirty_barrier_end_fn) {
        pr_err("uvm_dirty: one or more function pointers not registered\n");
        uvm_dirty_page_table_destroy(false);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    // EDIT BY SANKALP MITTAL
    if (unlikely(g_dirty_ds.stats.enabled)) {
        ktime_t t_cb = ktime_get();
        status = uvm_dirty_invalidate_fn();
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t_cb)), &g_dirty_ds.stats.invalidate_time_ns);
        atomic_long_inc(&g_dirty_ds.stats.invalidate_count);
    } else {
        status = uvm_dirty_invalidate_fn();
    }
    // END OF EDIT
    if (status != NV_OK) {
        pr_err("uvm_dirty: invalidate fn failed for pid %d (status %d) — is the pid a valid tgid with a UVM va_space open?\n",
               target_pid, status);
        uvm_dirty_page_table_destroy(false);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    // EDIT BY SANKALP MITTAL
    if (unlikely(g_dirty_ds.stats.enabled)) {
        ktime_t t_cb = ktime_get();
        status = uvm_dirty_activate_now_fn();
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t_cb)), &g_dirty_ds.stats.activate_time_ns);
        atomic_long_inc(&g_dirty_ds.stats.activate_count);
    } else {
        status = uvm_dirty_activate_now_fn();
    }
    // END OF EDIT
    if (status != NV_OK) {
        pr_err("uvm_dirty: activate fn failed for pid %d (status %d)\n", target_pid, status);
        uvm_dirty_barrier_end_fn();
        uvm_dirty_page_table_destroy(false);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    // EDIT BY SANKALP MITTAL
    if (unlikely(g_dirty_ds.stats.enabled)) {
        ktime_t t_cb = ktime_get();
        status = uvm_dirty_barrier_end_fn();
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t_cb)), &g_dirty_ds.stats.barrier_end_time_ns);
        atomic_long_inc(&g_dirty_ds.stats.barrier_end_count);
    } else {
        status = uvm_dirty_barrier_end_fn();
    }
    // END OF EDIT
    if (status != NV_OK) {
        pr_err("uvm_dirty: barrier_end fn failed for pid %d (status %d)\n", target_pid, status);
        uvm_dirty_page_table_destroy(false);
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    mutex_unlock(&uvm_dirty_lifecycle_lock);
    return count;
}

static const struct proc_ops dirty_tracking_start_fops = {
    .proc_write = dirty_tracking_start,
};


// EDIT BY KUSHAGRA
#define UVM_DIRTY_DS_STATS_FILE "/tmp/uvm_dirty_ds_stats.txt"

/*
 * Write accumulated stats to UVM_DIRTY_DS_STATS_FILE.
 * Called once when dirty tracking stops — no I/O during the experiment.
 */

/* original stats flush (no lock wait fields)
static void uvm_dirty_ds_stats_flush(void)
{
    struct file *f;
    char buf[768];
    int len;
    loff_t pos = 0;
    long ic, lc, fc;
    s64 it, lt, ft;

    if (!g_dirty_ds.stats.enabled)
        return;

    ic = DIRTY_DS_STATS_READ(&g_dirty_ds, insert_count);
    lc = DIRTY_DS_STATS_READ(&g_dirty_ds, lookup_count);
    fc = DIRTY_DS_STATS_READ(&g_dirty_ds, for_each_in_range_count);
    it = DIRTY_DS_STATS_READ64(&g_dirty_ds, insert_time_ns);
    lt = DIRTY_DS_STATS_READ64(&g_dirty_ds, lookup_time_ns);
    ft = DIRTY_DS_STATS_READ64(&g_dirty_ds, for_each_in_range_time_ns);

    len = snprintf(buf, sizeof(buf),
        "insert:            count=%ld  total_ns=%lld  avg_ns=%lld\n"
        "lookup:            count=%ld  total_ns=%lld  avg_ns=%lld\n"
        "for_each_in_range: count=%ld  total_ns=%lld  avg_ns=%lld\n"
        "init:              count=%ld  total_ns=%lld\n"
        "destroy:           count=%ld  total_ns=%lld\n",
        ic, it, ic ? it / ic : 0,
        lc, lt, lc ? lt / lc : 0,
        fc, ft, fc ? ft / fc : 0,
        DIRTY_DS_STATS_READ(&g_dirty_ds, init_count),
        DIRTY_DS_STATS_READ64(&g_dirty_ds, init_time_ns),
        DIRTY_DS_STATS_READ(&g_dirty_ds, destroy_count),
        DIRTY_DS_STATS_READ64(&g_dirty_ds, destroy_time_ns));

    f = filp_open(UVM_DIRTY_DS_STATS_FILE, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(f)) {
        pr_warn("uvm: could not open stats file %s\n", UVM_DIRTY_DS_STATS_FILE);
        return;
    }
    kernel_write(f, buf, len, &pos);
    filp_close(f, NULL);
}
*/

// EDIT BY VIDHI JAIN
static void uvm_dirty_ds_stats_flush(void)
{
    struct file *f;
    char buf[1280];
    int len;
    loff_t pos = 0;
    long ic, lc, fc;
    s64 it, lt, ft, ilw, llw, flw;

    if (!g_dirty_ds.stats.enabled)
        return;

    ic  = DIRTY_DS_STATS_READ(&g_dirty_ds, insert_count);
    lc  = DIRTY_DS_STATS_READ(&g_dirty_ds, lookup_count);
    fc  = DIRTY_DS_STATS_READ(&g_dirty_ds, for_each_in_range_count);
    it  = DIRTY_DS_STATS_READ64(&g_dirty_ds, insert_time_ns);
    lt  = DIRTY_DS_STATS_READ64(&g_dirty_ds, lookup_time_ns);
    ft  = DIRTY_DS_STATS_READ64(&g_dirty_ds, for_each_in_range_time_ns);
    ilw = DIRTY_DS_STATS_READ64(&g_dirty_ds, insert_lock_wait_ns);
    llw = DIRTY_DS_STATS_READ64(&g_dirty_ds, lookup_lock_wait_ns);
    flw = DIRTY_DS_STATS_READ64(&g_dirty_ds, for_each_lock_wait_ns);

    // EDIT BY SANKALP MITTAL
    long invc = DIRTY_DS_STATS_READ(&g_dirty_ds, invalidate_count);
    long actc = DIRTY_DS_STATS_READ(&g_dirty_ds, activate_count);
    long barc = DIRTY_DS_STATS_READ(&g_dirty_ds, barrier_end_count);
    s64  invt = DIRTY_DS_STATS_READ64(&g_dirty_ds, invalidate_time_ns);
    s64  actt = DIRTY_DS_STATS_READ64(&g_dirty_ds, activate_time_ns);
    s64  bart = DIRTY_DS_STATS_READ64(&g_dirty_ds, barrier_end_time_ns);
    // END OF EDIT

    len = snprintf(buf, sizeof(buf),
        "insert:            count=%ld  avg_ns=%lld  avg_lock_wait_ns=%lld\n"
        "lookup:            count=%ld  avg_ns=%lld  avg_lock_wait_ns=%lld\n"
        "for_each_in_range: count=%ld  avg_ns=%lld  avg_lock_wait_ns=%lld\n"
        "init:              count=%ld  total_ns=%lld\n"
        "destroy:           count=%ld  total_ns=%lld\n"
        "invalidate:        count=%ld  avg_ns=%lld\n"
        "activate_now:      count=%ld  avg_ns=%lld\n"
        "barrier_end:       count=%ld  avg_ns=%lld\n",
        ic, ic ? it / ic : 0, ic ? ilw / ic : 0,
        lc, lc ? lt / lc : 0, lc ? llw / lc : 0,
        fc, fc ? ft / fc : 0, fc ? flw / fc : 0,
        DIRTY_DS_STATS_READ(&g_dirty_ds, init_count),
        DIRTY_DS_STATS_READ64(&g_dirty_ds, init_time_ns),
        DIRTY_DS_STATS_READ(&g_dirty_ds, destroy_count),
        DIRTY_DS_STATS_READ64(&g_dirty_ds, destroy_time_ns),
        // EDIT BY SANKALP MITTAL
        invc, invc ? invt / invc : 0,
        actc, actc ? actt / actc : 0,
        barc, barc ? bart / barc : 0);
        // END OF EDIT

    f = filp_open(UVM_DIRTY_DS_STATS_FILE, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(f)) {
        pr_warn("uvm: could not open stats file %s\n", UVM_DIRTY_DS_STATS_FILE);
        return;
    }
    kernel_write(f, buf, len, &pos);
    filp_close(f, NULL);
}
// END OF EDIT

static ssize_t dirty_tracking_stop(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {

    mutex_lock(&uvm_dirty_lifecycle_lock);

    char kbuf[16];
    size_t to_copy = min(count, sizeof(kbuf) - 1);
    if (copy_from_user(kbuf, buf, to_copy)) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }
    kbuf[to_copy] = '\0';

    if (!uvm_dirty_tracking_started()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EFAULT;
    }

    pid_t target_pid = 0;
    if (sscanf(kbuf, "%d", &target_pid) != 1 || target_pid <= 0) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -EINVAL;
    }
    if (target_pid != uvm_owner_tgid_with_dirty_tracking_started()) {
        mutex_unlock(&uvm_dirty_lifecycle_lock);
        return -ESRCH;
    }

    // EDIT BY VIDHI JAIN
    if (uvm_dirty_restore_fn) uvm_dirty_restore_fn();
    // END OF EDIT

    uvm_dirty_page_table_destroy(false);
    uvm_dirty_ds_stats_flush();
    g_snapshot_pending = false;
    mutex_unlock(&uvm_dirty_lifecycle_lock);
    return count;
}

static const struct proc_ops dirty_tracking_stop_fops = {
    .proc_write = dirty_tracking_stop,
};

// END OF EDIT

/* original procfs stats reader (no lock wait fields)
static int nv_procfs_read_dirty_ds_stats(struct seq_file *s, void *v)
{
    long ic = DIRTY_DS_STATS_READ(&g_dirty_ds, insert_count);
    long lc = DIRTY_DS_STATS_READ(&g_dirty_ds, lookup_count);
    long fc = DIRTY_DS_STATS_READ(&g_dirty_ds, for_each_in_range_count);
    s64  it = DIRTY_DS_STATS_READ64(&g_dirty_ds, insert_time_ns);
    s64  lt = DIRTY_DS_STATS_READ64(&g_dirty_ds, lookup_time_ns);
    s64  ft = DIRTY_DS_STATS_READ64(&g_dirty_ds, for_each_in_range_time_ns);

    seq_printf(s, "enabled:           %d\n", g_dirty_ds.stats.enabled);
    seq_printf(s, "insert:            count=%ld  total_ns=%lld  avg_ns=%lld\n",
               ic, it, ic ? it / ic : 0);
    seq_printf(s, "lookup:            count=%ld  total_ns=%lld  avg_ns=%lld\n",
               lc, lt, lc ? lt / lc : 0);
    seq_printf(s, "for_each_in_range: count=%ld  total_ns=%lld  avg_ns=%lld\n",
               fc, ft, fc ? ft / fc : 0);
    seq_printf(s, "init:              count=%ld  total_ns=%lld\n",
               DIRTY_DS_STATS_READ(&g_dirty_ds, init_count),
               DIRTY_DS_STATS_READ64(&g_dirty_ds, init_time_ns));
    seq_printf(s, "destroy:           count=%ld  total_ns=%lld\n",
               DIRTY_DS_STATS_READ(&g_dirty_ds, destroy_count),
               DIRTY_DS_STATS_READ64(&g_dirty_ds, destroy_time_ns));
    return 0;
}
*/

// EDIT BY VIDHI JAIN
static int nv_procfs_read_dirty_ds_stats(struct seq_file *s, void *v)
{
    long ic  = DIRTY_DS_STATS_READ(&g_dirty_ds, insert_count);
    long lc  = DIRTY_DS_STATS_READ(&g_dirty_ds, lookup_count);
    long fc  = DIRTY_DS_STATS_READ(&g_dirty_ds, for_each_in_range_count);
    s64  it  = DIRTY_DS_STATS_READ64(&g_dirty_ds, insert_time_ns);
    s64  lt  = DIRTY_DS_STATS_READ64(&g_dirty_ds, lookup_time_ns);
    s64  ft  = DIRTY_DS_STATS_READ64(&g_dirty_ds, for_each_in_range_time_ns);
    s64  ilw = DIRTY_DS_STATS_READ64(&g_dirty_ds, insert_lock_wait_ns);
    s64  llw = DIRTY_DS_STATS_READ64(&g_dirty_ds, lookup_lock_wait_ns);
    s64  flw = DIRTY_DS_STATS_READ64(&g_dirty_ds, for_each_lock_wait_ns);

    seq_printf(s, "enabled:           %d\n", g_dirty_ds.stats.enabled);
    seq_printf(s, "insert:            count=%ld  avg_ns=%lld  avg_lock_wait_ns=%lld\n",
               ic, ic ? it / ic : 0, ic ? ilw / ic : 0);
    seq_printf(s, "lookup:            count=%ld  avg_ns=%lld  avg_lock_wait_ns=%lld\n",
               lc, lc ? lt / lc : 0, lc ? llw / lc : 0);
    seq_printf(s, "for_each_in_range: count=%ld  avg_ns=%lld  avg_lock_wait_ns=%lld\n",
               fc, fc ? ft / fc : 0, fc ? flw / fc : 0);
    seq_printf(s, "init:              count=%ld  total_ns=%lld\n",
               DIRTY_DS_STATS_READ(&g_dirty_ds, init_count),
               DIRTY_DS_STATS_READ64(&g_dirty_ds, init_time_ns));
    seq_printf(s, "destroy:           count=%ld  total_ns=%lld\n",
               DIRTY_DS_STATS_READ(&g_dirty_ds, destroy_count),
               DIRTY_DS_STATS_READ64(&g_dirty_ds, destroy_time_ns));
    // EDIT BY SANKALP MITTAL
    {
        long invc = DIRTY_DS_STATS_READ(&g_dirty_ds, invalidate_count);
        long actc = DIRTY_DS_STATS_READ(&g_dirty_ds, activate_count);
        long barc = DIRTY_DS_STATS_READ(&g_dirty_ds, barrier_end_count);
        s64  invt = DIRTY_DS_STATS_READ64(&g_dirty_ds, invalidate_time_ns);
        s64  actt = DIRTY_DS_STATS_READ64(&g_dirty_ds, activate_time_ns);
        s64  bart = DIRTY_DS_STATS_READ64(&g_dirty_ds, barrier_end_time_ns);
        seq_printf(s, "invalidate:        count=%ld  avg_ns=%lld\n",
                   invc, invc ? invt / invc : 0);
        seq_printf(s, "activate_now:      count=%ld  avg_ns=%lld\n",
                   actc, actc ? actt / actc : 0);
        seq_printf(s, "barrier_end:       count=%ld  avg_ns=%lld\n",
                   barc, barc ? bart / barc : 0);
    }
    // END OF EDIT
    return 0;
}
// END OF EDIT

static int nv_procfs_read_dirty_ds_stats_entry(struct seq_file *s, void *v)
{
    return nv_procfs_read_dirty_ds_stats(s, v);
}

UVM_DEFINE_SINGLE_PROCFS_FILE(dirty_ds_stats_entry);

static ssize_t dirty_ds_stats_toggle(struct file *file,
                                      const char __user *buf,
                                      size_t count,
                                      loff_t *ppos)
{
    char kbuf[8];
    size_t to_copy = min(count, sizeof(kbuf) - 1);

    if (copy_from_user(kbuf, buf, to_copy))
        return -EFAULT;
    kbuf[to_copy] = '\0';

    if (strncmp(kbuf, "enable", 6) == 0){
        g_dirty_ds.stats.enabled = true;
        // clear counters when enabling to ensure clean stats for the experiment
        atomic_long_set(&g_dirty_ds.stats.insert_count, 0);
        atomic64_set(&g_dirty_ds.stats.insert_time_ns, 0);
        atomic_long_set(&g_dirty_ds.stats.lookup_count, 0);
        atomic64_set(&g_dirty_ds.stats.lookup_time_ns, 0);
        atomic_long_set(&g_dirty_ds.stats.for_each_in_range_count, 0);
        atomic64_set(&g_dirty_ds.stats.for_each_in_range_time_ns, 0);
        atomic_long_set(&g_dirty_ds.stats.init_count, 0);
        atomic64_set(&g_dirty_ds.stats.init_time_ns, 0);
        atomic_long_set(&g_dirty_ds.stats.destroy_count, 0);
        atomic64_set(&g_dirty_ds.stats.destroy_time_ns, 0);
        // EDIT BY VIDHI JAIN
        atomic64_set(&g_dirty_ds.stats.insert_lock_wait_ns, 0);
        atomic64_set(&g_dirty_ds.stats.lookup_lock_wait_ns, 0);
        atomic64_set(&g_dirty_ds.stats.for_each_lock_wait_ns, 0);
        // END OF EDIT
        // EDIT BY SANKALP MITTAL
        atomic_long_set(&g_dirty_ds.stats.invalidate_count, 0);
        atomic64_set(&g_dirty_ds.stats.invalidate_time_ns, 0);
        atomic_long_set(&g_dirty_ds.stats.activate_count, 0);
        atomic64_set(&g_dirty_ds.stats.activate_time_ns, 0);
        atomic_long_set(&g_dirty_ds.stats.barrier_end_count, 0);
        atomic64_set(&g_dirty_ds.stats.barrier_end_time_ns, 0);
        // END OF EDIT
    }
    else if (strncmp(kbuf, "disable", 7) == 0)
        g_dirty_ds.stats.enabled = false;
    else
        return -EINVAL;

    return count;
}

static const struct proc_ops dirty_ds_stats_toggle_fops = {
    .proc_write = dirty_ds_stats_toggle,
};
// END OF EDIT

NV_STATUS uvm_dirty_procfs_init(struct proc_dir_entry *parent)
{
    struct proc_dir_entry *entry;
    
    entry = NV_CREATE_PROC_FILE("dirty_tracking_query_dump", parent, dirty_tracking_query_dump_entry, NULL); 
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;   

    // EDIT BY ADITI KHANDELIA
    entry = proc_create("dirty_tracking_start",
                        0666,
                        parent,
                        &dirty_tracking_start_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    entry = proc_create("dirty_tracking_stop",
                        0666,
                        parent,
                        &dirty_tracking_stop_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;
    
    entry = proc_create("dirty_tracking_pause",
                        0666,
                        parent,
                        &dirty_tracking_pause_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;
    
    entry = proc_create("dirty_tracking_resume",
                        0666,
                        parent,
                        &dirty_tracking_resume_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    entry = proc_create("dirty_tracking_query_cutover",
                        0666,
                        parent,
                        &dirty_tracking_query_cutover_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    // EDIT BY KUSHAGRA
    entry = NV_CREATE_PROC_FILE("dirty_ds_stats", parent, dirty_ds_stats_entry, NULL);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    entry = proc_create("dirty_ds_stats_toggle",
                        0222,
                        parent,
                        &dirty_ds_stats_toggle_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;
    // END OF EDIT

    return NV_OK;
}
// END OF EDIT - procfs query interface

// END OF EDIT

// TODO: Bug 1710855: Tweak this number through benchmarks
#define UVM_SPIN_LOOP_SCHEDULE_TIMEOUT_NS   (10*1000ULL)
#define UVM_SPIN_LOOP_PRINT_TIMEOUT_SEC     30ULL

// Default to debug prints being enabled for debug and develop builds and
// disabled for release builds.
static int uvm_debug_prints = UVM_IS_DEBUG() || UVM_IS_DEVELOP();

// Make the module param writable so that prints can be enabled or disabled at
// any time by modifying the module parameter.
module_param(uvm_debug_prints, int, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(uvm_debug_prints, "Enable uvm debug prints.");

bool uvm_debug_prints_enabled(void)
{
    return uvm_debug_prints != 0;
}

// This parameter allows a program in user mode to call the kernel tests
// defined in this module. This parameter should only be used for testing and
// must not be set to true otherwise since it breaks security when it is
// enabled. By default and for safety reasons this parameter is set to false.
int uvm_enable_builtin_tests __read_mostly = 0;
module_param(uvm_enable_builtin_tests, int, S_IRUGO);
MODULE_PARM_DESC(uvm_enable_builtin_tests,
                 "Enable the UVM built-in tests. (This is a security risk)");

// Default to release asserts being enabled.
int uvm_release_asserts __read_mostly = 1;

// Make the module param writable so that release asserts can be enabled or
// disabled at any time by modifying the module parameter.
module_param(uvm_release_asserts, int, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(uvm_release_asserts, "Enable uvm asserts included in release builds.");

// Default to failed release asserts not dumping stack.
int uvm_release_asserts_dump_stack __read_mostly = 0;

// Make the module param writable so that dumping the stack can be enabled and
// disabled at any time by modifying the module parameter.
module_param(uvm_release_asserts_dump_stack, int, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(uvm_release_asserts_dump_stack, "dump_stack() on failed UVM release asserts.");

// Default to failed release asserts not setting the global UVM error.
int uvm_release_asserts_set_global_error __read_mostly = 0;

// Make the module param writable so that setting the global fatal error can be
// enabled and disabled at any time by modifying the module parameter.
module_param(uvm_release_asserts_set_global_error, int, S_IRUGO|S_IWUSR);
MODULE_PARM_DESC(uvm_release_asserts_set_global_error, "Set UVM global fatal error on failed release asserts.");

// A separate flag to enable setting global error, to be used by tests only.
bool uvm_release_asserts_set_global_error_for_tests __read_mostly = false;

//
// Convert kernel errno codes to corresponding NV_STATUS
//
NV_STATUS errno_to_nv_status(int errnoCode)
{
    if (errnoCode < 0)
        errnoCode = -errnoCode;

    switch (errnoCode)
    {
        case 0:
            return NV_OK;

        case E2BIG:
        case EINVAL:
            return NV_ERR_INVALID_ARGUMENT;

        case EACCES:
            return NV_ERR_INVALID_ACCESS_TYPE;

        case EADDRINUSE:
        case EADDRNOTAVAIL:
            return NV_ERR_UVM_ADDRESS_IN_USE;

        case EFAULT:
            return NV_ERR_INVALID_ADDRESS;

        case EOVERFLOW:
            return NV_ERR_OUT_OF_RANGE;

        case EINTR:
        case EBUSY:
        case EAGAIN:
            return NV_ERR_BUSY_RETRY;

        case ENXIO:
        case ENODEV:
            return NV_ERR_MODULE_LOAD_FAILED;

        case ENOMEM:
            return NV_ERR_NO_MEMORY;

        case EPERM:
            return NV_ERR_INSUFFICIENT_PERMISSIONS;

        case ESRCH:
            return NV_ERR_PID_NOT_FOUND;

        case ETIMEDOUT:
            return NV_ERR_TIMEOUT;

        case EEXIST:
            return NV_ERR_IN_USE;

        case ENOSYS:
        case EOPNOTSUPP:
            return NV_ERR_NOT_SUPPORTED;

        case ENOENT:
            return NV_ERR_NO_VALID_PATH;

        case EIO:
            return NV_ERR_RC_ERROR;

        case ENODATA:
            return NV_ERR_OBJECT_NOT_FOUND;

        default:
            return NV_ERR_GENERIC;
    };
}

// Returns POSITIVE errno
int nv_status_to_errno(NV_STATUS status)
{
    switch (status) {
        case NV_OK:
            return 0;

        case NV_ERR_BUSY_RETRY:
            return EAGAIN;

        case NV_ERR_INSUFFICIENT_PERMISSIONS:
            return EPERM;

        case NV_ERR_GPU_UUID_NOT_FOUND:
            return ENODEV;

        case NV_ERR_INSUFFICIENT_RESOURCES:
        case NV_ERR_NO_MEMORY:
            return ENOMEM;

        case NV_ERR_INVALID_ACCESS_TYPE:
            return EACCES;

        case NV_ERR_INVALID_ADDRESS:
            return EFAULT;

        case NV_ERR_INVALID_ARGUMENT:
        case NV_ERR_INVALID_DEVICE:
        case NV_ERR_INVALID_PARAMETER:
        case NV_ERR_INVALID_REQUEST:
        case NV_ERR_INVALID_STATE:
            return EINVAL;

        case NV_ERR_NOT_SUPPORTED:
            return ENOSYS;

        case NV_ERR_OBJECT_NOT_FOUND:
            return ENODATA;

        case NV_ERR_MODULE_LOAD_FAILED:
            return ENXIO;

        case NV_ERR_OVERLAPPING_UVM_COMMIT:
        case NV_ERR_UVM_ADDRESS_IN_USE:
            return EADDRINUSE;

        case NV_ERR_PID_NOT_FOUND:
            return ESRCH;

        case NV_ERR_TIMEOUT:
        case NV_ERR_TIMEOUT_RETRY:
            return ETIMEDOUT;

        case NV_ERR_IN_USE:
            return EEXIST;

        case NV_ERR_NO_VALID_PATH:
            return ENOENT;

        case NV_ERR_RC_ERROR:
        case NV_ERR_ECC_ERROR:
            return EIO;

        case NV_ERR_OUT_OF_RANGE:
            return EOVERFLOW;

        default:
            UVM_ASSERT_MSG(0, "No errno conversion set up for NV_STATUS %s\n", nvstatusToString(status));
            return EINVAL;
    }
}

//
// This routine retrieves the process ID of current, but makes no attempt to
// refcount or lock the pid in place.
//
unsigned uvm_get_stale_process_id(void)
{
    return (unsigned)task_tgid_vnr(current);
}

unsigned uvm_get_stale_thread_id(void)
{
    return (unsigned)task_pid_vnr(current);
}

void on_uvm_test_fail(void)
{
    (void)NULL;
}

void on_uvm_assert(void)
{
    (void)NULL;
#ifdef __COVERITY__
    __coverity_panic__()
#endif
}

NV_STATUS uvm_spin_loop(uvm_spin_loop_t *spin)
{
    NvU64 curr = NV_GETTIME();

    // This schedule() is required for functionality, not just system
    // performance. It allows RM to run and unblock the UVM driver:
    //
    // - UVM must service faults in order for RM to idle/preempt a context
    // - RM must service interrupts which stall UVM (SW methods, stalling CE
    //   interrupts, etc) in order for UVM to service faults
    //
    // Even though UVM's bottom half is preemptable, we have encountered cases
    // in which a user thread running in RM won't preempt the UVM driver's
    // thread unless the UVM driver thread gives up its timeslice. This is also
    // theoretically possible if the RM thread has a low nice priority.
    //
    // TODO: Bug 1710855: Look into proper prioritization of these threads as a longer-term
    //       solution.
    if (curr - spin->start_time_ns >= UVM_SPIN_LOOP_SCHEDULE_TIMEOUT_NS && NV_MAY_SLEEP()) {
        schedule();
        curr = NV_GETTIME();
    }

    cpu_relax();

    // TODO: Bug 1710855: Also check fatal_signal_pending() here if the caller can handle it.

    if (curr - spin->print_time_ns >= 1000*1000*1000*UVM_SPIN_LOOP_PRINT_TIMEOUT_SEC) {
        spin->print_time_ns = curr;
        return NV_ERR_TIMEOUT_RETRY;
    }

    return NV_OK;
}

static char uvm_digit_to_hex(unsigned value)
{
    if (value >= 10)
        return value - 10 + 'a';
    else
        return value + '0';
}

void uvm_uuid_string(char *buffer, const NvProcessorUuid *pUuidStruct)
{
    char *str = buffer;
    unsigned i;
    unsigned dashMask = 1 << 4 | 1 << 6 | 1 << 8 | 1 << 10;

    for (i = 0; i < 16; i++) {
        *str++ = uvm_digit_to_hex(pUuidStruct->uuid[i] >> 4);
        *str++ = uvm_digit_to_hex(pUuidStruct->uuid[i] & 0xF);

        if (dashMask & (1 << (i + 1)))
            *str++ = '-';
    }

    *str = 0;
}
