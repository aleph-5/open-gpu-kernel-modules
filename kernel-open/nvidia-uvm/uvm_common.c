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
#include "uvm_linux.h"
#include "uvm_forward_decl.h"

// EDIT BY ARUSH
#if defined(CONFIG_PROC_FS)
#include "uvm_procfs.h"
#include "nv-procfs.h"
#endif
// END OF EDIT

// EDIT BY VIDHI JAIN
static unsigned long dirty_query_start = 0UL;
static unsigned long dirty_query_end = ~0UL;
// END OF EDIT

// EDIT BY ARUSH
// Registered from uvm_va_space.c at module init; NULL until then.
void (*uvm_dirty_invalidate_fn)(void) = NULL;
// END OF EDIT

// EDIT BY ADITI KHANDELIA
static DEFINE_XARRAY(pid_to_page_table_xa);
static struct xarray *pid_to_page_table = &pid_to_page_table_xa;

bool uvm_dirty_tracking_active_for_pid(pid_t pid) {
    struct uvm_dirty_page_table* page_table = uvm_dirty_page_table_by_pid(pid);
    return page_table != NULL;
}

struct uvm_dirty_page_table* uvm_dirty_page_table_by_pid(pid_t pid) {
    struct uvm_dirty_page_table* page_table = xa_load(pid_to_page_table, pid);
    if (page_table == NULL) {
        printk(KERN_ERR "Dirty page table not initialized for pid %d\n", pid);
        return NULL;
    }
    return page_table;
}

NV_STATUS uvm_dirty_page_table_init(pid_t pid) {

    struct uvm_dirty_page_table* page_table_pointer = uvm_dirty_page_table_by_pid(pid);

    if (page_table_pointer != NULL) {
        NV_STATUS destruction_status = uvm_dirty_page_table_destroy(pid);
        if (destruction_status != NV_OK) {
            printk(KERN_ERR "Failed to destroy existing dirty page table for pid %d\n", pid);
            return NV_ERR_GENERIC;
        }
        printk(KERN_WARNING "Dirty page table was already initialized, reinitializing it for pid %d\n", pid);
    }

    page_table_pointer = kmalloc(sizeof(struct uvm_dirty_page_table), GFP_KERNEL);
    if (page_table_pointer == NULL) {
        printk(KERN_ERR "Failed to allocate memory for dirty page table for pid %d\n", pid);
        return NV_ERR_NO_MEMORY;
    }
    xa_init(&page_table_pointer->pages);

    void* ret = xa_store(pid_to_page_table, pid, page_table_pointer, GFP_KERNEL);
    if (xa_err(ret)) {
        printk(KERN_ERR "Failed to store dirty page table in xarray for pid %d\n", pid);
        kfree(page_table_pointer);
        return NV_ERR_NO_MEMORY;
    }

    printk(KERN_INFO "Dirty page table initialized for pid %d\n", pid);
    return NV_OK;
}

NV_STATUS uvm_dirty_page_table_destroy(pid_t pid) {
    struct uvm_dirty_page_table* page_table_pointer = uvm_dirty_page_table_by_pid(pid);

    if (page_table_pointer == NULL) {
        printk(KERN_ERR "Dirty page table not initialized for pid %d\n", pid);
        return NV_ERR_GENERIC;
    }

    unsigned long index;
    struct dirty_page_info *info;
    xa_for_each(&page_table_pointer->pages, index, info) {
        kfree(info);
    }
    xa_destroy(&page_table_pointer->pages);
    kfree(page_table_pointer);
    xa_erase(pid_to_page_table, pid);
    printk(KERN_INFO "Dirty page table destroyed for pid %d\n", pid);

    return NV_OK;
}

NV_STATUS uvm_dirty_page_table_record(unsigned long page_number,
    unsigned long timestamp,
    pid_t pid) { // (EDIT BY VIDHI JAIN)

    struct dirty_page_info* existing_info = uvm_dirty_page_table_lookup(page_number, pid);
    if (existing_info != NULL) {
        printk(KERN_INFO "Dirty page info already exists for page_number=0x%lx for pid %d, skipping\n", page_number, pid);
        return NV_OK;
    }

    struct dirty_page_info* info = kmalloc(sizeof(struct dirty_page_info), GFP_ATOMIC);
    if (info == NULL) {
        printk(KERN_ERR "Failed to allocate memory for dirty page info\n");
        return NV_ERR_NO_MEMORY;
    }

    info->page_number = page_number;
    info->timestamp = timestamp;
    // EDIT BY VIDHI JAIN
    info->pid = pid;
    // END OF EDIT

    struct uvm_dirty_page_table* page_table_pointer = uvm_dirty_page_table_by_pid(pid);

    if (page_table_pointer == NULL) {
        printk(KERN_ERR "Dirty page table not initialized for pid %d\n", pid);
        kfree(info);
        return NV_ERR_NO_MEMORY;
    }

    void* ret = xa_store(&page_table_pointer->pages, page_number, info, GFP_ATOMIC);
    if (xa_err(ret)) {
        printk(KERN_ERR "Failed to store dirty page info in xarray for pid %d\n", pid);
        kfree(info);
        return NV_ERR_NO_MEMORY;
    }

    printk(KERN_INFO "Recorded dirty page: page_number=0x%lx, timestamp=%lu\n", page_number, timestamp);
    return NV_OK;
}

struct dirty_page_info* uvm_dirty_page_table_lookup(unsigned long page_number, 
    pid_t pid) {
    struct uvm_dirty_page_table* page_table_pointer = uvm_dirty_page_table_by_pid(pid);
    if (page_table_pointer == NULL) {
        printk(KERN_ERR "Dirty page table not initialized for pid %d\n", pid);
        return NULL;
    }

    struct dirty_page_info *info = xa_load(&page_table_pointer->pages, page_number);
    if (info == NULL) {
        printk(KERN_INFO "No dirty page info found for page_number=0x%lx for pid %d\n", page_number, pid);
        return NULL;    
    }

    printk(KERN_INFO "Found dirty page info: page_number=0x%lx, timestamp=%lu for pid %d\n",
           info->page_number, info->timestamp, info->pid);
    return info;
}

// EDIT BY VIDHI JAIN
static ssize_t dirty_range_write(struct file *file,
                                 const char __user *buf,
                                 size_t count,
                                 loff_t *ppos)
{
    char kbuf[64];

    if(copy_from_user(kbuf, buf, min(count, sizeof(kbuf))))
        return -EFAULT;

    sscanf(kbuf, "%lx %lx", &dirty_query_start, &dirty_query_end);

    printk(KERN_INFO "DIRTY_RANGE set: 0x%lx - 0x%lx\n",
           dirty_query_start, dirty_query_end);

    return count;
}

static const struct proc_ops dirty_range_fops = {
    .proc_write = dirty_range_write,
};

// END OF EDIT

// EDIT BY ARUSH - procfs query interface
#if defined(CONFIG_PROC_FS)
// EDIT BY ADITI KHANDELIA
static pid_t dirty_query_pid = 0;
// END OF EDIT
static int nv_procfs_read_dirty_pages(struct seq_file *s, void *__v)
{
    unsigned long index;
    struct dirty_page_info *info;

    struct uvm_dirty_page_table* page_table_pointer = uvm_dirty_page_table_by_pid(dirty_query_pid);

    if (page_table_pointer == NULL) {
        seq_printf(s, "# dirty tracking not active for pid %d\n", dirty_query_pid);
        return 0;
    }

    seq_printf(s, "# page_address_hex timestamp_ns pid\n");

    // EDIT BY VIDHI JAIN

    unsigned long start_index = dirty_query_start >> PAGE_SHIFT;
    unsigned long end_index = (dirty_query_end - 1) >> PAGE_SHIFT;

    index = start_index;

    while((info = xa_find(&page_table_pointer->pages,
                          &index,
                          end_index,
                          XA_PRESENT))){
        seq_printf(s,
                   "0x%lx %lu %d\n",
                   index << PAGE_SHIFT,
                   info->timestamp,
                   info->pid);
        index++;
    }

    // END OF EDIT

    return 0;
}

static int nv_procfs_read_dirty_pages_entry(struct seq_file *s, void *v)
{
    return nv_procfs_read_dirty_pages(s, v);
}

UVM_DEFINE_SINGLE_PROCFS_FILE(dirty_pages_entry);


// EDIT BY ADITI KHANDELIA
static ssize_t dirty_pids_start_write(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {

    pid_t pid = current->tgid;
    uvm_dirty_page_table_init(pid);
    if (uvm_dirty_invalidate_fn) {
        uvm_dirty_invalidate_fn();
    }
    return count;
}

static const struct proc_ops dirty_pids_start_fops = {
    .proc_write = dirty_pids_start_write,
};

static ssize_t dirty_pids_stop_write(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {

    pid_t pid = current->tgid;
    uvm_dirty_page_table_destroy(pid);
    return count;
}

static const struct proc_ops dirty_pids_stop_fops = {
    .proc_write = dirty_pids_stop_write,
};

static ssize_t pid_to_query_for_dirty_tracking(struct file* file,
    const char __user* buf,
    size_t count,
    loff_t* ppos) {

    char kbuf[32];

    if (copy_from_user(kbuf, buf, min(count, sizeof(kbuf))))
        return -EFAULT;

    pid_t pid;
    sscanf(kbuf, "%d", &pid);
    dirty_query_pid = pid;

    printk(KERN_INFO "DIRTY_PIDS set: tracking dirty pages for pid %d\n", dirty_query_pid);
    return count;
}

static const struct proc_ops pid_to_query_fops = {
    .proc_write = pid_to_query_for_dirty_tracking,
};
// END OF EDIT

NV_STATUS uvm_dirty_procfs_init(struct proc_dir_entry *parent)
{
    struct proc_dir_entry *entry;

    entry = NV_CREATE_PROC_FILE("dirty_pages", parent, dirty_pages_entry, NULL);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;
    
    // EDIT BY VIDHI JAIN
    entry = proc_create("dirty_range",
                        0666,
                        parent,
                        &dirty_range_fops);
    if(entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;
    // END OF EDIT

    // EDIT BY ADITI KHANDELIA
    entry = proc_create("dirty_pids_start_track",
                        0666,
                        parent,
                        &dirty_pids_start_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    entry = proc_create("dirty_pids_stop_track",
                        0666,
                        parent,
                        &dirty_pids_stop_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;

    entry = proc_create("dirty_pid_to_query",
                        0666,
                        parent,
                        &pid_to_query_fops);
    if (entry == NULL)
        return NV_ERR_OPERATING_SYSTEM;
    
    return NV_OK;
}

#endif // CONFIG_PROC_FS
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
