// EDIT BY VIDHI JAIN
/*******************************************************************************
    Chunked-vector backend for the dirty-page data structure abstraction layer.
    
    Splits the array into a sequence of independently-sorted chunks and defers the sort of each chunk to a background kthread.

*******************************************************************************/

#include "uvm_dirty_ds.h"
#include "uvm_linux.h"
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/string.h>
#include <linux/sort.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>

#define INITIAL_CHUNK_BYTES   4096u
#define INITIAL_CHUNK_ENTRIES (INITIAL_CHUNK_BYTES / sizeof(struct dirty_page_info *))

#define MAX_CHUNKS 64u

/* How long the sorter sleeps between forced wakeups (ms). */
#define SORTER_POLL_MS 500u

struct chunk {
    struct dirty_page_info **entries;
    size_t                   count;
    size_t                   capacity;
    /*
     * Set to true by the kthread after the chunk has been sorted.
     * Written with smp_store_release, read with smp_load_acquire so that
     * the sorted contents are visible to query callers without a full lock.
     */
    bool                     sorted;
};

struct chunked_backend {
    // Fixed-size pointer table pre-allocated at init. Slots beyond num_chunks are NULL
    struct chunk  *chunks[MAX_CHUNKS];
    size_t         num_chunks;

    spinlock_t     lock;

    /* Background sorter */
    struct task_struct *sorter;
    wait_queue_head_t   sort_wq;
    atomic_t            needs_sort;   // non-zero → a chunk was sealed
};


static size_t chunk_lower_bound(struct dirty_page_info **entries,
                                 size_t count, unsigned long target)
{
    size_t lo = 0, hi = count;

    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;

        if (entries[mid]->page_number < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/* Comparison function for kernel sort(). */
static int cmp_page_info_ptr(const void *a, const void *b)
{
    const struct dirty_page_info *pa = *(const struct dirty_page_info **)a;
    const struct dirty_page_info *pb = *(const struct dirty_page_info **)b;

    if (pa->page_number < pb->page_number) return -1;
    if (pa->page_number > pb->page_number) return  1;
    return 0;
}

static size_t next_chunk_capacity(const struct chunked_backend *b)
{
    return b->chunks[b->num_chunks - 1]->capacity * 2;
}


static struct chunk *alloc_chunk(size_t capacity)
{
    struct chunk *c = kzalloc(sizeof(*c), GFP_ATOMIC);

    if (!c)
        return NULL;

    c->entries = kvmalloc_array(capacity, sizeof(struct dirty_page_info *),
                                GFP_ATOMIC);
    if (!c->entries) {
        kfree(c);
        return NULL;
    }

    c->capacity = capacity;
    c->count    = 0;
    c->sorted   = false;
    return c;
}

static struct dirty_page_info *
search_completed_chunks(const struct chunked_backend *b,
                         unsigned long page_num, size_t num_chunks)
{
    size_t i;

    /* All chunks except the last are completed/immutable. */
    for (i = 0; i + 1 < num_chunks; i++) {
        const struct chunk *c = b->chunks[i];

        if (smp_load_acquire(&c->sorted)) {
            /* Fast path: binary search. */
            size_t pos = chunk_lower_bound(c->entries, c->count, page_num);

            if (pos < c->count && c->entries[pos]->page_number == page_num)
                return c->entries[pos];
        } else {
            /* Kthread hasn't sorted this chunk yet — linear scan. */
            size_t j;

            for (j = 0; j < c->count; j++) {
                if (c->entries[j]->page_number == page_num)
                    return c->entries[j];
            }
        }
    }
    return NULL;
}

static struct dirty_page_info *
search_active_chunk(const struct chunked_backend *b, unsigned long page_num)
{
    const struct chunk *cur = b->chunks[b->num_chunks - 1];
    size_t i;

    for (i = 0; i < cur->count; i++) {
        if (cur->entries[i]->page_number == page_num)
            return cur->entries[i];
    }
    return NULL;
}

static void sort_completed_chunks(struct chunked_backend *b)
{
    size_t i, n;
    spin_lock_irq(&b->lock);
    n = b->num_chunks;
    spin_unlock_irq(&b->lock);

    if (n == 0)
        return;

    for (i = 0; i < n - 1; i++) {
        struct chunk *c = b->chunks[i];

        if (smp_load_acquire(&c->sorted))
            continue;   /* already sorted */

        sort(c->entries, c->count,
             sizeof(struct dirty_page_info *),
             cmp_page_info_ptr, NULL);

        /* Publish sorted contents before setting the flag. */
        smp_store_release(&c->sorted, true);
    }
}

static int chunked_sorter_thread(void *data)
{
    struct chunked_backend *b = data;

    while (!kthread_should_stop()) {
        wait_event_interruptible_timeout(
            b->sort_wq,
            atomic_read(&b->needs_sort) || kthread_should_stop(),
            msecs_to_jiffies(SORTER_POLL_MS));

        if (kthread_should_stop())
            break;

        atomic_set(&b->needs_sort, 0);
        sort_completed_chunks(b);
    }
    return 0;
}


static NV_STATUS chunked_backend_init(struct uvm_dirty_ds *ds)
{
    struct chunked_backend *b;
    struct chunk *first;

    b = kzalloc(sizeof(*b), GFP_KERNEL);
    if (!b)
        return NV_ERR_NO_MEMORY;

    spin_lock_init(&b->lock);
    init_waitqueue_head(&b->sort_wq);
    atomic_set(&b->needs_sort, 0);

    /* Allocate the first chunk. */
    first = alloc_chunk(INITIAL_CHUNK_ENTRIES);
    if (!first) {
        kfree(b);
        return NV_ERR_NO_MEMORY;
    }

    b->chunks[0] = first;
    b->num_chunks = 1;

    /* Start the background sorter. */
    b->sorter = kthread_run(chunked_sorter_thread, b,
                             "uvm_dirty_sorter");
    if (IS_ERR(b->sorter)) {
        kvfree(first->entries);
        kfree(first);
        kfree(b);
        return NV_ERR_OPERATING_SYSTEM;
    }

    ds->priv = b;
    return NV_OK;
}

/*
 * Hot path — called from fault-servicing context.
 *
 * 1. Allocate a dirty_page_info upfront (we may not need it if duplicate).
 * 2. Check completed chunks (no lock needed — they are immutable).
 * 3. Acquire lock, check active chunk, then append.
 * 4. If active chunk is now full, seal it, allocate a new one, signal kthread.
 */
static NV_STATUS chunked_backend_insert(struct uvm_dirty_ds *ds,
                                         unsigned long page_num,
                                         unsigned long timestamp)
{
    struct chunked_backend *b = ds->priv;
    struct dirty_page_info *info;
    struct chunk *cur;
    unsigned long flags;
    size_t snap_num_chunks;

    info = kmalloc(sizeof(*info), GFP_ATOMIC);
    if (!info)
        return NV_ERR_NO_MEMORY;
    info->page_number = page_num;
    info->timestamp   = timestamp;

    /*
     * Snapshot num_chunks before taking the lock to check completed chunks.
     * It's fine if num_chunks grows between the snapshot and the lock
     * acquisition — the new chunk will be the active chunk and we check
     * that under the lock anyway.
     */
    snap_num_chunks = READ_ONCE(b->num_chunks);

    if (search_completed_chunks(b, page_num, snap_num_chunks)) {
        kfree(info);
        return NV_OK;
    }

    spin_lock_irqsave(&b->lock, flags);

    if (search_active_chunk(b, page_num)) {
        spin_unlock_irqrestore(&b->lock, flags);
        kfree(info);
        return NV_OK;
    }

    cur = b->chunks[b->num_chunks - 1];

    if (cur->count == cur->capacity) {
        size_t new_cap;
        struct chunk *next;

        if (b->num_chunks >= MAX_CHUNKS) {
            spin_unlock_irqrestore(&b->lock, flags);
            kfree(info);
            return NV_ERR_NO_MEMORY;
        }

        new_cap = next_chunk_capacity(b);
        next    = alloc_chunk(new_cap);
        if (!next) {
            spin_unlock_irqrestore(&b->lock, flags);
            kfree(info);
            return NV_ERR_NO_MEMORY;
        }

        /* Publish new chunk before incrementing num_chunks. */
        b->chunks[b->num_chunks] = next;
        smp_wmb();
        WRITE_ONCE(b->num_chunks, b->num_chunks + 1);

        /* Signal the kthread that a chunk was sealed. */
        atomic_set(&b->needs_sort, 1);
        wake_up_interruptible(&b->sort_wq);

        cur = next;
    }

    /* Append to active chunk — O(1), no shifting. */
    cur->entries[cur->count++] = info;

    spin_unlock_irqrestore(&b->lock, flags);
    return NV_OK;
}

static struct dirty_page_info *chunked_backend_lookup(struct uvm_dirty_ds *ds,
                                                        unsigned long page_num)
{
    struct chunked_backend *b = ds->priv;
    struct dirty_page_info *result;
    unsigned long flags;
    size_t n;

    n = READ_ONCE(b->num_chunks);
    if (n == 0)
        return NULL;

    /* Search sorted completed chunks without the lock. */
    result = search_completed_chunks(b, page_num, n);
    if (result)
        return result;

    /* Search active chunk under the lock. */
    spin_lock_irqsave(&b->lock, flags);
    result = search_active_chunk(b, page_num);
    spin_unlock_irqrestore(&b->lock, flags);

    return result;
}

static void chunked_backend_for_each_in_range(struct uvm_dirty_ds *ds,
                                               unsigned long start_page,
                                               unsigned long end_page,
                                               uvm_dirty_ds_range_fn fn,
                                               void *ctx)
{
    struct chunked_backend *b = ds->priv;
    unsigned long flags;
    size_t i, n;

    n = READ_ONCE(b->num_chunks);
    if (n == 0)
        return;

    /* Walk completed chunks (no lock needed). */
    for (i = 0; i + 1 < n; i++) {
        const struct chunk *c = b->chunks[i];

        if (smp_load_acquire(&c->sorted)) {
            /* Binary search to start, then sequential walk. */
            size_t j = chunk_lower_bound(c->entries, c->count, start_page);

            for (; j < c->count && c->entries[j]->page_number <= end_page; j++)
                fn(c->entries[j]->page_number, c->entries[j]->timestamp, ctx);
        } else {
            /* Not yet sorted — full linear scan with range filter. */
            size_t j;

            for (j = 0; j < c->count; j++) {
                unsigned long pn = c->entries[j]->page_number;

                if (pn >= start_page && pn <= end_page)
                    fn(pn, c->entries[j]->timestamp, ctx);
            }
        }
    }

    /* Walk active chunk under lock. */
    spin_lock_irqsave(&b->lock, flags);
    {
        const struct chunk *cur = b->chunks[b->num_chunks - 1];
        size_t j;

        for (j = 0; j < cur->count; j++) {
            unsigned long pn = cur->entries[j]->page_number;

            if (pn >= start_page && pn <= end_page)
                fn(pn, cur->entries[j]->timestamp, ctx);
        }
    }
    spin_unlock_irqrestore(&b->lock, flags);
}

static void chunked_backend_destroy(struct uvm_dirty_ds *ds)
{
    struct chunked_backend *b = ds->priv;
    size_t i, j;

    /* Stop the kthread first. */
    if (b->sorter) {
        kthread_stop(b->sorter);
        b->sorter = NULL;
    }

    for (i = 0; i < b->num_chunks; i++) {
        struct chunk *c = b->chunks[i];

        for (j = 0; j < c->count; j++)
            kfree(c->entries[j]);

        kvfree(c->entries);
        kfree(c);
    }

    kfree(b);
    ds->priv = NULL;
}

const struct uvm_dirty_ds_ops uvm_dirty_ds_chunked_ops = {
    .init              = chunked_backend_init,
    .insert            = chunked_backend_insert,
    .lookup            = chunked_backend_lookup,
    .for_each_in_range = chunked_backend_for_each_in_range,
    .destroy           = chunked_backend_destroy,
};

// END OF EDIT