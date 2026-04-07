// EDIT BY VIDHI JAIN
/*******************************************************************************
    Vector backend for the dirty-page data structure abstraction layer.

    Implements uvm_dirty_ds_ops using a sorted dynamic array of pointers to heap-allocated dirty_page_info structs.
    The array is kept sorted ascending by page_number at all times.
    
*******************************************************************************/

#include "uvm_dirty_ds.h"
#include "uvm_linux.h"
#include <linux/slab.h>

#include <linux/vmalloc.h>
// #include <linux/spinlock.h>
#include <linux/string.h>

#define VEC_INITIAL_CAPACITY  64u

struct vec_backend {
    struct dirty_page_info **entries;  /* sorted ascending by page_number */
    size_t                   count;
    size_t                   capacity;
    // spinlock_t               lock;
};


// (standard lower_bound)
static size_t vec_lower_bound(struct dirty_page_info **entries,
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

static int vec_grow(struct vec_backend *v)
{
    size_t new_cap = v->capacity * 2;
    struct dirty_page_info **new_buf;

    new_buf = kvmalloc_array(new_cap, sizeof(struct dirty_page_info *),
                             GFP_ATOMIC);
    if (!new_buf)
        return -ENOMEM;

    memcpy(new_buf, v->entries,
           v->count * sizeof(struct dirty_page_info *));
    kvfree(v->entries);
    v->entries  = new_buf;
    v->capacity = new_cap;
    return 0;
}

static NV_STATUS vec_backend_init(struct uvm_dirty_ds *ds)
{
    struct vec_backend *v = kzalloc(sizeof(*v), GFP_KERNEL);

    if (!v)
        return NV_ERR_NO_MEMORY;

    v->entries = kvmalloc_array(VEC_INITIAL_CAPACITY,
                                sizeof(struct dirty_page_info *),
                                GFP_KERNEL);
    if (!v->entries) {
        kfree(v);
        return NV_ERR_NO_MEMORY;
    }

    v->capacity = VEC_INITIAL_CAPACITY;
    v->count    = 0;
    // spin_lock_init(&v->lock);
    ds->priv = v;
    return NV_OK;
}

/*
 * Hot path — called from fault-servicing context.
 * First-write-wins: if the page is already present its timestamp is kept.
 */
static NV_STATUS vec_backend_insert(struct uvm_dirty_ds *ds,
                                     unsigned long page_num,
                                     unsigned long timestamp)
{
    struct vec_backend *v = ds->priv;
    struct dirty_page_info *info;
    // unsigned long flags;
    size_t pos;

    info = kmalloc(sizeof(*info), GFP_ATOMIC);
    if (!info)
        return NV_ERR_NO_MEMORY;
    info->page_number = page_num;
    info->timestamp   = timestamp;

    // spin_lock_irqsave(&v->lock, flags);

    pos = vec_lower_bound(v->entries, v->count, page_num);

    /* Duplicate: page already recorded — first-write-wins. */
    if (pos < v->count && v->entries[pos]->page_number == page_num) {
        // spin_unlock_irqrestore(&v->lock, flags);
        kfree(info);
        return NV_OK;
    }

    /* Grow the pointer array if at capacity. */
    if (v->count == v->capacity) {
        if (vec_grow(v) != 0) {
            // spin_unlock_irqrestore(&v->lock, flags);
            kfree(info);
            return NV_ERR_NO_MEMORY;
        }
    }

    /* Shift existing pointers right to open a slot at pos. */
    if (pos < v->count)
        memmove(&v->entries[pos + 1], &v->entries[pos],
                (v->count - pos) * sizeof(struct dirty_page_info *));

    v->entries[pos] = info;
    v->count++;

    // spin_unlock_irqrestore(&v->lock, flags);
    return NV_OK;
}

static struct dirty_page_info *vec_backend_lookup(struct uvm_dirty_ds *ds,
                                                    unsigned long page_num)
{
    struct vec_backend *v = ds->priv;
    // unsigned long flags;
    size_t pos;
    struct dirty_page_info *result = NULL;

    // spin_lock_irqsave(&v->lock, flags);
    pos = vec_lower_bound(v->entries, v->count, page_num);
    if (pos < v->count && v->entries[pos]->page_number == page_num)
        result = v->entries[pos];
    // spin_unlock_irqrestore(&v->lock, flags);

    return result;
}

// Binary search to skip to start_page, then a sequential pointer walk.
static void vec_backend_for_each_in_range(struct uvm_dirty_ds *ds,
                                           unsigned long start_page,
                                           unsigned long end_page,
                                           uvm_dirty_ds_range_fn fn,
                                           void *ctx)
{
    struct vec_backend *v = ds->priv;
    // unsigned long flags;
    size_t i, start_idx;

    // spin_lock_irqsave(&v->lock, flags);
    start_idx = vec_lower_bound(v->entries, v->count, start_page);
    for (i = start_idx;
         i < v->count && v->entries[i]->page_number <= end_page;
         i++)
        fn(v->entries[i]->page_number, v->entries[i]->timestamp, ctx);
    // spin_unlock_irqrestore(&v->lock, flags);
}

static void vec_backend_destroy(struct uvm_dirty_ds *ds)
{
    struct vec_backend *v = ds->priv;
    size_t i;

    for (i = 0; i < v->count; i++)
        kfree(v->entries[i]);

    kvfree(v->entries);
    kfree(v);
    ds->priv = NULL;
}

const struct uvm_dirty_ds_ops uvm_dirty_ds_vector_ops = {
    .init              = vec_backend_init,
    .insert            = vec_backend_insert,
    .lookup            = vec_backend_lookup,
    .for_each_in_range = vec_backend_for_each_in_range,
    .destroy           = vec_backend_destroy,
};

// END OF EDIT