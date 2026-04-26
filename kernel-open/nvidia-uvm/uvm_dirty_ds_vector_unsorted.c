/*******************************************************************************
    Unsorted vector backend for the dirty-page data structure abstraction layer.

*******************************************************************************/

#include "uvm_dirty_ds.h"
#include "uvm_linux.h"
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/spinlock.h>
#include <linux/string.h>

#define VEC_UNSORTED_INITIAL_CAPACITY  64u

struct vec_unsorted_backend {
    struct dirty_page_info **entries;  /* unsorted; appended in fault order */
    size_t                   count;
    size_t                   capacity;
    spinlock_t               lock;
};

static int vec_unsorted_grow(struct vec_unsorted_backend *v)
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

static NV_STATUS vec_unsorted_backend_init(struct uvm_dirty_ds *ds)
{
    struct vec_unsorted_backend *v = kzalloc(sizeof(*v), GFP_KERNEL);

    if (!v)
        return NV_ERR_NO_MEMORY;

    v->entries = kvmalloc_array(VEC_UNSORTED_INITIAL_CAPACITY,
                                sizeof(struct dirty_page_info *),
                                GFP_KERNEL);
    if (!v->entries) {
        kfree(v);
        return NV_ERR_NO_MEMORY;
    }

    v->capacity = VEC_UNSORTED_INITIAL_CAPACITY;
    v->count    = 0;
    spin_lock_init(&v->lock);
    ds->priv = v;
    return NV_OK;
}

static NV_STATUS vec_unsorted_backend_insert(struct uvm_dirty_ds *ds,
                                              unsigned long page_num,
                                              unsigned long timestamp)
{
    struct vec_unsorted_backend *v = ds->priv;
    struct dirty_page_info *info;
    unsigned long flags;
    size_t i;

    info = kmalloc(sizeof(*info), GFP_ATOMIC);
    if (!info)
        return NV_ERR_NO_MEMORY;
    info->page_number = page_num;
    info->timestamp   = timestamp;

    spin_lock_irqsave(&v->lock, flags);

    /* Dedup: skip if page already recorded — first-write-wins. */
    for (i = 0; i < v->count; i++) {
        if (v->entries[i]->page_number == page_num) {
            spin_unlock_irqrestore(&v->lock, flags);
            kfree(info);
            return NV_OK;
        }
    }

    /* Grow if at capacity. */
    if (v->count == v->capacity) {
        if (vec_unsorted_grow(v) != 0) {
            spin_unlock_irqrestore(&v->lock, flags);
            kfree(info);
            return NV_ERR_NO_MEMORY;
        }
    }

    /* Append to tail — O(1). */
    v->entries[v->count++] = info;

    spin_unlock_irqrestore(&v->lock, flags);
    return NV_OK;
}

static struct dirty_page_info *vec_unsorted_backend_lookup(
        struct uvm_dirty_ds *ds, unsigned long page_num)
{
    struct vec_unsorted_backend *v = ds->priv;
    unsigned long flags;
    size_t i;
    struct dirty_page_info *result = NULL;

    spin_lock_irqsave(&v->lock, flags);
    for (i = 0; i < v->count; i++) {
        if (v->entries[i]->page_number == page_num) {
            result = v->entries[i];
            break;
        }
    }
    spin_unlock_irqrestore(&v->lock, flags);

    return result;
}


static void vec_unsorted_backend_for_each_in_range(struct uvm_dirty_ds *ds,
                                                    unsigned long start_page,
                                                    unsigned long end_page,
                                                    uvm_dirty_ds_range_fn fn,
                                                    void *ctx)
{
    struct vec_unsorted_backend *v = ds->priv;
    unsigned long flags;
    size_t i;

    spin_lock_irqsave(&v->lock, flags);
    for (i = 0; i < v->count; i++) {
        unsigned long pn = v->entries[i]->page_number;

        if (pn >= start_page && pn <= end_page)
            fn(pn, v->entries[i]->timestamp, ctx);
    }
    spin_unlock_irqrestore(&v->lock, flags);
}

static void vec_unsorted_backend_destroy(struct uvm_dirty_ds *ds)
{
    struct vec_unsorted_backend *v = ds->priv;
    size_t i;

    for (i = 0; i < v->count; i++)
        kfree(v->entries[i]);

    kvfree(v->entries);
    kfree(v);
    ds->priv = NULL;
}

const struct uvm_dirty_ds_ops uvm_dirty_ds_vector_unsorted_ops = {
    .init              = vec_unsorted_backend_init,
    .insert            = vec_unsorted_backend_insert,
    .lookup            = vec_unsorted_backend_lookup,
    .for_each_in_range = vec_unsorted_backend_for_each_in_range,
    .destroy           = vec_unsorted_backend_destroy,
};

