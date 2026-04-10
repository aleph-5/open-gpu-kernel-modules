/*******************************************************************************
    Dirty-page data structure abstraction layer.

    Defines a generic ops-table interface (init / insert / lookup /
    for_each_in_range / destroy).
*******************************************************************************/

#ifndef __UVM_DIRTY_DS_H__
#define __UVM_DIRTY_DS_H__

// EDIT BY KUSHAGRA
#include "uvm_common.h"
#include <linux/atomic.h>
#include <linux/ktime.h>

struct uvm_dirty_ds;

struct uvm_dirty_ds_stats {
    bool           enabled;
    atomic_long_t  init_count;
    atomic64_t     init_time_ns;
    atomic_long_t  insert_count;
    atomic64_t     insert_time_ns;
    atomic_long_t  lookup_count;
    atomic64_t     lookup_time_ns;
    atomic_long_t  for_each_in_range_count;
    atomic64_t     for_each_in_range_time_ns;
    atomic_long_t  destroy_count;
    atomic64_t     destroy_time_ns;
    // EDIT BY VIDHI JAIN
    /* time spent blocked in down_read/down_write before each op */
    atomic64_t     insert_lock_wait_ns;
    atomic64_t     lookup_lock_wait_ns;
    atomic64_t     for_each_lock_wait_ns;
    // END OF EDIT
    // EDIT BY SANKALP MITTAL
    atomic_long_t  invalidate_count;
    atomic64_t     invalidate_time_ns;
    atomic_long_t  activate_count;
    atomic64_t     activate_time_ns;
    atomic_long_t  barrier_end_count;
    atomic64_t     barrier_end_time_ns;
    // END OF EDIT
};

#define DIRTY_DS_STATS_READ(ds, field)   atomic_long_read(&(ds)->stats.field)
#define DIRTY_DS_STATS_READ64(ds, field) atomic64_read(&(ds)->stats.field)

typedef void (*uvm_dirty_ds_range_fn)(unsigned long page_num,
                                      unsigned long timestamp,
                                      void *ctx);

// Operations Table
struct uvm_dirty_ds_ops {
    NV_STATUS (*init)(struct uvm_dirty_ds *ds);

    NV_STATUS (*insert)(struct uvm_dirty_ds *ds,
                        unsigned long page_num,
                        unsigned long timestamp);

    struct dirty_page_info *(*lookup)(struct uvm_dirty_ds *ds,
                                      unsigned long page_num);


    void (*for_each_in_range)(struct uvm_dirty_ds *ds,
                               unsigned long start_page,
                               unsigned long end_page,
                               uvm_dirty_ds_range_fn fn,
                               void *ctx);

    void (*destroy)(struct uvm_dirty_ds *ds);
};

struct uvm_dirty_ds {
    const struct uvm_dirty_ds_ops *ops;
    void                          *priv;     /* backend-private; NULL when not initialized */
    struct uvm_dirty_ds_stats      stats;    /* call counters; zero-initialised; enable with stats.enabled = true */
};


static inline NV_STATUS uvm_dirty_ds_init(struct uvm_dirty_ds *ds)
{
    if (unlikely(ds->stats.enabled)) {
        ktime_t t0 = ktime_get();
        NV_STATUS ret = ds->ops->init(ds);
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)), &ds->stats.init_time_ns);
        atomic_long_inc(&ds->stats.init_count);
        return ret;
    }
    return ds->ops->init(ds);
}

static inline NV_STATUS uvm_dirty_ds_insert(struct uvm_dirty_ds *ds,
                                              unsigned long page_num,
                                              unsigned long timestamp)
{
    if (unlikely(ds->stats.enabled)) {
        ktime_t t0 = ktime_get();
        NV_STATUS ret = ds->ops->insert(ds, page_num, timestamp);
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)), &ds->stats.insert_time_ns);
        atomic_long_inc(&ds->stats.insert_count);
        return ret;
    }
    return ds->ops->insert(ds, page_num, timestamp);
}

static inline struct dirty_page_info *uvm_dirty_ds_lookup(struct uvm_dirty_ds *ds,
                                                            unsigned long page_num)
{
    if (unlikely(ds->stats.enabled)) {
        ktime_t t0 = ktime_get();
        struct dirty_page_info *ret = ds->ops->lookup(ds, page_num);
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)), &ds->stats.lookup_time_ns);
        atomic_long_inc(&ds->stats.lookup_count);
        return ret;
    }
    return ds->ops->lookup(ds, page_num);
}

static inline void uvm_dirty_ds_for_each_in_range(struct uvm_dirty_ds *ds,
                                                    unsigned long start_page,
                                                    unsigned long end_page,
                                                    uvm_dirty_ds_range_fn fn,
                                                    void *ctx)
{
    if (unlikely(ds->stats.enabled)) {
        ktime_t t0 = ktime_get();
        ds->ops->for_each_in_range(ds, start_page, end_page, fn, ctx);
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)), &ds->stats.for_each_in_range_time_ns);
        atomic_long_inc(&ds->stats.for_each_in_range_count);
        return;
    }
    ds->ops->for_each_in_range(ds, start_page, end_page, fn, ctx);
}

static inline void uvm_dirty_ds_destroy(struct uvm_dirty_ds *ds)
{
    if (unlikely(ds->stats.enabled)) {
        ktime_t t0 = ktime_get();
        ds->ops->destroy(ds);
        atomic64_add(ktime_to_ns(ktime_sub(ktime_get(), t0)), &ds->stats.destroy_time_ns);
        atomic_long_inc(&ds->stats.destroy_count);
        return;
    }
    ds->ops->destroy(ds);
}

/* xarray backend – defined in uvm_dirty_ds_xarray.c */
extern const struct uvm_dirty_ds_ops uvm_dirty_ds_xarray_ops;

/* bitmap backend – defined in uvm_dirty_ds_bitmap.c */
extern const struct uvm_dirty_ds_ops uvm_dirty_ds_bitmap_ops;
// END OF EDIT

// EDIT BY VIDHI JAIN

/* vector backend – defined in uvm_dirty_ds_vector.c */
extern const struct uvm_dirty_ds_ops uvm_dirty_ds_vector_ops;
/* unsorted vector backend – defined in uvm_dirty_ds_vector_unsorted.c */
extern const struct uvm_dirty_ds_ops uvm_dirty_ds_vector_unsorted_ops;
/* chunked vector backend – defined in uvm_dirty_ds_chunked.c */
extern const struct uvm_dirty_ds_ops uvm_dirty_ds_chunked_ops;
/* nested bitmap backend – defined in uvm_dirty_ds_nested_bitmap.c */
extern const struct uvm_dirty_ds_ops uvm_dirty_ds_nested_bitmap_ops;

// END OF EDIT

// EDIT BY SANKALP MITTAL
extern const struct uvm_dirty_ds_ops uvm_dirty_ds_linked_list_ops;
// END OF EDIT


#endif /* __UVM_DIRTY_DS_H__ */
