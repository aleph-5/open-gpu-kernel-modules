// EDIT BY ARUSH
/*******************************************************************************
    Bitmap backend for the dirty-page data structure abstraction layer.

    Implements uvm_dirty_ds_ops using a flat vzalloc'd bitmap.

    Insert is a single set_bit() — no allocation, no prior lookup.
    Timestamps are accepted by the interface but not stored.

    Covers up to DIRTY_BITMAP_MAX_PAGES pages (128 GB VA at PAGE_SIZE = 4 KB).
*******************************************************************************/

#include "uvm_dirty_ds.h"
#include "uvm_linux.h"
#include <linux/vmalloc.h>
#include <linux/bitmap.h>

#define DIRTY_BITMAP_MAX_PAGES  (1UL << 38)   /* 2^38 pages = 246 TB VA, 32 MB bitmap */

/*
 * Sentinel returned by bitmap_backend_lookup when a page is dirty.
 * Callers only test NULL vs non-NULL; timestamp and page_number are unused.
 */
static struct dirty_page_info bitmap_lookup_sentinel = { .page_number = 0, .timestamp = 0 };

static NV_STATUS bitmap_backend_init(struct uvm_dirty_ds *ds)
{
    unsigned long *bits = vzalloc(BITS_TO_LONGS(DIRTY_BITMAP_MAX_PAGES) * sizeof(unsigned long));

    if (!bits)
        return NV_ERR_NO_MEMORY;

    ds->priv = bits;
    return NV_OK;
}

/*
 * Hot path — called from fault-servicing context.
 * set_bit is atomic; timestamp is ignored (bitmap stores only presence).
 */
static NV_STATUS bitmap_backend_insert(struct uvm_dirty_ds *ds,
                                       unsigned long page_num,
                                       unsigned long timestamp)
{
    if (page_num >= DIRTY_BITMAP_MAX_PAGES)
        return NV_ERR_INVALID_ARGUMENT;
        
    set_bit(page_num, (unsigned long *)ds->priv);
    return NV_OK;
}

/*
 * Returns a non-NULL sentinel when the page is dirty, NULL otherwise.
 * Callers treat the return value as a presence indicator only.
 */
static struct dirty_page_info *bitmap_backend_lookup(struct uvm_dirty_ds *ds,
                                                     unsigned long page_num)
{
    if (page_num >= DIRTY_BITMAP_MAX_PAGES)
        return NULL;

    return test_bit(page_num, (unsigned long *)ds->priv) ? &bitmap_lookup_sentinel : NULL;
}

static void bitmap_backend_for_each_in_range(struct uvm_dirty_ds *ds,
                                             unsigned long start_page,
                                             unsigned long end_page,
                                             uvm_dirty_ds_range_fn fn,
                                             void *ctx)
{
    unsigned long *bits = ds->priv;
    unsigned long bound = min(end_page + 1, DIRTY_BITMAP_MAX_PAGES);
    unsigned long bit = start_page;

    while ((bit = find_next_bit(bits, bound, bit)) < bound) {
        fn(bit, 0 /* timestamp not stored */, ctx);
        bit++;
    }
}

static void bitmap_backend_destroy(struct uvm_dirty_ds *ds)
{
    vfree(ds->priv);
    ds->priv = NULL;
}

const struct uvm_dirty_ds_ops uvm_dirty_ds_bitmap_ops = {
    .init              = bitmap_backend_init,
    .insert            = bitmap_backend_insert,
    .lookup            = bitmap_backend_lookup,
    .for_each_in_range = bitmap_backend_for_each_in_range,
    .destroy           = bitmap_backend_destroy,
};
