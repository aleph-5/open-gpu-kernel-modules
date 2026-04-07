// EDIT BY VIDHI JAIN
/*******************************************************************************
    Nested-bitmap backend for the dirty-page data structure abstraction layer.

*******************************************************************************/

#include "uvm_dirty_ds.h"
#include "uvm_linux.h"
#include <linux/slab.h>
#include <linux/bitmap.h>
#include <linux/xarray.h>

// Number of 4 KB pages in one UVM VA block (UVM_VA_BLOCK_SIZE / PAGE_SIZE = 2 MB / 4 KB = 512).  Defined explicitly to avoid pulling in uvm_va_block.h
#define PAGES_PER_BLOCK 512UL
#define BLOCK_SHIFT 9UL  // log2(PAGES_PER_BLOCK)
#define BLOCK_MASK  (PAGES_PER_BLOCK - 1)

struct block_entry {
    unsigned long bitmap[BITS_TO_LONGS(PAGES_PER_BLOCK)];
    unsigned long timestamps[PAGES_PER_BLOCK];
};

struct nb_backend {
    struct xarray xa; // block_index (u64) -> struct block_entry
};


static struct block_entry *nb_get_or_alloc_block(struct nb_backend *b, unsigned long block_idx)
{
    struct block_entry *entry;
    struct block_entry *new_entry;
    void *old;

    // Fast path: block already exists (lockless RCU read)
    entry = xa_load(&b->xa, block_idx);
    if(entry) return entry;

    // Slow path: first dirty page in this block
    new_entry = kzalloc(sizeof(*new_entry), GFP_ATOMIC);
    if(!new_entry) return NULL;

    old = xa_cmpxchg(&b->xa, block_idx, NULL, new_entry, GFP_ATOMIC);

    if(xa_err(old)){
        kfree(new_entry);
        return NULL;
    }
    if(old != NULL){
        /* Lost the race — use the winner's entry. */
        kfree(new_entry);
        return old;
    }
    return new_entry;
}


static NV_STATUS nb_backend_init(struct uvm_dirty_ds *ds)
{
    struct nb_backend *b = kzalloc(sizeof(*b), GFP_KERNEL);
    if (!b) return NV_ERR_NO_MEMORY;
    xa_init(&b->xa);
    ds->priv = b;
    return NV_OK;
}

/*
 * Hot path — O(1), lock-free.
 * 1. Derive block_index and page_in_block.
 * 2. Get (or lazily allocate) the block_entry for this VA block.
 * 3. Atomically set the page's bit.  If the bit was already clear (first write to this page), record the timestamp.
 */
static NV_STATUS nb_backend_insert(struct uvm_dirty_ds *ds, unsigned long page_num, unsigned long timestamp)
{
    struct nb_backend *b = ds->priv;
    unsigned long block_idx = page_num >> BLOCK_SHIFT;
    unsigned long page_in_block = page_num &  BLOCK_MASK;
    struct block_entry *entry;

    entry = nb_get_or_alloc_block(b, block_idx);
    if(!entry) return NV_ERR_NO_MEMORY;

    /*
     * test_and_set_bit is atomic.  It returns the old value:
     *   0 → bit was clear — we are the first writer; record timestamp.
     *   1 → already dirty — first-write-wins; do nothing.
     */
    if(!test_and_set_bit(page_in_block, entry->bitmap)) WRITE_ONCE(entry->timestamps[page_in_block], timestamp);
    return NV_OK;
}

static struct dirty_page_info *nb_backend_lookup(struct uvm_dirty_ds *ds, unsigned long page_num)
{
    struct nb_backend *b = ds->priv;
    unsigned long block_idx = page_num >> BLOCK_SHIFT;
    unsigned long page_in_block = page_num &  BLOCK_MASK;
    struct block_entry *entry;

    entry = xa_load(&b->xa, block_idx);
    if(!entry) return NULL;
    return test_bit(page_in_block, entry->bitmap) ? (struct dirty_page_info *)entry : NULL;
}


static void nb_backend_for_each_in_range(struct uvm_dirty_ds *ds,
                                          unsigned long start_page,
                                          unsigned long end_page,
                                          uvm_dirty_ds_range_fn fn,
                                          void *ctx)
{
    struct nb_backend *b = ds->priv;
    unsigned long first_block = start_page >> BLOCK_SHIFT;
    unsigned long last_block  = end_page   >> BLOCK_SHIFT;
    unsigned long block_idx;
    struct block_entry *entry;

    block_idx = first_block;
    while((entry = xa_find(&b->xa, &block_idx, last_block, XA_PRESENT)) != NULL){
        unsigned long first_page_in_block, last_page_in_block, bit;

        first_page_in_block = (block_idx == first_block) ? (start_page & BLOCK_MASK) : 0;
        last_page_in_block = (block_idx == last_block) ? (end_page & BLOCK_MASK) : (PAGES_PER_BLOCK - 1);

        bit = first_page_in_block;
        
        while((bit = find_next_bit(entry->bitmap, last_page_in_block + 1, bit)) <= last_page_in_block){
            unsigned long abs_page = (block_idx << BLOCK_SHIFT) | bit;
            unsigned long ts = READ_ONCE(entry->timestamps[bit]);
            fn(abs_page, ts, ctx);
            bit++;
        }
        block_idx++;
    }
}

static void nb_backend_destroy(struct uvm_dirty_ds *ds)
{
    struct nb_backend *b = ds->priv;
    struct block_entry *entry;
    unsigned long block_idx;
    xa_for_each(&b->xa, block_idx, entry) kfree(entry);
    xa_destroy(&b->xa);
    kfree(b);
    ds->priv = NULL;
}

const struct uvm_dirty_ds_ops uvm_dirty_ds_nested_bitmap_ops = {
    .init              = nb_backend_init,
    .insert            = nb_backend_insert,
    .lookup            = nb_backend_lookup,
    .for_each_in_range = nb_backend_for_each_in_range,
    .destroy           = nb_backend_destroy,
};
// END OF EDIT