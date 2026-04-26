/*******************************************************************************
    xarray backend for the dirty-page data structure abstraction layer.

    Implements uvm_dirty_ds_ops using Linux's xarray.
*******************************************************************************/

#include "uvm_dirty_ds.h"
#include "uvm_linux.h"

static NV_STATUS xa_backend_init(struct uvm_dirty_ds *ds)
{
    struct xarray *xa = kmalloc(sizeof(struct xarray), GFP_KERNEL);

    if (!xa)
        return NV_ERR_NO_MEMORY;
    xa_init(xa);
    ds->priv = xa;
    return NV_OK;
}

static NV_STATUS xa_backend_insert(struct uvm_dirty_ds *ds,
                                    unsigned long page_num,
                                    unsigned long timestamp)
{
    struct xarray *xa = ds->priv;
    struct dirty_page_info *info;
    void *ret;

    /* Skip allocation if page is already recorded. */
    if (xa_load(xa, page_num) != NULL)
        return NV_OK;

    info = kmalloc(sizeof(struct dirty_page_info), GFP_ATOMIC);
    if (!info)
        return NV_ERR_NO_MEMORY;

    info->page_number = page_num;
    info->timestamp   = timestamp;

    /* Atomic insert: only succeeds if slot is still empty. */
    ret = xa_cmpxchg(xa, page_num, NULL, info, GFP_ATOMIC);
    if (xa_err(ret)) {
        kfree(info);
        return NV_ERR_NO_MEMORY;
    }
    if (ret != NULL)    /* another thread won the race – entry already present */
        kfree(info);

    return NV_OK;
}

static struct dirty_page_info *xa_backend_lookup(struct uvm_dirty_ds *ds,
                                                   unsigned long page_num)
{
    return xa_load((struct xarray *)ds->priv, page_num);
}

static void xa_backend_for_each_in_range(struct uvm_dirty_ds *ds,
                                          unsigned long start_page,
                                          unsigned long end_page,
                                          uvm_dirty_ds_range_fn fn,
                                          void *ctx)
{
    struct xarray *xa = ds->priv;
    struct dirty_page_info *info;
    unsigned long index = start_page;

    while ((info = xa_find(xa, &index, end_page, XA_PRESENT))) {
        fn(index, info->timestamp, ctx);
        index++;
    }
}

static void xa_backend_destroy(struct uvm_dirty_ds *ds)
{
    struct xarray *xa = ds->priv;
    struct dirty_page_info *info;
    unsigned long index;

    xa_for_each(xa, index, info)
        kfree(info);

    xa_destroy(xa);
    kfree(xa);
    ds->priv = NULL;
}

const struct uvm_dirty_ds_ops uvm_dirty_ds_xarray_ops = {
    .init              = xa_backend_init,
    .insert            = xa_backend_insert,
    .lookup            = xa_backend_lookup,
    .for_each_in_range = xa_backend_for_each_in_range,
    .destroy           = xa_backend_destroy,
};
