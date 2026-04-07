// EDIT BY SANKALP MITTAL
/*******************************************************************************
    Linked List backend for the dirty-page data structure abstraction layer.

    Implements uvm_dirty_ds_ops using a singly-linked list with a sentinel
    head node.
*******************************************************************************/

#include "uvm_dirty_ds.h"
#include "uvm_linux.h"

struct dirty_ll_node {
    struct dirty_page_info *info;
    struct dirty_ll_node   *next;
};

struct dirty_linked_list {
    struct dirty_ll_node *head;  /* sentinel – info is NULL */
    struct dirty_ll_node *tail;
};

static NV_STATUS ll_backend_init(struct uvm_dirty_ds *ds)
{
    struct dirty_linked_list *ll;
    struct dirty_ll_node     *sentinel;

    ll = kmalloc(sizeof(struct dirty_linked_list), GFP_KERNEL);
    if (!ll)
        return NV_ERR_NO_MEMORY;

    sentinel = kmalloc(sizeof(struct dirty_ll_node), GFP_KERNEL);
    if (!sentinel) {
        kfree(ll);
        return NV_ERR_NO_MEMORY;
    }

    sentinel->info = NULL;
    sentinel->next = NULL;
    ll->head = sentinel;
    ll->tail = sentinel;
    ds->priv = ll;
    return NV_OK;
}

static NV_STATUS ll_backend_insert(struct uvm_dirty_ds *ds,
                                    unsigned long page_num,
                                    unsigned long timestamp)
{
    struct dirty_linked_list *ll = ds->priv;
    struct dirty_ll_node     *cur;
    struct dirty_ll_node     *node;
    struct dirty_page_info   *info;

    /* Skip if page is already recorded. */
    for (cur = ll->head->next; cur != NULL; cur = cur->next) {
        if (cur->info->page_number == page_num)
            return NV_OK;
    }

    info = kmalloc(sizeof(struct dirty_page_info), GFP_ATOMIC);
    if (!info)
        return NV_ERR_NO_MEMORY;

    info->page_number = page_num;
    info->timestamp   = timestamp;

    node = kmalloc(sizeof(struct dirty_ll_node), GFP_ATOMIC);
    if (!node) {
        kfree(info);
        return NV_ERR_NO_MEMORY;
    }
    node->info = info;
    node->next = NULL;

    ll->tail->next = node;
    ll->tail = node;
    return NV_OK;
}

static struct dirty_page_info *ll_backend_lookup(struct uvm_dirty_ds *ds,
                                                   unsigned long page_num)
{
    struct dirty_linked_list *ll = ds->priv;
    struct dirty_ll_node     *cur;

    for (cur = ll->head->next; cur != NULL; cur = cur->next) {
        if (cur->info->page_number == page_num)
            return cur->info;
    }
    return NULL;
}

static void ll_backend_for_each_in_range(struct uvm_dirty_ds *ds,
                                          unsigned long start_page,
                                          unsigned long end_page,
                                          uvm_dirty_ds_range_fn fn,
                                          void *ctx)
{
    struct dirty_linked_list *ll = ds->priv;
    struct dirty_ll_node     *cur;

    for (cur = ll->head->next; cur != NULL; cur = cur->next) {
        unsigned long pn = cur->info->page_number;
        if (pn >= start_page && pn <= end_page)
            fn(pn, cur->info->timestamp, ctx);
    }
}

static void ll_backend_destroy(struct uvm_dirty_ds *ds)
{
    struct dirty_linked_list *ll = ds->priv;
    struct dirty_ll_node     *cur;
    struct dirty_ll_node     *next;

    cur = ll->head->next;
    while (cur != NULL) {
        next = cur->next;
        kfree(cur->info);
        kfree(cur);
        cur = next;
    }

    kfree(ll->head);   /* free the sentinel */
    kfree(ll);
    ds->priv = NULL;
}

const struct uvm_dirty_ds_ops uvm_dirty_ds_linked_list_ops = {
    .init              = ll_backend_init,
    .insert            = ll_backend_insert,
    .lookup            = ll_backend_lookup,
    .for_each_in_range = ll_backend_for_each_in_range,
    .destroy           = ll_backend_destroy,
};
// END OF EDIT
