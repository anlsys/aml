/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/higher/allocator.h"
#include "aml/higher/allocator/freelist.h"

# define DYNAMICALLY_REALLOC 1

int aml_allocator_freelist_give(struct aml_allocator_data *data, void * ptr, size_t size)
{
    struct aml_allocator_freelist_chunk * chunk = (struct aml_allocator_freelist_chunk *) malloc(sizeof(struct aml_allocator_freelist_chunk));
    if (chunk == NULL)
        return -AML_ENOMEM;

    chunk->super.ptr = ptr;
    chunk->super.size = size;
    chunk->super.user_data = 0;
    chunk->state = FREELIST_CHUNK_STATE_FREE;
    chunk->prev = NULL;
    chunk->next = NULL;

    struct aml_allocator_freelist * freelist = (struct aml_allocator_freelist *) data;
    chunk->freelink = freelist->free_chunk_list;
    freelist->free_chunk_list = chunk;

    return AML_SUCCESS;
}

static inline struct aml_allocator_freelist_chunk * aml_allocator_freelist_alloc_chunk_bestfit(struct aml_allocator_data *data, size_t size)
{
    assert((size % 8) == 0);

    /* best fit strategy */
    struct aml_allocator_freelist * freelist = (struct aml_allocator_freelist *) data;
    struct aml_allocator_freelist_chunk * curr = freelist->free_chunk_list;

    struct aml_allocator_freelist_chunk * prevfree = NULL;
    size_t min_size = 0;
    struct aml_allocator_freelist_chunk * min_size_curr = NULL;
    struct aml_allocator_freelist_chunk * min_size_prevfree = NULL;

    while (curr)
    {
        size_t curr_size = curr->super.size;
        if (curr_size >= size)
        {
            if ((min_size_curr == 0) || (min_size > curr_size))
            {
                min_size = curr_size;
                min_size_curr = curr;
                min_size_prevfree = prevfree;
            }
        }
        prevfree = curr;
        curr = curr->freelink;
    }

    /* and the winner is min_size_curr ! */
    curr = min_size_curr;
    prevfree = min_size_prevfree;

    /* split chunk */
    if ((curr != NULL) && (min_size - size >= (size_t)(0.5*(double)size)))
    {
        size_t curr_size = curr->super.size;
        struct aml_allocator_freelist_chunk * remainder = (struct aml_allocator_freelist_chunk *) malloc(sizeof(struct aml_allocator_freelist_chunk));
        remainder->super.ptr        = (void *) ((uintptr_t)curr->super.ptr + size);
        remainder->super.size       = (curr_size - size);
        remainder->super.user_data  = 0;
        remainder->state            = FREELIST_CHUNK_STATE_FREE;
        remainder->prev             = curr;
        remainder->next             = curr->next;
        remainder->freelink         = curr->freelink;

        /* link remainder segment after curr */
        if (curr->next)
            curr->next->prev = remainder;
        curr->next = remainder;
        curr->super.size = size;
        curr->freelink = remainder;
    }

    if (curr != NULL)
    {
        if (prevfree)
            prevfree->freelink = curr->freelink;
        else
            freelist->free_chunk_list = curr->freelink;
        curr->state = FREELIST_CHUNK_STATE_ALLOCATED;
        curr->freelink = NULL;
    }

    return curr;
}

struct aml_allocator_chunk *aml_allocator_freelist_alloc_chunk(struct aml_allocator_data *data, size_t user_size)
{
    struct aml_allocator_freelist * freelist = (struct aml_allocator_freelist *) data;

    /* align data */
    const size_t size = (user_size + 7UL) & ~7UL;

    struct aml_allocator_freelist_chunk *chunk;
    pthread_mutex_lock(&freelist->lock);
    {
        chunk = aml_allocator_freelist_alloc_chunk_bestfit(data, size);
        # if DYNAMICALLY_REALLOC
        if (chunk == NULL)
        {
            # define MIN_ALLOC_SIZE (512*1024*1024)
            const size_t mapsize = (size < MIN_ALLOC_SIZE) ? MIN_ALLOC_SIZE : size;
            # undef MIN_ALLOC_SIZE
            void * ptr = aml_area_mmap(freelist->area, mapsize, freelist->opts);
            if (ptr)
            {
                aml_allocator_freelist_give(data, ptr, size);
                chunk = aml_allocator_freelist_alloc_chunk_bestfit(data, size);
            }
        }
        # endif
    }
    pthread_mutex_unlock(&freelist->lock);

    return (struct aml_allocator_chunk *) chunk;
}

int aml_allocator_freelist_free_chunk(struct aml_allocator_data *data, struct aml_allocator_chunk *user_chunk)
{
    struct aml_allocator_freelist * freelist = (struct aml_allocator_freelist *) data;
    struct aml_allocator_freelist_chunk * chunk = (struct aml_allocator_freelist_chunk *) user_chunk;

    int delete_chunk = 0;
    pthread_mutex_lock(&freelist->lock);
    {
        chunk->state = FREELIST_CHUNK_STATE_FREE;

        /* can we merge chunk into next_chunk ? */
        struct aml_allocator_freelist_chunk * next_chunk = chunk->next;
        if (next_chunk && next_chunk->state == FREELIST_CHUNK_STATE_FREE)
        {
            next_chunk->prev = chunk->prev;
            if (chunk->prev)
                chunk->prev->next = next_chunk;
            next_chunk->super.size += chunk->super.size;
            assert(next_chunk->super.ptr > chunk->super.ptr);
            next_chunk->super.ptr = chunk->super.ptr;
            delete_chunk = 1;
        }

        struct aml_allocator_freelist_chunk * prev_chunk = chunk->prev;
        if (prev_chunk)
        {
            /*  if prev_chunk is a free chunk and 'delete_chunk' is 1,
             *  then we have to merge prev and next */
            if (prev_chunk->state == FREELIST_CHUNK_STATE_FREE)
            {
                if (delete_chunk)
                {
                    assert(prev_chunk->super.ptr < chunk->super.ptr);
                    assert(prev_chunk->super.ptr < next_chunk->super.ptr);

                    prev_chunk->super.size += next_chunk->super.size;
                    prev_chunk->next = next_chunk->next;
                    if (next_chunk->next)
                        next_chunk->next->prev = prev_chunk;
                    prev_chunk->freelink = next_chunk->freelink;
                    free(next_chunk);
                }
                else
                {
                    /* merge chunk into prev_chunk */
                    assert(prev_chunk->super.ptr < chunk->super.ptr);
                    prev_chunk->next = chunk->next;
                    if (chunk->next)
                        chunk->next->prev = prev_chunk;
                    prev_chunk->super.size += chunk->super.size;
                    delete_chunk = 1;
                }
            }
            else if (!delete_chunk)
            {
                /* free_chunk_list is ordered by increasing adress: search form prev the previous bloc */
                while (prev_chunk && prev_chunk->state != FREELIST_CHUNK_STATE_FREE)
                    prev_chunk = prev_chunk->prev;

                if (!prev_chunk)
                {
                    chunk->freelink = freelist->free_chunk_list;
                    freelist->free_chunk_list = chunk;
                }
                else
                {
                    chunk->freelink = prev_chunk->freelink;
                    prev_chunk->freelink = chunk;
                }
            }
        }
        else if (!delete_chunk)
        {
            chunk->freelink = freelist->free_chunk_list;
            freelist->free_chunk_list = chunk;
        }
    }
    pthread_mutex_unlock(&freelist->lock);

    if (delete_chunk)
        free(chunk);

    return AML_SUCCESS;
}

int aml_allocator_freelist_destroy(struct aml_allocator **allocator)
{
    if (allocator == NULL || *allocator == NULL || (*allocator)->data == NULL)
        return -AML_EINVAL;

    // struct aml_allocator_freelist * alloc = (struct aml_allocator_freelist *) (*allocator)->data;
    // TODO - free allocated cuda memory
    free(*allocator);
    return AML_SUCCESS;
}

int aml_allocator_freelist_create(struct aml_allocator **allocator,
                               struct aml_area *area,
                               struct aml_area_mmap_options *opts)
{
    static struct aml_allocator_ops aml_allocator_freelist_ops = {
        .alloc = NULL,
        .free = NULL,
        .give = aml_allocator_freelist_give,
        .alloc_chunk = aml_allocator_freelist_alloc_chunk,
        .free_chunk = aml_allocator_freelist_free_chunk
    };

    if (allocator == NULL || area == NULL)
        return -AML_EINVAL;

    struct aml_allocator * alloc = (struct aml_allocator *) malloc(sizeof(struct aml_allocator) + sizeof(struct aml_allocator_freelist));
    if (alloc == NULL)
        return -AML_FAILURE;

    struct aml_allocator_freelist * freelist = (struct aml_allocator_freelist *) (alloc + 1);
    if (pthread_mutex_init(&freelist->lock, NULL))
    {
        free(alloc);
        return -AML_FAILURE;
    }

    freelist->area = area;
    freelist->opts = opts;
    freelist->free_chunk_list = NULL;

    alloc->data = (struct aml_allocator_data *) freelist;
    alloc->ops  = &aml_allocator_freelist_ops;

    (*allocator) = alloc;

    return AML_SUCCESS;
}
