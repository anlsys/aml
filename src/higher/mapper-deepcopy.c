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

#include "aml/higher/mapper.h"
#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/deepcopy.h"
#include "aml/utils/inner-malloc.h"
#define utarray_oom()                                                          \
	do {                                                                   \
		err = -AML_ENOMEM;                                             \
		goto ut_error;                                                 \
	} while (0)
#include "internal/utarray.h"

struct mapped_ptr {
	void *ptr;
	size_t size;
	struct aml_area *area;
};

static void mapped_ptr_destroy(void *ptr)
{
	struct mapped_ptr *p = (struct mapped_ptr *)ptr;
	(void)aml_area_munmap(p->area, p->ptr, p->size);
}

UT_icd mapped_ptr_icd = {
        .sz = sizeof(struct mapped_ptr),
        .init = NULL,
        .copy = NULL,
        .dtor = mapped_ptr_destroy,
};

static void aml_mapper_creator_destroy(void *elt)
{
	struct aml_mapper_creator *c = (struct aml_mapper_creator *)elt;
	(void)aml_mapper_creator_abort(c);
}

UT_icd creator_icd = {
        .sz = sizeof(struct aml_mapper_creator *),
        .init = NULL,
        .copy = NULL,
        .dtor = aml_mapper_creator_destroy,
};

int aml_mapper_deepcopy(aml_deepcopy_data *out,
                        void *src_ptr,
                        struct aml_mapper *mapper,
                        struct aml_area *area,
                        struct aml_area_mmap_options *area_opts,
                        struct aml_dma *dma_src_host,
                        struct aml_dma *dma_host_dst,
                        aml_dma_operator memcpy_src_host,
                        aml_dma_operator memcpy_host_dst)
{
	int err;
	UT_array *ptrs = NULL;
	UT_array crtrs;
	struct aml_mapper_creator *crtr = NULL, *next = NULL;
	struct mapped_ptr ptr = {.ptr = NULL, .size = 0, .area = area};

	// Allocate array of creators spawned in branches.
	utarray_init(&crtrs, &creator_icd);

	// Allocate and initialize pointer array.
	utarray_new(ptrs, &mapped_ptr_icd);

	// Allocate and initialize first constructor.
	err = aml_mapper_creator_create(&crtr, src_ptr, 0, mapper, area,
	                                area_opts, dma_src_host, dma_host_dst,
	                                memcpy_src_host, memcpy_host_dst);
	if (err != AML_SUCCESS)
		goto ut_error;

iterate_creator:
	err = aml_mapper_creator_next(crtr);
	if (err == AML_SUCCESS)
		goto iterate_creator;
	if (err == -AML_EINVAL)
		goto branch;
	if (err == -AML_EDOM)
		goto next_creator;
	goto ut_error;
branch:
	// Create branch.
	err = aml_mapper_creator_branch(&next, crtr, area, area_opts,
	                                dma_host_dst, memcpy_host_dst);
	if (err != AML_SUCCESS)
		goto ut_error;
	// Push new creator to be in the stack of pending creators.
	utarray_push_back(&crtrs, &next);
	next = NULL; // Avoids double destruction on error.
	goto iterate_creator;
next_creator:
	err = aml_mapper_creator_finish(crtr, &ptr.ptr, &ptr.size);
	if (err != AML_SUCCESS)
		goto ut_error;
	crtr = utarray_back(&crtrs);
	utarray_pop_back(&crtrs);
	utarray_push_back(ptrs, &ptr);
	if (crtr == NULL)
		goto success;
	goto iterate_creator;
success:
	utarray_done(&crtrs);
	*out = (aml_deepcopy_data)ptrs;
	return AML_SUCCESS;
ut_error:
	if (crtr != NULL)
		aml_mapper_creator_abort(crtr);
	if (next != NULL)
		aml_mapper_creator_abort(next);
	if (ptrs != NULL)
		utarray_free(ptrs);
	utarray_done(&crtrs);
	return err;
}

void *aml_deepcopy_ptr(aml_deepcopy_data data)
{
	UT_array *ptrs = (UT_array *)data;
	struct mapped_ptr *ptr = utarray_eltptr(ptrs, 0);
	return ptr->ptr;
}

int aml_mapper_deepfree(aml_deepcopy_data data)
{
	UT_array *ptrs = (UT_array *)data;
	struct mapped_ptr *ptr;
	int err;

	if (ptrs == NULL)
		return -AML_EINVAL;

	while (1) {
		ptr = utarray_back(ptrs);
		if (ptr == NULL) {
			utarray_free(ptrs);
			return AML_SUCCESS;
		}
		err = aml_area_munmap(ptr->area, ptr->ptr, ptr->size);
		if (err != AML_SUCCESS)
			return err;
		utarray_pop_back(ptrs);
	}
}
