/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/
#include <stdlib.h>
#include <errno.h>

#include "aml.h"
#include "aml/area/area.h"
#include "aml/area/linux.h"

#define AML_AREA_LINUX_MBIND_FLAGS MPOL_MF_MOVE

struct aml_area* aml_area_linux_create(const int mmap_flags,
				       const struct aml_bitmap *nodemask,
				       const int binding_flags)
{
	struct aml_area_linux_data *d;
	struct aml_area *area;

	d = malloc(sizeof(*d));
	if(d == NULL){
		aml_errno = AML_AREA_ENOMEM;
		return NULL;
	}

	d->nodeset = numa_allocate_nodemask();
	if(d->nodeset == NULL){
	        aml_errno = AML_AREA_ENOMEM;
		goto err_with_data;
	}

	aml_bitmap_copy_to_ulong(nodemask, d->nodeset->maskp, d->nodeset->size);
	d->binding_flags = binding_flags;
	d->mmap_flags = mmap_flags | MAP_ANONYMOUS;

	area = malloc(sizeof(*area));
	if(area == NULL){
	        aml_errno = AML_AREA_ENOMEM;
		goto err_with_data;
	}

	area->data = (struct aml_area_data*)d;
	area->ops = &aml_area_linux_ops;
	
	return area;
	
 err_with_data:
	free(d);
	return NULL;
}

void
aml_area_linux_destroy(struct aml_area* area){
	if(area == NULL || area->data == NULL)
		return;
	
	struct aml_area_linux_data *d = (struct aml_area_linux_data *)area->data;
	
	if(d->nodeset != NULL)
		numa_free_nodemask(d->nodeset);
        free(d);
}

int
aml_area_linux_mbind(struct aml_area_linux_data    *bind,
		     void                          *ptr,
		     size_t                         size)
{	
	struct bitmask *nodeset;

	if(bind->nodeset != NULL)
		nodeset = bind->nodeset;
	else
		nodeset = numa_get_mems_allowed();		
	
	long err = mbind(ptr,
			 size,
			 bind->binding_flags,
			 nodeset->maskp,
			 nodeset->size,
			 AML_AREA_LINUX_MBIND_FLAGS);

	if(err == 0)
		return AML_SUCCESS;

	switch(errno){
	case EFAULT:
		return AML_AREA_EDOM;
	case EINVAL:
		return AML_AREA_EINVAL;
	case EIO:
		return AML_AREA_EINVAL;
	case ENOMEM:
		return AML_AREA_ENOMEM;
	case EPERM:
		return AML_AREA_ENOTSUP;
	}
}

int
aml_area_linux_check_binding(struct aml_area_linux_data *area_data,
			     void                       *ptr,
			     size_t                      size)
{
	int err, mode, i;
	struct bitmask *nodeset;

	nodeset = numa_allocate_nodemask();
	if(nodeset == NULL)
		return AML_AREA_ENOMEM;
	
	err = get_mempolicy(&mode,
			    nodeset->maskp,
			    nodeset->size,
			    ptr,
			    AML_AREA_LINUX_MBIND_FLAGS);
	
	if(err < 0){
		err = AML_AREA_EINVAL;
		goto out;
	}
	
	err = 1;	
	if(mode != area_data->binding_flags)
	        err = 0;
	for(i=0; i<numa_max_possible_node(); i++){
		if(numa_bitmask_isbitset(nodeset, i) !=
		   numa_bitmask_isbitset(area_data->nodeset, i)){
			err = 0;
			break;
		}
	}	
 out:
	numa_free_nodemask(nodeset);
	return err;
}

void* aml_area_linux_mmap(const struct aml_area_data  *area_data,
			  void                        *ptr,
			  size_t                       size)
{
	struct aml_area_linux_data *data = (struct aml_area_linux_data *) area_data;
	
	void * out = mmap(ptr,
			  size,
			  PROT_READ|PROT_WRITE,
			  data->mmap_flags,
			  0, 0);

	if(out == MAP_FAILED){
		out = NULL;
		switch(errno){
		case EAGAIN:
			aml_errno = AML_AREA_ENOMEM;
			break;
		case EINVAL:
			aml_errno = AML_AREA_EINVAL;
			break;
		case ENOMEM:
		        aml_errno = AML_AREA_ENOMEM;
			break;
		default:
			aml_errno = AML_AREA_EINVAL;
			break;
		}
	}
	
	if(out != NULL && (data->nodeset != NULL ||
			   data->binding_flags != MPOL_DEFAULT)){
		int err = aml_area_linux_mbind(data, out, size);
		if(err != AML_SUCCESS){
			aml_errno = err;
			goto binding_failed;
		}
	}
	
	return out;
	
 binding_failed:	
	munmap(out, size);	
	return NULL;
}

int
aml_area_linux_munmap(__attribute__ ((unused)) const struct aml_area_data* area_data,
		      void *ptr,
		      const size_t size)
{
	int err = munmap(ptr, size);
	if(err == -1)
		return AML_AREA_EINVAL;
	return AML_SUCCESS;
}

/*****************************************************************************
 * Areas declaration
 *****************************************************************************/

const struct aml_area_linux_data aml_area_linux_data_default = {
	.nodeset = NULL,
	.binding_flags = MPOL_DEFAULT,
	.mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS
};

struct aml_area_ops aml_area_linux_ops = {
	.mmap = aml_area_linux_mmap,
	.munmap = aml_area_linux_munmap
};
	
const struct aml_area aml_area_linux = {
	.ops = &aml_area_linux_ops,
	.data = (struct aml_area_data *)(&aml_area_linux_data_default)
};

