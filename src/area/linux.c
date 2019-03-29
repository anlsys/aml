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

static int aml_area_linux_check_mmap_flags(const int mmap_flags){
	switch(mmap_flags){
	case AML_AREA_LINUX_MMAP_FLAG_PRIVATE:
		break;
	case AML_AREA_LINUX_MMAP_FLAG_SHARED:
		break;
	default:
		return 0;
	}
	
	return 1;
}

static int aml_area_linux_check_binding_flags(const int binding_flags){
	switch(binding_flags){
	case AML_AREA_LINUX_BINDING_FLAG_BIND:
		break;
	case AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE:
		break;
	case AML_AREA_LINUX_BINDING_FLAG_PREFERRED:
		break;
	default:
		return 0;
	}
	
	return 1;
}

struct aml_area* aml_area_linux_create(const int mmap_flags,
				       const struct aml_bitmap *nodemask,
				       const int binding_flags)
{
	struct aml_area_linux_data *d;
	struct aml_area *area;	

	/* Check flags */
	if(!aml_area_linux_check_mmap_flags(mmap_flags) ||
	   !aml_area_linux_check_binding_flags(binding_flags)){
		aml_errno = AML_AREA_EINVAL;
		return NULL;
	}

	d = malloc(sizeof(*d));
	if(d == NULL){
		aml_errno = AML_AREA_ENOMEM;
		return NULL;
	}

	/* Check/set nodemask */
	d->nodeset = numa_get_mems_allowed();
	if(d->nodeset == NULL){
		aml_errno = AML_AREA_ENOMEM;
		goto err_with_data;
	}
	
	if(nodemask != NULL){					
		int aml_last = aml_bitmap_last(nodemask);
		int allowed_last = numa_bitmask_weight(d->nodeset);
		while(!numa_bitmask_isbitset(d->nodeset, --allowed_last));
		if(aml_last > allowed_last){
		        aml_errno = AML_AREA_EDOM;			
			goto err_with_nodeset;
		}
		aml_bitmap_copy_to_ulong(nodemask,
					 d->nodeset->maskp,
					 d->nodeset->size);
	}

	/* set area_data and area */
	d->binding_flags = binding_flags;
	d->mmap_flags = mmap_flags;

	area = malloc(sizeof(*area));
	if(area == NULL){
	        aml_errno = AML_AREA_ENOMEM;
		goto err_with_nodeset;
	}

	area->data = (struct aml_area_data*)d;
	area->ops = &aml_area_linux_ops;
	
	return area;

 err_with_nodeset:
	numa_free_nodemask(d->nodeset);
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
	return AML_FAILURE;
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
		int ptr_set = numa_bitmask_isbitset(nodeset, i);
		int bitmask_set = numa_bitmask_isbitset(area_data->nodeset, i);
		if(mode == AML_AREA_LINUX_BINDING_FLAG_BIND && ptr_set != bitmask_set)
			goto binding_failed;
		if(mode == AML_AREA_LINUX_BINDING_FLAG_INTERLEAVE && ptr_set && !bitmask_set)
			goto binding_failed;
	}
	
	goto out;
 binding_failed:
	err = 0;
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
		aml_errno = AML_FAILURE;
	}
		
	return out;
}

int
aml_area_linux_munmap(__attribute__ ((unused)) const struct aml_area_data* area_data,
		      void *ptr,
		      const size_t size)
{
	int err = munmap(ptr, size);
	if(err == -1)
		return AML_FAILURE;
	return AML_SUCCESS;
}


static void*
aml_area_linux_mmap_mbind(const struct aml_area_data  *area_data,
			  void                        *ptr,
			  size_t                       size)
{
	void * out = aml_area_linux_mmap(area_data, ptr, size);

	if(out == NULL)
		return NULL;

	struct aml_area_linux_data *data = (struct aml_area_linux_data *) area_data;

	if(data->nodeset != NULL || data->binding_flags != MPOL_DEFAULT){
		int err = aml_area_linux_mbind(data, out, size);
		if(err != AML_SUCCESS){
			aml_errno = err;
			munmap(out, size);	
			return NULL;
		}
	}

	return out;
}

/*****************************************************************************
 * Areas declaration
 *****************************************************************************/

const struct aml_area_linux_data aml_area_linux_data_default = {
	.nodeset = NULL,
	.binding_flags = MPOL_DEFAULT,
	.mmap_flags = AML_AREA_LINUX_MMAP_FLAG_PRIVATE
};

struct aml_area_ops aml_area_linux_ops = {
	.mmap = aml_area_linux_mmap_mbind,
	.munmap = aml_area_linux_munmap
};
	
const struct aml_area aml_area_linux = {
	.ops = &aml_area_linux_ops,
	.data = (struct aml_area_data *)(&aml_area_linux_data_default)
};

