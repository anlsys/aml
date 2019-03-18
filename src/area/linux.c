#define _GNU_SOURCE
#include <stdlib.h>
#include <sys/mman.h>
#include <errno.h>
#include "config.h"
#include "aml.h"
#include "aml/area/area.h"
#ifdef HAVE_HWLOC
#include "aml/utils/hwloc.h"

#define aml_binding_flags (HWLOC_MEMBIND_PROCESS|HWLOC_MEMBIND_MIGRATE|HWLOC_MEMBIND_NOCPUBIND|HWLOC_MEMBIND_BYNODESET)
#define __hwloc_unused__

extern hwloc_topology_t aml_topology;
#else
#define __hwloc_unused__ __attribute__ ((unused))
#endif

struct host_binding{
	struct aml_bitmap bitmap;
	int flags;
};

static int
host_create(struct aml_area* area)
{
	struct host_binding *binding = malloc(sizeof(struct host_binding));
	if(binding == NULL)
		return AML_AREA_ENOMEM;
	aml_bitmap_zero(&(binding->bitmap));
	binding->flags = 0;
	area->data = binding;
	return AML_AREA_SUCCESS;
}

static void
host_destroy(struct aml_area* area){
        free(area->data);
}

static int
host_bind(__hwloc_unused__ struct aml_area *area,
	  __hwloc_unused__ const struct aml_bitmap *binding,
	  __hwloc_unused__ const unsigned long flags)	  
{
#ifdef HAVE_HWLOC
	int nnodes = hwloc_get_nbobjs_by_depth(aml_topology, HWLOC_TYPE_DEPTH_NUMANODE);
        if(aml_bitmap_last(binding) >= nnodes)
		return AML_AREA_EDOM;
	struct host_binding *bind = area->data;
	aml_bitmap_copy(&(bind->bitmap), binding);
	bind->flags = flags;
	return AML_AREA_SUCCESS;
#else
	return AML_AREA_ENOTSUP;
#endif	
}

static int
host_mbind(__hwloc_unused__ struct host_binding *bind,
	   __hwloc_unused__ void                *ptr,
	   __hwloc_unused__ size_t               size)
{
	int err = AML_AREA_ENOTSUP;	
#ifdef HAVE_HWLOC
	hwloc_bitmap_t bitmap = hwloc_bitmap_from_aml_bitmap(&bind->bitmap);

	if(bitmap == NULL)
		return AML_AREA_ENOMEM;
	
	err = hwloc_set_area_membind(aml_topology,
				     ptr, size,
				     bitmap,
				     bind->flags,
				     aml_binding_flags);

	hwloc_bitmap_free(bitmap);
	
	if(err >= 0)
		err = AML_AREA_SUCCESS;
	else {
		switch(errno){
		case ENOSYS:
			err = AML_AREA_ENOTSUP;
			break;
		case EXDEV:
			err = AML_AREA_ENOTSUP;
			break;
		default:
			err = AML_AREA_ENOTSUP;
			break;
		}
	}
#endif
	return err;
}

static int
host_check_binding(__hwloc_unused__ struct aml_area *area,
		   __hwloc_unused__ void            *ptr,
		   __hwloc_unused__ size_t           size)
{
	int err = AML_AREA_ENOTSUP;
#ifdef HAVE_HWLOC	
	struct aml_bitmap nodemask;	
	hwloc_membind_policy_t policy;
	hwloc_bitmap_t hwloc_bitmap;
	struct host_binding *bind = area->data;

	if(bind == NULL)
		return AML_AREA_EINVAL;

	hwloc_bitmap = hwloc_bitmap_alloc();
	if(hwloc_bitmap == NULL)
		return AML_AREA_ENOMEM;
	
	err = hwloc_get_area_membind(aml_topology,
				     ptr, size,
				     hwloc_bitmap,
				     &policy, aml_binding_flags);
	if(err < 0){
		err = AML_AREA_EINVAL;
		goto out;
	}

	err = aml_bitmap_copy_hwloc_bitmap(&nodemask, hwloc_bitmap);
	if(err > (int)AML_BITMAP_MAX){
		err = AML_AREA_EDOM;
		goto out;
	}
	
	if(policy != bind->flags)
	        err = 0;
	else if(!aml_bitmap_isequal(&nodemask, &bind->bitmap))
		err = 0;
	else
		err = 1;
 out:
	hwloc_bitmap_free(hwloc_bitmap);
#endif	
	return err;
}

static int
host_mmap_generic(__attribute__ ((unused)) const struct aml_area* area,
		  void **ptr,
		  size_t size,
		  int    flags)
{
	*ptr = mmap(*ptr,
		    size,
		    PROT_READ|PROT_WRITE,
		    flags|MAP_ANONYMOUS,
		    0, 0);

	if(*ptr == MAP_FAILED){
		*ptr = NULL;
		switch(errno){
		case EAGAIN:
			return AML_AREA_ENOMEM;
		case EINVAL:
			return AML_AREA_EINVAL;
		case ENOMEM:
			return AML_AREA_ENOMEM;
		default:
			return AML_AREA_EINVAL;
		}
	}

	if(!aml_bitmap_iszero(&(((struct host_binding *)area->data)->bitmap)))
	        host_mbind(area->data, *ptr, size);
	return AML_AREA_SUCCESS;
}

static int
host_mmap_private(const struct aml_area* area,
		  void **ptr,
		  size_t size)
{
	return host_mmap_generic(area, ptr, size, MAP_PRIVATE);
}

static int
host_mmap_shared(const struct aml_area* area,
		 void **ptr,
		 size_t size)
{
	return host_mmap_generic(area, ptr, size, MAP_SHARED);
}

	
static int
host_munmap(__attribute__ ((unused)) const struct aml_area* area,
	    void *ptr,
	    const size_t size)
{
	int err = munmap(ptr, size);
	if(err == -1)
		return AML_AREA_EINVAL;
	return AML_AREA_SUCCESS;
}

static int
host_malloc(struct aml_area *area,
		void           **ptr,
		size_t           size,
		size_t           alignement)
{
	void *data;
	
	if(alignement == 0)
		data  = malloc(size);
	else{
		int err = posix_memalign(&data, alignement, size);
		switch(err){
		case EINVAL:
			return AML_AREA_EINVAL;
		case ENOMEM:
			return AML_AREA_ENOMEM;
		default:
			break;
		}
	}
	if(data == NULL)
		return AML_AREA_ENOMEM;
	*ptr = data;
	
	if(!aml_bitmap_iszero(&(((struct host_binding *)area->data)->bitmap)))
	        host_mbind(area->data, *ptr, size);
	return AML_AREA_SUCCESS;
}

static int
host_free(__attribute__ ((unused)) struct aml_area *area,
	      void *ptr)	
{
	free(ptr);
	return AML_AREA_SUCCESS;
}


/*********************************************************************************
 * Areas declaration
 *********************************************************************************/

static struct host_binding host_binding_default = {{0,0,0,0,0,0,0,0}, 0};

static struct aml_area_ops aml_area_host_private_ops = {
	.create = host_create,
	.destroy = host_destroy,
	.map = host_mmap_private,
	.unmap = host_munmap,
	.malloc = host_malloc,
	.free = host_free,
	.bind = host_bind,
	.check_binding = host_check_binding
};

static struct aml_area aml_area_host_private_s = {
	.ops = &aml_area_host_private_ops,
	.data = &host_binding_default
};

struct aml_area *aml_area_host_private = &aml_area_host_private_s;

static struct aml_area_ops aml_area_host_shared_ops = {
	.create = host_create,
	.destroy = host_destroy,
	.map = host_mmap_shared,
	.unmap = host_munmap,
	.malloc = NULL,
	.free = NULL,	
	.bind = host_bind,
	.check_binding = host_check_binding
};

static struct aml_area aml_area_host_shared_s = {
	.ops = &aml_area_host_shared_ops,
	.data = &host_binding_default
};

struct aml_area *aml_area_host_shared = &aml_area_host_shared_s;

