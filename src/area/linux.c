#include <stdlib.h>
#include <sys/mman.h>
#include <errno.h>
#include "config.h"
#include "aml.h"
#include "aml/area/area.h"

#ifdef HAVE_LINUX_NUMA

#include <numa.h>
#include <numaif.h>

#define linux_mbind_flags MPOL_MF_MOVE

struct linux_binding{
	struct bitmask *nodeset;
	int flags;
};

struct linux_binding linux_binding_default = {NULL, MPOL_DEFAULT};

const unsigned long aml_area_linux_flag_bind = MPOL_BIND;
const unsigned long aml_area_linux_flag_interleave = MPOL_INTERLEAVE;
const unsigned long aml_area_linux_flag_preferred = MPOL_PREFERRED;

static int
linux_create(struct aml_area* area)
{
	struct linux_binding *binding = malloc(sizeof(struct linux_binding));
	if(binding == NULL)
		return AML_AREA_ENOMEM;
	binding->nodeset = NULL;
	binding->flags = MPOL_DEFAULT;
	area->data = binding;
	return AML_AREA_SUCCESS;
}

static void
linux_destroy(struct aml_area* area){
	struct linux_binding *binding = area->data;
	if(binding->nodeset != NULL)
		numa_free_nodemask(binding->nodeset);
        free(area->data);
}

static int
linux_bind(struct aml_area         *area,
	   const struct aml_bitmap *binding,
	   const unsigned long      flags)
{
	struct linux_binding *bind = area->data;
	struct bitmask *allowed = numa_get_mems_allowed();
	int last  = aml_bitmap_last(binding);
	int first = aml_bitmap_first(binding);
	int i, allowed_bit, set_bit;

	if(flags != aml_area_linux_flag_bind ||
	   flags != aml_area_linux_flag_interleave ||
	   flags != aml_area_linux_flag_preferred)
		return AML_AREA_EINVAL;
	
	if(last > numa_max_node())
		return AML_AREA_EDOM;

	if(allowed == NULL)
		return AML_AREA_ENOMEM;
		
	if(binding == NULL)
		goto set_flags;

	for(i=first; i<= last; i++){
		allowed_bit = numa_bitmask_isbitset(allowed, i);
		set_bit = aml_bitmap_isset(binding, i);
		if(!allowed_bit && set_bit)
			return AML_AREA_EDOM;
		if(allowed_bit && !set_bit)
			numa_bitmask_clearbit(allowed, i);
	}
	
	if(bind->nodeset != NULL)
		numa_free_nodemask(bind->nodeset);
	bind->nodeset = allowed;

 set_flags:
	bind->flags = flags;
	return AML_AREA_SUCCESS;
}

static int
linux_mbind(struct linux_binding *bind,
	    void                 *ptr,
	    size_t                size)
{
	
	struct bitmask *nodeset;

	if(bind == NULL)
		return AML_AREA_EINVAL;

	if(bind->nodeset == NULL || (bind->flags & MPOL_DEFAULT))
		return AML_AREA_SUCCESS;

	if(bind->nodeset != NULL)
		nodeset = bind->nodeset;
	else
		nodeset = numa_get_mems_allowed();			
	
	long err = mbind(ptr,
			 size,
			 bind->flags,
			 nodeset,
			 numa_max_node(),
			 linux_mbind_flags);

	if(err == 0)
		return AML_AREA_SUCCESS;

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

static int
linux_check_binding(struct aml_area *area,
		    void            *ptr,
		    size_t           size)
{
	int err, mode, i;
	struct bitmask *nodeset;
	struct linux_binding *bind = area->data;

	nodeset = numa_allocate_nodemask();
	if(nodeset == NULL)
		return AML_AREA_ENOMEM;
	
	err = get_mempolicy(&mode,
			    nodeset,
			    numa_max_node(),
			    ptr,
			    linux_mbind_flags);
	
	if(err < 0){
		err = AML_AREA_EINVAL;
		goto out;
	}

	
	err = 1;	
	if(mode != bind->flags)
	        err = 0;
	for(i=0; i<numa_max_possible_node(); i++){
		if(numa_bitmask_isbitset(nodeset, i) !=
		   numa_bitmask_isbitset(bind->nodeset, i)){
			err = 0;
			break;
		}
	}	
 out:
	numa_free_nodemask(nodeset);
	return err;
}
#else
const unsigned long aml_area_linux_flag_bind = 0;
const unsigned long aml_area_linux_flag_interleave = 0;
const unsigned long aml_area_linux_flag_preferred = 0;
#endif //HAVE_LINUX_NUMA

static int
linux_mmap_generic(void **ptr,
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
	return AML_AREA_SUCCESS;
}

static int
linux_mmap_mbind(const struct aml_area* area,
		 void **ptr,
		 size_t size,
		 int    flags)
{
	int err = linux_mmap_generic(ptr, size, flags);
	if(err != AML_AREA_SUCCESS)
		return err;

#ifdef HAVE_LINUX_NUMA
	return linux_mbind(area->data, *ptr, size);
#else
	return AML_AREA_SUCCESS;
#endif		
}

int
linux_mmap_private(__attribute__ ((unused)) const struct aml_area* area,
		   void **ptr,
		   size_t size)
{
	return linux_mmap_generic(ptr, size, MAP_PRIVATE);
}

int
linux_mmap_shared(__attribute__ ((unused)) const struct aml_area* area,
		  void **ptr,
		  size_t size)
{
	return linux_mmap_generic(ptr, size, MAP_SHARED);
}

static int
linux_mmap_private_mbind(const struct aml_area* area,
			 void **ptr,
			 size_t size)
{
	return linux_mmap_mbind(area, ptr, size, MAP_PRIVATE);
}

static int
linux_mmap_shared_mbind(const struct aml_area* area,
			void **ptr,
			size_t size)
{
	return linux_mmap_mbind(area, ptr, size, MAP_SHARED);
}

	
int
linux_munmap(__attribute__ ((unused)) const struct aml_area* area,
	     void *ptr,
	     const size_t size)
{
	int err = munmap(ptr, size);
	if(err == -1)
		return AML_AREA_EINVAL;
	return AML_AREA_SUCCESS;
}

int
linux_malloc(struct aml_area *area,
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
	
	return AML_AREA_SUCCESS;
}

static int
linux_malloc_mbind(struct aml_area *area,
		     void           **ptr,
		     size_t           size,
		     size_t           alignement)
{
	int err = linux_malloc(area, ptr, size, alignement);
	if(err != AML_AREA_SUCCESS)
		return err;
#ifdef HAVE_LINUX_NUMA
	return linux_mbind(area->data, *ptr, size);
#else
	return AML_AREA_SUCCESS;
#endif
}

int
linux_free(__attribute__ ((unused)) struct aml_area *area,
	   void *ptr)	
{
	free(ptr);
	return AML_AREA_SUCCESS;
}


/*********************************************************************************
 * Areas declaration
 *********************************************************************************/

static struct aml_area_ops aml_area_linux_private_ops_s = {
	.map = linux_mmap_private_mbind,
	.unmap = linux_munmap,
	.malloc = linux_malloc_mbind,
	.free = linux_free,
#ifdef HAVE_LINUX_NUMA
	.create = linux_create,
	.destroy = linux_destroy,	
	.bind = linux_bind,
	.check_binding = linux_check_binding
#else
	.create = NULL,
	.destroy = NULL,
	.bind = NULL,
	.check_binding = NULL
#endif
};

static struct aml_area_ops aml_area_linux_shared_ops_s = {
	.map = linux_mmap_shared_mbind,
	.unmap = linux_munmap,
	.malloc = NULL,
	.free = NULL,
#ifdef HAVE_LINUX_NUMA
	.create = linux_create,
	.destroy = linux_destroy,
	.bind = linux_bind,
	.check_binding = linux_check_binding
#else
	.create = NULL,
	.destroy = NULL,
	.bind = NULL,
	.check_binding = NULL
#endif
};
	
static struct aml_area aml_area_linux_private_s = {
	.ops = &aml_area_linux_private_ops_s,
#ifdef HAVE_LINUX_NUMA	
	.data = &linux_binding_default
#else
	.data = NULL
#endif
};


static struct aml_area aml_area_linux_shared_s = {
	.ops = &aml_area_linux_shared_ops_s,
#ifdef HAVE_LINUX_NUMA	
	.data = &linux_binding_default
#else
	.data = NULL
#endif
};

struct aml_area *aml_area_linux_shared = &aml_area_linux_shared_s;
struct aml_area *aml_area_linux_private = &aml_area_linux_private_s;

