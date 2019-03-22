#include "aml.h"
#include "aml/area/area.h"
#include "aml/area/linux.h"
#include "aml/utils/hwloc.h"
#include "aml/area/hwloc.h"
#include "aml/area/linux.h"

#define HWLOC_BINDING_FLAGS (HWLOC_MEMBIND_PROCESS|	\
			     HWLOC_MEMBIND_MIGRATE|	\
			     HWLOC_MEMBIND_NOCPUBIND|	\
			     HWLOC_MEMBIND_BYNODESET)

extern hwloc_topology_t aml_topology;

const unsigned long aml_area_hwloc_flag_bind = HWLOC_MEMBIND_BIND;
const unsigned long aml_area_hwloc_flag_interleave = HWLOC_MEMBIND_INTERLEAVE;
const unsigned long aml_area_hwloc_flag_firsttouch = HWLOC_MEMBIND_FIRSTTOUCH;
const unsigned long aml_area_hwloc_flag_nexttouch = HWLOC_MEMBIND_NEXTTOUCH;

int
aml_area_hwloc_create(struct aml_area* area)
{
	int err = AML_AREA_SUCCESS;
	struct hwloc_binding *binding;
	   
	binding = malloc(sizeof(struct hwloc_binding));
	if(binding == NULL)
		return AML_AREA_ENOMEM;
	
	binding->nodeset = hwloc_bitmap_alloc();
	if(binding->nodeset == NULL){
		err = AML_AREA_ENOMEM;
		goto err_with_binding;
	}
		
	binding->policy = HWLOC_MEMBIND_DEFAULT;
	
	struct hwloc_binding * area_binding = area->data;
	binding->ops.create = NULL;
	binding->ops.destroy = NULL;
	binding->ops.bind = NULL;
	binding->ops.check_binding = NULL;
	if(area_binding != NULL){
		binding->ops.map = area_binding->ops.map;
		binding->ops.unmap = area_binding->ops.unmap;
		binding->ops.malloc = area_binding->ops.malloc;
		binding->ops.free = area_binding->ops.free;
	} else {
		binding->ops.map = aml_area_linux_mmap_private;
		binding->ops.unmap = aml_area_linux_munmap;
		binding->ops.malloc = aml_area_linux_malloc;
		binding->ops.free = aml_area_linux_free;
	}
	area->data = binding;

	return AML_AREA_SUCCESS;
	
 err_with_binding:
	free(binding);
	return err;
}

void
aml_area_hwloc_destroy(struct aml_area* area)
{
	struct hwloc_binding *binding = area->data;
	hwloc_bitmap_free(binding->nodeset);	
        free(binding);	
}

static int
aml_hwloc_bind(struct aml_area             *area,
	       const hwloc_bitmap_t        *nodeset,
	       const hwloc_membind_policy_t hwloc_policy)
{
	const struct hwloc_topology_support * sup =
		hwloc_topology_get_support(aml_topology);

	if(!sup->discovery->numa || !sup->discovery->numa_memory)
		return AML_AREA_ENOTSUP;

	if(!sup->membind->set_area_membind || !sup->membind->alloc_membind)
		return AML_AREA_ENOTSUP;

	if(hwloc_policy == HWLOC_MEMBIND_BIND &&
	   !sup->membind->bind_membind)
		return AML_AREA_ENOTSUP;

	if(hwloc_policy == HWLOC_MEMBIND_FIRSTTOUCH &&
	   !sup->membind->firsttouch_membind)
		return AML_AREA_ENOTSUP;
	
	if(hwloc_policy == HWLOC_MEMBIND_INTERLEAVE &&
	   !sup->membind->interleave_membind)
		return AML_AREA_ENOTSUP;
	
	if(hwloc_policy == HWLOC_MEMBIND_NEXTTOUCH &&
	   !sup->membind->nexttouch_membind)
		return AML_AREA_ENOTSUP;

	const hwloc_bitmap_t allowed_nodeset =
		hwloc_topology_get_allowed_nodeset(aml_topology);
	
	if(nodeset && !hwloc_bitmap_isincluded(nodeset, allowed_nodeset))
		return AML_AREA_EDOM;

	if(hwloc_policy != HWLOC_MEMBIND_FIRSTTOUCH &&
	   hwloc_policy != HWLOC_MEMBIND_BIND       &&
	   hwloc_policy != HWLOC_MEMBIND_INTERLEAVE &&
	   hwloc_policy != HWLOC_MEMBIND_NEXTTOUCH)
		return AML_AREA_EINVAL;

	struct hwloc_binding *binding = area->data;

	if(binding->nodeset == NULL)
		binding->nodeset = hwloc_bitmap_alloc();
	if(binding->nodeset == NULL)
		return AML_AREA_ENOMEM;
	
	if(nodeset == NULL)
		hwloc_bitmap_copy(binding->nodeset, allowed_nodeset);
	else
		hwloc_bitmap_copy(binding->nodeset, nodeset);
	
	binding->policy = hwloc_policy;
	
	return AML_AREA_SUCCESS;
}

int
aml_area_hwloc_bind(struct aml_area         *area,
		    const struct aml_bitmap *bitmap,
		    const unsigned long      policy)
{
	hwloc_bitmap_t hwloc_bitmap = hwloc_bitmap_from_aml_bitmap(bitmap);

	int err = aml_hwloc_bind(area, hwloc_bitmap, policy);
	
	hwloc_bitmap_free(hwloc_bitmap);

	return err;
}

static int
aml_area_hwloc_apply_binding(struct hwloc_binding *binding,
			     void                 *ptr,
			     size_t                size)
{
	int err = AML_AREA_SUCCESS;
	hwloc_bitmap_t nodeset = binding->nodeset;
	
	if(nodeset == NULL)
		nodeset = hwloc_topology_get_allowed_nodeset(aml_topology);
	
	err = hwloc_set_area_membind(aml_topology,
				     ptr, size,
				     nodeset,
				     binding->policy,
				     HWLOC_BINDING_FLAGS);
	
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
	return err;
}

int
aml_area_hwloc_check_binding(struct aml_area *area,
			     void            *ptr,
			     size_t           size)
{
	int err;
	hwloc_membind_policy_t policy;
	hwloc_bitmap_t nodeset;
	struct hwloc_binding *bind = area->data;

        nodeset = hwloc_bitmap_alloc();

	if(nodeset == NULL)
		return AML_AREA_ENOMEM;
	
	err = hwloc_get_area_membind(aml_topology,
				     ptr, size,
				     nodeset,
				     &policy,
				     HWLOC_BINDING_FLAGS);
	
	if(err < 0){
		err = AML_AREA_EINVAL;
		goto out;
	}

	err = 1;

	if(policy != bind->policy)
	        err = 0;
	
	if(!hwloc_bitmap_isequal(nodeset, bind->nodeset))
		err = 0;
	
 out:
	hwloc_bitmap_free(nodeset);
	return err;
}

int
aml_area_hwloc_mmap(const struct aml_area *area,
		    void                      **ptr,
		    size_t                      size)
{
	struct hwloc_binding *binding = area->data;
        int err = binding->ops.map(area, ptr, size);

	if(err != AML_AREA_SUCCESS)
		return err;

	return aml_area_hwloc_apply_binding(binding, *ptr, size);
}

int
aml_area_hwloc_munmap(const struct aml_area *area,
		      void                  *ptr,
		      const size_t           size)
{
	struct hwloc_binding *binding = area->data;
        return binding->ops.unmap(area, ptr, size);
}

int
aml_area_hwloc_malloc(const struct aml_area *area,
		      void                 **ptr,
		      size_t                 size,
		      size_t                 alignement)
{
	struct hwloc_binding *binding = area->data;
	if(binding->ops.malloc == NULL)
		return AML_AREA_ENOTSUP;
	
        int err = binding->ops.malloc(area, ptr, size, alignement);
	
	if(err != AML_AREA_SUCCESS)
		return err;

	return aml_area_hwloc_apply_binding(binding, *ptr, size);
}

int
aml_area_hwloc_free(const struct aml_area *area,
		    void                  *ptr)	
{
	struct hwloc_binding *binding = area->data;
	if(binding->ops.free == NULL)
		return AML_AREA_ENOTSUP;
	return binding->ops.free(area, ptr);
}

/*********************************************************************************
 * Areas declaration
 *********************************************************************************/

struct hwloc_binding hwloc_binding_private = {
	.ops     = {
		.create = NULL,
		.destroy = NULL,
		.bind = NULL,
		.check_binding = NULL,
		.map = aml_area_linux_mmap_private,
		.unmap = aml_area_linux_munmap,
		.malloc = aml_area_linux_malloc,
		.free = aml_area_linux_free,
	},
	.nodeset = NULL,
	.policy  = HWLOC_MEMBIND_DEFAULT
};

struct hwloc_binding hwloc_binding_shared = {
	.ops     = {
		.create = NULL,
		.destroy = NULL,
		.bind = NULL,
		.check_binding = NULL,
		.map = aml_area_linux_mmap_shared,
		.unmap = aml_area_linux_munmap,
		.malloc = NULL,
		.free = NULL
	},
	.nodeset = NULL,
	.policy  = HWLOC_MEMBIND_DEFAULT
};

struct aml_area_ops aml_area_hwloc_ops = {
	.create = aml_area_hwloc_create,
	.destroy = aml_area_hwloc_destroy,
	.map = aml_area_hwloc_mmap,
	.unmap = aml_area_hwloc_munmap,
	.malloc = aml_area_hwloc_malloc,
	.free = aml_area_hwloc_free,
	.bind = aml_area_hwloc_bind,
	.check_binding = aml_area_hwloc_check_binding
};

struct aml_area aml_area_hwloc_private_s = {
	.ops = &aml_area_hwloc_ops,
	.data = &hwloc_binding_private
};

struct aml_area aml_area_hwloc_shared_s = {
	.ops = &aml_area_hwloc_ops,
	.data = &hwloc_binding_shared
};

struct aml_area *aml_area_hwloc_private = &aml_area_hwloc_private_s;
struct aml_area *aml_area_hwloc_shared  = &aml_area_hwloc_shared_s;

