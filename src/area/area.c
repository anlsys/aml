#include <stdlib.h>
#include "aml.h"
#include "aml/area/area.h"
#include "aml/utils/bitmap.h"

extern struct aml_area_ops aml_area_host_private_ops;
extern struct aml_area_ops aml_area_host_shared_ops;

struct aml_area*
aml_local_area_create(struct aml_area    *a,
		      const aml_bitmap    binding,
		      const unsigned long flags)
{
	struct aml_area* area;
		
	if(a == NULL || binding == NULL || a->ops==NULL || a->ops->bind == NULL)
		return NULL;
	
	area = malloc(sizeof(*area));
	
	if(area == NULL)
		return NULL;

	area->data = NULL;
	
	area->ops  = a->ops;
	
	if(a->ops->create && a->ops->create(area) != AML_AREA_SUCCESS)
		goto err_with_area;
	
        if(area->ops->bind(area, binding, flags) != AML_AREA_SUCCESS)
		goto err_with_area_data;
	
	return area;

 err_with_area_data:
	if(a->ops->create && a->ops->destroy)
		a->ops->destroy(area);
 err_with_area:
	free(area);
	return NULL;
}

void
aml_local_area_destroy(struct aml_area* area)
{
	if(area == NULL)
		return;

	if(area->ops && area->ops->destroy)
		area->ops->destroy(area);

	free(area);
}

int
aml_area_mmap(struct aml_area *area,
	      void           **ptr,
	      size_t           size)
{
	if(ptr == NULL)
		return AML_AREA_EINVAL;
	
	if(size == 0){
		*ptr = NULL;
		return AML_AREA_SUCCESS;
	}
	
	if(area == NULL)
		return AML_AREA_EINVAL;
	
	if(area->ops->map == NULL)
		return AML_AREA_ENOTSUP;

	return area->ops->map(area, ptr, size);
}

int
aml_area_munmap(struct aml_area *area,
		void            *ptr,
		size_t           size)
{
	if(ptr == NULL || size == 0)
		return AML_AREA_SUCCESS;
	
	if(area == NULL)
		return AML_AREA_EINVAL;
	
	if(area->ops->unmap == NULL)
		return AML_AREA_ENOTSUP;
	
	return area->ops->unmap(area, ptr, size);
}

int
aml_area_malloc(struct aml_area *area,
		void           **ptr,
		size_t           size,
		size_t           alignement)
{
	if(area == NULL)
		return AML_AREA_EINVAL;
	
	if(area->ops->malloc == NULL)
		return AML_AREA_ENOTSUP;

	if(ptr == NULL)
		return AML_AREA_EINVAL;
	
	if(size == 0){
		*ptr = NULL;
		return AML_AREA_SUCCESS;
	}
	
	return area->ops->malloc(area, ptr, size, alignement);
}

int
aml_area_free(struct aml_area *area,
	      void            *ptr)
{
	if(area == NULL)
		return AML_AREA_EINVAL;
	
	if(area->ops->free == NULL)
		return AML_AREA_ENOTSUP;

	if(ptr == NULL)
		return AML_AREA_SUCCESS;
	
	return area->ops->free(area, ptr);
}

