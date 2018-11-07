#include <aml.h>

int aml_layout_struct_init(struct aml_layout *p,
					 size_t ndims, void *data_ptr)
{
	p->ndims = ndims;
	p->dims = (size_t *)data_ptr;
	p->pitch = p->dims + ndims;
	p->stride = p->pitch + ndims;
	return 0;
}

int aml_layout_init(struct aml_layout *p, void *ptr,
				  size_t ndims, const size_t *dims,
				  const size_t *pitch,
				  const size_t *stride)
{
	assert(p->ndims == ndims);
	assert(p->dims);
	assert(p->pitch);
	assert(p->stride);
	p->ptr = ptr;
	memcpy(p->dims, dims, ndims * sizeof(size_t));
	memcpy(p->pitch, pitch, ndims * sizeof(size_t));
	memcpy(p->stride, stride, ndims * sizeof(size_t));
	return 0;
}

int aml_layout_create(struct aml_layout **p, void *ptr,
			     size_t ndims, const size_t *dims,
			     const size_t *pitch,
			     const size_t *stride)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_LAYOUT_ALLOCSIZE(ndims));
	*p = (struct aml_layout *)baseptr;
	baseptr = (void *)((uintptr_t)baseptr +
		      sizeof(struct aml_layout));
	aml_layout_struct_init(*p, ndims, baseptr);
	aml_layout_init(*p, ptr, ndims, dims, pitch, stride);
	return 0;
}
