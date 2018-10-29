#include <aml.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <citerators.h>

struct aml_tiling_nd_ops {
};

struct aml_tiling_nd_ops aml_tiling_nd_ops_s;

struct aml_smart_pointer {
	void *ptr;
	size_t ndims;
	size_t *dims;
	size_t *pitch;
};

struct aml_tiling_nd_data {
	struct aml_smart_pointer sptr;
	size_t *tile_dims;
};

struct aml_tiling_nd {
	struct aml_tiling_nd_ops *ops;
	struct aml_tiling_nd_data *data;
};


struct aml_tiling_nd_iterator_ops {
};

struct aml_tiling_nd_iterator_ops aml_tiling_nd_iterator_ops_s;

struct aml_tiling_nd_iterator_data {
	citerator_t inner_it;
	struct aml_tiling_nd_data *tiling;
};

struct aml_tiling_nd_iterator {
	struct aml_tiling_nd_iterator_ops *ops;
	struct aml_tiling_nd_iterator_data *data;
};

#define AML_SMART_POINTER_ALLOCSIZE(ndims) (sizeof(struct aml_smart_pointer) +\
					    ndims * 2 * sizeof(size_t))

#define AML_TILING_ND_ALLOCSIZE(ndims) (sizeof(struct aml_tiling_nd) +\
					sizeof(struct aml_tiling_nd_data) +\
					ndims * 3 * sizeof(size_t))

#define AML_TILING_ND_ITERATOR_ALLOCSIZE \
				(sizeof(struct aml_tiling_nd_iterator) +\
				 sizeof(struct aml_tiling_nd_iterator_data))

int aml_smart_pointer_create(struct aml_smart_pointer **p, void *ptr,
			     size_t ndims, const size_t *dims,
			     const size_t *pitch)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_SMART_POINTER_ALLOCSIZE(ndims));
	struct aml_smart_pointer *ret = (struct aml_smart_pointer *)baseptr;
	baseptr = (void *)((uintptr_t)baseptr +
		      sizeof(struct aml_smart_pointer));
	ret->ptr = ptr;
	ret->ndims = ndims;
	ret->dims = (size_t *)baseptr;
	ret->pitch = ret->dims + ndims;

	memcpy(ret->dims, dims, ndims * sizeof(size_t));
	memcpy(ret->pitch, pitch, ndims * sizeof(size_t));
	*p = ret;
	return 0;
}

int aml_tiling_nd_create(struct aml_tiling_nd **t, void *ptr, size_t ndims,
			 const size_t *dims, const size_t *pitch,
			 const size_t *tile_dims)
{
	assert(ndims > 0);
	void *baseptr = calloc(1, AML_TILING_ND_ALLOCSIZE(ndims));
	struct aml_tiling_nd *ret = (struct aml_tiling_nd *)baseptr;
	baseptr = (void *)((uintptr_t)baseptr + sizeof(struct aml_tiling_nd));
	ret->ops = &aml_tiling_nd_ops_s;
	ret->data = (struct aml_tiling_nd_data *)baseptr;
	baseptr = (void *)((uintptr_t)baseptr +
		      sizeof(struct aml_tiling_nd_data));

	ret->data->sptr.ptr = ptr;
	ret->data->sptr.ndims = ndims;
	ret->data->sptr.dims = (size_t *)baseptr;
	ret->data->sptr.pitch = ret->data->sptr.dims + ndims;
	ret->data->tile_dims = ret->data->sptr.pitch + ndims;

	memcpy(ret->data->sptr.dims, dims, ndims * sizeof(size_t));
	memcpy(ret->data->sptr.pitch, pitch, ndims * sizeof(size_t));
	memcpy(ret->data->tile_dims, tile_dims, ndims * sizeof(size_t));
        *t = ret;
	return 0;
}

int aml_tiling_nd_destroy(struct aml_tiling_nd *t)
{
	free(t);
	return 0;
}

static inline size_t aml_tiling_nd_get_ith_it_space(
					const struct aml_tiling_nd *t,
					size_t i)
{
	assert(i < t->data->sptr.ndims);
	size_t res = t->data->sptr.dims[i] / t->data->tile_dims[i];
	if (t->data->sptr.dims[i] % t->data->tile_dims[i] != 0)
		res += 1;
	return res;
}

int aml_tiling_nd_init_iterator(struct aml_tiling_nd *tiling,
				struct aml_tiling_nd_iterator *iterator,
				int flags)
{
	citerator_t inner_it = citerator_alloc(CITERATOR_PRODUCT);
	for (int i; i < tiling->data->sptr.ndims; i++) {
		citerator_t tmp_it = citerator_alloc(CITERATOR_RANGE);
		size_t it_space = aml_tiling_nd_get_ith_it_space(tiling, i);
		citerator_range_init(tmp_it, 0, it_space - 1, 1);
		citerator_product_add(inner_it, tmp_it);
	}
	iterator->data->inner_it = inner_it;
	iterator->data->tiling = tiling->data;
	return 0;
}

int aml_tiling_nd_create_iterator( struct aml_tiling_nd *tiling,
				   struct aml_tiling_nd_iterator **iterator,
				   int flags)
{
	void *baseptr = calloc(1, AML_TILING_ND_ITERATOR_ALLOCSIZE);
	struct aml_tiling_nd_iterator *ret = (struct aml_tiling_nd_iterator *)
						 baseptr;
	baseptr = (void *)((uintptr_t)baseptr +
		      sizeof(struct aml_tiling_nd_iterator));
	ret->ops = &aml_tiling_nd_iterator_ops_s;
	ret->data = (struct aml_tiling_nd_iterator_data *)baseptr;

	aml_tiling_nd_init_iterator(tiling, ret, flags);
	*iterator = ret;
	return 0;
}

int aml_tiling_nd_destroy_iterator( struct aml_tiling_nd *tiling,
				    struct aml_tiling_nd_iterator *iterator)
{
	citerator_free(iterator->data->inner_it);
	free(iterator);
	return 0;
}
