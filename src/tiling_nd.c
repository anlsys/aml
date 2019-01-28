#include <aml.h>

struct aml_layout *aml_tiling_nd_index(const struct aml_tiling_nd *t, ...)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	va_list ap;
	struct aml_layout *ret;
	va_start(ap, t);
	ret = t->ops->index(t->data, ap);
        va_end(ap);
	return ret;
}

struct aml_layout *aml_tiling_nd_aindex(const struct aml_tiling_nd *t, const size_t *coords)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	return t->ops->aindex(t->data, coords);
}

int aml_tiling_nd_order(const struct aml_tiling_nd *t)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	return t->ops->order(t->data);
}

int aml_tiling_nd_tile_dims(const struct aml_tiling_nd *t, ...)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	va_list ap;
	int ret;
	va_start(ap, t);
	ret = t->ops->tile_dims(t->data, ap);
	va_end(ap);
	return ret;
}

int aml_tiling_nd_tile_adims(const struct aml_tiling_nd *t, size_t *dims)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	return t->ops->tile_adims(t->data, dims);
}

int aml_tiling_nd_dims(const struct aml_tiling_nd *t, ...)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	va_list ap;
	int ret;
	va_start(ap, t);
	ret = t->ops->dims(t->data, ap);
	va_end(ap);
	return ret;
}

int aml_tiling_nd_adims(const struct aml_tiling_nd *t, size_t *dims)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	return t->ops->adims(t->data, dims);
}

size_t aml_tiling_nd_ndims(const struct aml_tiling_nd *t)
{
	assert(t != NULL);
	assert(t->ops != NULL);
	return t->ops->ndims(t->data);
}
