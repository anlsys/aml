#include <aml.h>

/*******************************************************************************
 * General API: common operators:
 ******************************************************************************/

void *aml_layout_deref(const struct aml_layout *layout, ...)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	va_list ap;
	void *ret;
	va_start(ap, layout);
	ret = layout->ops->deref(layout->data, ap);
	va_end(ap);
	return ret;
}

void *aml_layout_aderef(const struct aml_layout *layout, const size_t *coords)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	return layout->ops->aderef(layout->data, coords);
}

int aml_layout_order(const struct aml_layout *layout)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	return layout->ops->order(layout->data);
}

int aml_layout_dims(const struct aml_layout *layout, ...)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	va_list ap;
	int ret;
	va_start(ap, layout);
	ret = layout->ops->dims(layout->data, ap);
	va_end(ap);
	return ret;
}

int aml_layout_adims(const struct aml_layout *layout, size_t *dims)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	return layout->ops->adims(layout->data, dims);
}

size_t aml_layout_ndims(const struct aml_layout *layout)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	return layout->ops->ndims(layout->data);
}

size_t aml_layout_element_size(const struct aml_layout *layout)
{
	assert(layout != NULL);
	assert(layout->ops != NULL);
	return layout->ops->element_size(layout->data);
}

