#include <aml.h>

/*******************************************************************************
 * COLUMN OPERATORS:
 ******************************************************************************/

void *aml_layout_column_deref(const struct aml_layout_data *d, va_list coords)
{
	return NULL;
}

void *aml_layout_column_aderef(const struct aml_layout_data *d, size_t *coords)
{
	return NULL;
}

int aml_layout_column_order(const struct aml_layout_data *d)
{
	return 0;
}

int aml_layout_column_dims(const struct aml_layout_data *d, va_list dims)
{
	return 0;
}

int aml_layout_column_adims(const struct aml_layout_data *d, const size_t *dims)
{
	return 0;
}

struct aml_layout_ops aml_layout_column_ops = {
	aml_layout_column_deref,
	aml_layout_column_aderef,
	aml_layout_column_order,
	aml_layout_column_dims,
	aml_layout_column_adims,
};


/*******************************************************************************
 * ROW OPERATORS:
 ******************************************************************************/

void *aml_layout_row_deref(const struct aml_layout_data *d, va_list coords)
{
	return NULL;
}

void *aml_layout_row_aderef(const struct aml_layout_data *d, size_t *coords)
{
	return NULL;
}

int aml_layout_row_order(const struct aml_layout_data *d)
{
	return 0;
}

int aml_layout_row_dims(const struct aml_layout_data *d, va_list dims)
{
	return 0;
}

int aml_layout_row_adims(const struct aml_layout_data *d, const size_t *dims)
{
	return 0;
}

struct aml_layout_ops aml_layout_row_ops = {
	aml_layout_row_deref,
	aml_layout_row_aderef,
	aml_layout_row_order,
	aml_layout_row_dims,
	aml_layout_row_adims,
};

