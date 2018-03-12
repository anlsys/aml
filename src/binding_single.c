#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Single Binding
 * All pages on the same node
 ******************************************************************************/

int aml_binding_single_getinfo(struct aml_binding_data *data,
			       intptr_t *start, intptr_t *end,
			       struct aml_tiling *tiling, void *ptr,
			       int tileid)
{
	assert(data != NULL);
	assert(tiling != NULL);
	size_t size = aml_tiling_tilesize(tiling, tileid);
	*start = (intptr_t) aml_tiling_tilestart(tiling, ptr, tileid);
	*end = *start + size;

	/* include first and last pages */
	*start -= *start % PAGE_SIZE;
	*end += (PAGE_SIZE - (*end % PAGE_SIZE)) % PAGE_SIZE;

	return (*end - *start) / PAGE_SIZE; 
}


int aml_binding_single_nbpages(struct aml_binding_data *data,
			       struct aml_tiling *tiling, void *ptr,
			       int tileid)
{
	assert(data != NULL);
	assert(tiling != NULL);
	intptr_t start, end;
	return aml_binding_single_getinfo(data, &start, &end, tiling, ptr, tileid);
}

int aml_binding_single_pages(struct aml_binding_data *data, void **pages,
			     struct aml_tiling *tiling, void *ptr,
			     int tileid)
{
	assert(data != NULL);
	assert(pages != NULL);
	assert(tiling != NULL);
	intptr_t start, end;
	int i, count;
	count = aml_binding_single_getinfo(data, &start, &end, tiling, ptr, tileid);
	for(i = 0; i < count; i++)
		pages[i] = (void *)(start + i * PAGE_SIZE);
	return 0;
}

int aml_binding_single_nodes(struct aml_binding_data *data, int *nodes,
			     struct aml_tiling *tiling, void *ptr,
			     int tileid)
{
	assert(data != NULL);
	assert(nodes != NULL);
	assert(tiling != NULL);
	struct aml_binding_single_data *binding =
		(struct aml_binding_single_data *)data;
	intptr_t start, end;
	int i, count;
	count = aml_binding_single_getinfo(data, &start, &end, tiling, ptr,
					   tileid);
	for(i = 0; i < count; i++)
		nodes[i] = binding->node;
	return 0;
}

struct aml_binding_ops aml_binding_single_ops = {
	aml_binding_single_nbpages,
	aml_binding_single_pages,
	aml_binding_single_nodes,
};
