#include <aml.h>
#include <assert.h>

memkind_t *type2kind[AML_AREA_TYPE_MAX] = {
	&MEMKIND_HBW_ALL,
	&MEMKIND_REGULAR,
};

/*******************************************************************************
 * memkind additional functions:
 * memkind is missing some features, that we add by re-implementing some of the
 * needed hooks.
 ******************************************************************************/

int aml_memkind_areanodemask(struct memkind *kind, unsigned long *nodemask,
			     unsigned long maxnode)
{
	/* transform back kind into an area */
	struct aml_area *area = (struct aml_area*)kind;
	struct bitmask ret = {maxnode, nodemask};
	copy_bitmask_to_bitmask(area->nodemask, &ret);
	return 0;
}


/*******************************************************************************
 * area implementation
 * At this point, use memkind internally to implement our stuff
 ******************************************************************************/

int aml_area_init(struct aml_area *area, unsigned int type)
{
	assert(type < AML_AREA_TYPE_MAX);
	area->kind = *type2kind[type];
	area->nodemask = numa_allocate_nodemask();
	copy_bitmask_to_bitmask(area->nodemask,numa_all_nodes_ptr);
	return 0;
}

int aml_area_from_nodestring(struct aml_area *area, unsigned int type,
			     const char *nodes)
{
	aml_area_init(area, type);
	area->nodemask = numa_parse_nodestring(nodes);
	return 0;
}

int aml_area_from_nodemask(struct aml_area *area, unsigned int type,
			   struct bitmask *nodes)
{
	aml_area_init(area, type);
	copy_bitmask_to_bitmask(area->nodemask, nodes);
	return 0;
}

int aml_area_destroy(struct aml_area *area)
{
	numa_bitmask_free(area->nodemask);
	return 0;
}

void *aml_area_malloc(struct aml_area *area, size_t size)
{
	return memkind_malloc(area->kind, size);
}

void aml_area_free(struct aml_area *area, void *ptr)
{
	memkind_free(area->kind, ptr);
}

void *aml_area_calloc(struct aml_area *area, size_t num, size_t size)
{
	return memkind_calloc(area->kind, num, size);
}

void *aml_area_realloc(struct aml_area *area, void *ptr, size_t size)
{
	return memkind_realloc(area->kind, ptr, size);
}

void *aml_area_acquire(struct aml_area *area, size_t size)
{
	/* as far as we know memkind doesn't zero new areas
	 * TODO: find a way to assert it
	 */
	return aml_area_malloc(area,size);
}

void aml_area_release(struct aml_area *area, void *ptr)
{
	/* As far as we know memkind doesn't decommit new areas
	 * TODO: find a way to assert it
	 */
	aml_area_free(area, ptr);
}

