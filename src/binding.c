#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Binding functions
 * Most of the stuff is dispatched to a different layer, using type-specific
 * functions.
 ******************************************************************************/

int aml_binding_nbpages(struct aml_binding *binding,
			struct aml_tiling *tiling, void *ptr, ...)
{
	assert(binding != NULL);
	assert(tiling != NULL);
	va_list ap;
	int ret;
	va_start(ap, ptr);
	ret = binding->ops->nbpages(binding->data, tiling, ptr, ap);
	va_end(ap);
	return ret;
}

int aml_binding_pages(struct aml_binding *binding,
		      void **pages, struct aml_tiling *tiling, void *ptr, ...)
{
	assert(binding != NULL);
	assert(pages != NULL);
	assert(tiling != NULL);
	va_list ap;
	int ret;
	va_start(ap, ptr);
	ret = binding->ops->pages(binding->data, pages, tiling, ptr, ap);
	va_end(ap);
	return ret;
}

int aml_binding_nodes(struct aml_binding *binding,
		      int *nodes, struct aml_tiling *tiling, void *ptr, ...)
{
	assert(binding != NULL);
	assert(nodes != NULL);
	assert(tiling != NULL);
	va_list ap;
	int ret;
	va_start(ap, ptr);
	ret = binding->ops->nodes(binding->data, nodes, tiling, ptr, ap);
	va_end(ap);
	return ret;
}

/*******************************************************************************
 * Init functions
 ******************************************************************************/

/* allocate and init the binding according to type */
int aml_binding_create(struct aml_binding **b, int type, ...)
{
	va_list ap;
	va_start(ap, type);
	struct aml_binding *ret = NULL;
	intptr_t baseptr, dataptr;
	if(type == AML_BINDING_TYPE_SINGLE)
	{
		/* alloc */
		baseptr = (intptr_t) calloc(1, AML_BINDING_SINGLE_ALLOCSIZE);
		dataptr = baseptr + sizeof(struct aml_binding);

		ret = (struct aml_binding *)baseptr;
		ret->data = (struct aml_binding_data *)dataptr;

		aml_binding_vinit(ret, type, ap);
	}
	else if(type == AML_BINDING_TYPE_INTERLEAVE)
	{
		/* alloc */
		baseptr = (intptr_t) calloc(1, AML_BINDING_INTERLEAVE_ALLOCSIZE);
		dataptr = baseptr + sizeof(struct aml_binding);

		ret = (struct aml_binding *)baseptr;
		ret->data = (struct aml_binding_data *)dataptr;

		aml_binding_vinit(ret, type, ap);
	}
	va_end(ap);
	*b = ret;
	return 0;
}

int aml_binding_vinit(struct aml_binding *b, int type, va_list ap)
{
	if(type == AML_BINDING_TYPE_SINGLE)
	{
		b->ops = &aml_binding_single_ops;
		struct aml_binding_single_data *data =
			(struct aml_binding_single_data *)b->data;
		data->node = va_arg(ap, int);
	}
	else if(type == AML_BINDING_TYPE_INTERLEAVE)
	{
		b->ops = &aml_binding_interleave_ops;
		struct aml_binding_interleave_data *data =
			(struct aml_binding_interleave_data *)b->data;
		/* receive a nodemask, transform into a list of nodes */
		unsigned long *mask = va_arg(ap, unsigned long*);
		data->count = 0;
		for(int i = 0; i < AML_MAX_NUMA_NODES; i++)
			if(AML_NODEMASK_ISSET(mask, i))
			{
				data->nodes[data->count] = i;
				data->count++;
			}
	}
	return 0;
}

int aml_binding_init(struct aml_binding *b, int type, ...)
{
	int err;
	va_list ap;
	va_start(ap, type);
	err = aml_binding_vinit(b, type, ap);
	va_end(ap);
	return err;
}

int aml_binding_destroy(struct aml_binding *b, int type)
{
	return 0;
}

