#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <sys/mman.h>

/*******************************************************************************
 * Linux-backed, sequential dma
 * The dma itself is organized into several different components
 * - request types: copy or move
 * - implementation of the request
 * - user API (i.e. generic request creation and call)
 * - how to init the dma
 ******************************************************************************/

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_dma_request_linux_seq_copy(struct aml_dma_data *dma,
				   struct aml_dma_request_data *req)
{
	assert(dma != NULL);
	assert(req != NULL);
	struct aml_dma_request_linux_seq_data *data =
		(struct aml_dma_request_linux_seq_data*)req;
	memcpy(data->dest, data->src, data->size);
	return 0;
}

int aml_dma_request_linux_seq_move(struct aml_dma_data *dma,
				   struct aml_dma_request_data *req)
{
	assert(dma != NULL);
	assert(req != NULL);
	struct aml_dma_request_linux_seq_data *data =
		(struct aml_dma_request_linux_seq_data *)req;
	int status[data->count];
	int err;
	err = move_pages(0, data->count, data->pages, data->nodes, status,
			 MPOL_MF_MOVE);
	if(err)
	{
		perror("move_pages:");
		return errno;
	}
	return 0;
}

struct aml_dma_request_ops aml_dma_request_linux_seq_ops = {
	aml_dma_request_linux_seq_copy,
	aml_dma_request_linux_seq_move,
};

int aml_dma_request_linux_seq_copy_init(struct aml_dma_request *r,
					const struct aml_tiling *dt,
					void *dptr, int dtid,
					const struct aml_tiling *st,
					const void *sptr, int stid)
{
	assert(r != NULL);
	struct aml_dma_request_linux_seq_data *data =
		(struct aml_dma_request_linux_seq_data *)r->data;

	data->type = AML_DMA_REQUEST_TYPE_MOVE;
	/* figure out pointers */
	data->dest = aml_tiling_tilestart(dt, dptr, dtid);
	data->src = aml_tiling_tilestart(st, sptr, stid);
	data->size = aml_tiling_tilesize(st, stid);
	/* TODO: assert size match */
	return 0;
}

int aml_dma_request_linux_seq_copy_destroy(struct aml_dma_request *r)
{
	assert(r != NULL);
	return 0;
}

int aml_dma_request_linux_seq_move_init(struct aml_dma_request *r,
					struct aml_area *darea,
					const struct aml_tiling *tiling,
					void *startptr, int tileid)
{
	assert(r != NULL);
	struct aml_binding *binding;
	struct aml_dma_request_linux_seq_data *data =
		(struct aml_dma_request_linux_seq_data *)r->data;

	data->type = AML_DMA_REQUEST_TYPE_MOVE;
	aml_area_binding(darea, &binding);
	data->count = aml_binding_nbpages(binding, tiling, startptr, tileid);
	data->pages = calloc(data->count, sizeof(void *));
	data->nodes = calloc(data->count, sizeof(int));
	aml_binding_pages(binding, data->pages, tiling, startptr, tileid);
	aml_binding_nodes(binding, data->nodes, tiling, startptr, tileid);
	free(binding);
	return 0;
}

int aml_dma_request_linux_seq_move_destroy(struct aml_dma_request *r)
{
	struct aml_dma_request_linux_seq_data *data =
		(struct aml_dma_request_linux_seq_data *)r->data;
	free(data->pages);
	free(data->nodes);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/

int aml_dma_linux_seq_add_request(struct aml_dma_linux_seq_data *data,
				  struct aml_dma_request_linux_seq_data **req)
{
	for(int i = 0; i < data->size; i++)
	{
		if(data->requests[i].type == AML_DMA_REQUEST_TYPE_INVALID)
		{
			*req = &data->requests[i];
			return 0;
		}
	}
	/* TODO: slow path, need to resize the array */
	return 0;
}

int aml_dma_linux_seq_remove_request(struct aml_dma_linux_seq_data *data,
				     struct aml_dma_request_linux_seq_data **req)
{
	/* TODO: assert that the pointer is in the right place */
	(*req)->type = AML_DMA_REQUEST_TYPE_INVALID;
	return 0;
}

struct aml_dma_linux_seq_ops aml_dma_linux_seq_inner_ops = {
	aml_dma_linux_seq_add_request,
	aml_dma_linux_seq_remove_request,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_dma_linux_seq_create_request(struct aml_dma_data *d,
				     struct aml_dma_request *r,
				     int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;

	struct aml_dma_request_linux_seq_data *req;

	/* find an available request slot */
	dma->ops.add_request(&dma->data, &req);
	r->data = (struct aml_dma_request_data *)req;

	/* init the request */
	r->ops = &aml_dma_request_linux_seq_ops;
	if(type == AML_DMA_REQUEST_TYPE_COPY)
	{
		struct aml_tiling *dt, *st;
		void *dptr, *sptr;
		int dtid, stid;
		dt = va_arg(ap, struct aml_tiling *);
		dptr = va_arg(ap, void *);
		dtid = va_arg(ap, int);
		st = va_arg(ap, struct aml_tiling *);
		sptr = va_arg(ap, void *);
		stid = va_arg(ap, int);
		aml_dma_request_linux_seq_copy_init(r, dt, dptr, dtid,
						    st, sptr, stid);
	}
	else if(type == AML_DMA_REQUEST_TYPE_MOVE)
	{
		struct aml_area *darea = va_arg(ap, struct aml_area *);
		struct aml_tiling *st = va_arg(ap, struct aml_tiling *);
		void *sptr = va_arg(ap, void *);
		int stid = va_arg(ap, int);
		aml_dma_request_linux_seq_move_init(r, darea, st, sptr, stid);
	}
	return 0;
}

int aml_dma_linux_seq_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_seq *dma =
		(struct aml_dma_linux_seq *)d;

	struct aml_dma_request_linux_seq_data *req =
		(struct aml_dma_request_linux_seq_data *)r->data;

	if(req->type == AML_DMA_REQUEST_TYPE_COPY)
		aml_dma_request_linux_seq_copy_destroy(r);
	else if(req->type == AML_DMA_REQUEST_TYPE_MOVE)
		aml_dma_request_linux_seq_move_destroy(r);

	dma->ops.remove_request(&dma->data, &req);
	return 0;
}

int aml_dma_linux_seq_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_request_linux_seq_data *req =
		(struct aml_dma_request_linux_seq_data *)r->data;

	/* execute */
	if(req->type == AML_DMA_REQUEST_TYPE_COPY)
		r->ops->copy(d, r->data);
	else if(req->type == AML_DMA_REQUEST_TYPE_MOVE)
		r->ops->move(d, r->data);

	/* destroy a completed request */
	aml_dma_linux_seq_destroy_request(d, r);
	return 0;
}

struct aml_dma_ops aml_dma_linux_seq_ops = {
	aml_dma_linux_seq_create_request,
	aml_dma_linux_seq_destroy_request,
	aml_dma_linux_seq_wait_request,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_dma_linux_seq_create(struct aml_dma **d, ...)
{
	va_list ap;
	struct aml_dma *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, d);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_DMA_LINUX_SEQ_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_dma);

	ret = (struct aml_dma *)baseptr;
	ret->data = (struct aml_dma_data *)dataptr;

	aml_dma_linux_seq_vinit(ret, ap);

	va_end(ap);
	*d = ret;
	return 0;
}
int aml_dma_linux_seq_vinit(struct aml_dma *d, va_list ap)
{
	d->ops = &aml_dma_linux_seq_ops;
	struct aml_dma_linux_seq *dma = (struct aml_dma_linux_seq *)d->data;

	dma->ops = aml_dma_linux_seq_inner_ops;
	/* allocate request array */
	dma->data.size = va_arg(ap, size_t);
	dma->data.requests = calloc(dma->data.size,
				     sizeof(struct aml_dma_request_linux_seq_data));
	for(int i = 0; i < dma->data.size; i++)
		dma->data.requests[i].type = AML_DMA_REQUEST_TYPE_INVALID;
	return 0;
}
int aml_dma_linux_seq_init(struct aml_dma *d, ...)
{
	int err;
	va_list ap;
	va_start(ap, d);
	err = aml_dma_linux_seq_vinit(d, ap);
	va_end(ap);
	return err;
}

int aml_dma_linux_seq_destroy(struct aml_dma *d)
{
	struct aml_dma_linux_seq *dma = (struct aml_dma_linux_seq *)d->data;
	free(dma->data.requests);
	return 0;
}
