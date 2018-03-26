#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * Sequential scratchpad
 * The scratch itself is organized into several different components
 * - request types: push and pull
 * - implementation of the request
 * - user API (i.e. generic request creation and call)
 * - how to init the scratch
 ******************************************************************************/

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_scratch_request_seq_init(struct aml_scratch_request_seq *req, int type,
				 struct aml_tiling *dt, void *dstptr, int dstid,
				 struct aml_tiling *st, void *srcptr, int srcid)

{
	assert(req != NULL);
	req->type = type;
	req->stiling = st;
	req->srcptr = srcptr;
	req->srcid = srcid;
	req->dtiling = dt;
	req->dstptr = dstptr;
	req->dstid = dstid;
	return 0;
}

int aml_scratch_request_seq_destroy(struct aml_scratch_request_seq *r)
{
	assert(r != NULL);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/
int aml_scratch_seq_doit(struct aml_scratch_seq_data *scratch,
			      struct aml_scratch_request_seq *req)
{
	assert(scratch != NULL);
	assert(req != NULL);
	return aml_dma_async_copy(scratch->dma, &req->dma_req,
				  req->dtiling, req->dstptr, req->dstid,
				  req->stiling, req->srcptr, req->srcid);
}

int aml_scratch_seq_add_request(struct aml_scratch_seq_data *data,
				struct aml_scratch_request_seq **req)
{
	for(int i = 0; i < data->nbrequests; i++)
	{
		if(data->requests[i].type == AML_SCRATCH_REQUEST_TYPE_INVALID)
		{
			*req = &data->requests[i];
			return 0;
		}
	}
	/* TODO: slow path, need to resize the array */
	return 0;
}

int aml_scratch_seq_remove_request(struct aml_scratch_seq_data *data,
				     struct aml_scratch_request_seq **req)
{
	/* TODO: assert that the pointer is in the right place */
	(*req)->type = AML_SCRATCH_REQUEST_TYPE_INVALID;
	return 0;
}

int aml_scratch_seq_tilemap_find(struct aml_scratch_seq_data *data, int tid)
{
	assert(data != NULL);
	for(int i = 0; i < data->sch_nbtiles; i++)
	{
		if(data->tilemap[i] == tid)
			return i;
	}
	return -1;
}

struct aml_scratch_seq_ops aml_scratch_seq_inner_ops = {
	aml_scratch_seq_doit,
	aml_scratch_seq_add_request,
	aml_scratch_seq_remove_request,
	aml_scratch_seq_tilemap_find,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

/* TODO: not thread-safe */

int aml_scratch_seq_create_request(struct aml_scratch_data *d,
				   struct aml_scratch_request **r,
				   int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch =
		(struct aml_scratch_seq *)d;

	struct aml_scratch_request_seq *req;

	/* find an available request slot */
	scratch->ops.add_request(&scratch->data, &req);

	/* init the request */
	if(type == AML_SCRATCH_REQUEST_TYPE_PUSH)
	{
		int scratchid;
		int *srcid;
		void *srcptr;
		void *scratchptr;

		srcptr = va_arg(ap, void *);
		srcid = va_arg(ap, int *);
		scratchptr = va_arg(ap, void *);
		scratchid = va_arg(ap, int);

		/* find destination tile */
		assert(scratchid < scratch->data.sch_nbtiles);
		int slot = scratch->data.tilemap[scratchid];
		assert(slot != -1);
		*srcid = slot;

		/* init request */
		aml_scratch_request_seq_init(req, type, scratch->data.tiling,
					     srcptr, *srcid,
					     scratch->data.tiling,
					     scratchptr, scratchid);
	}
	else if(type == AML_SCRATCH_REQUEST_TYPE_PULL)
	{
		int *scratchid;
		int srcid;
		void *srcptr;
		void *scratchptr;

		scratchptr = va_arg(ap, void *);
		scratchid = va_arg(ap, int *);
		srcptr = va_arg(ap, void *);
		srcid = va_arg(ap, int);

		/* find destination tile */
		int slot = scratch->ops.tilemap_find(&scratch->data, srcid);
		if(slot == -1)
			slot = scratch->ops.tilemap_find(&scratch->data, -1);
		assert(slot != -1);
		scratch->data.tilemap[slot] = srcid;
		*scratchid = slot;

		/* init request */
		aml_scratch_request_seq_init(req, type,
					     scratch->data.tiling,
					     scratchptr, *scratchid,
					     scratch->data.tiling,
					     srcptr, srcid);
	}
	scratch->ops.doit(&scratch->data, req);

	*r = (struct aml_scratch_request *)req;
	return 0;
}

int aml_scratch_seq_destroy_request(struct aml_scratch_data *d,
					 struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch =
		(struct aml_scratch_seq *)d;

	struct aml_scratch_request_seq *req =
		(struct aml_scratch_request_seq *)r;

	aml_dma_cancel(scratch->data.dma, req->dma_req);
	aml_scratch_request_seq_destroy(req);
	scratch->ops.remove_request(&scratch->data, &req);
	if(req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
		scratch->data.tilemap[req->srcid] = -1;
	else if(req->type == AML_SCRATCH_REQUEST_TYPE_PULL)
		scratch->data.tilemap[req->dstid] = -1;
	return 0;
}

int aml_scratch_seq_wait_request(struct aml_scratch_data *d,
				   struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d;
	struct aml_scratch_request_seq *req =
		(struct aml_scratch_request_seq *)r;

	/* wait for completion of the request */
	aml_dma_wait(scratch->data.dma, req->dma_req);

	/* destroy a completed request */
	aml_scratch_request_seq_destroy(req);
	scratch->ops.remove_request(&scratch->data, &req);
	if(req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
		scratch->data.tilemap[req->srcid] = -1;
	else if(req->type == AML_SCRATCH_REQUEST_TYPE_PULL)
		scratch->data.tilemap[req->dstid] = -1;
	return 0;
}

void *aml_scratch_seq_baseptr(struct aml_scratch_data *d)
{
	assert(d != NULL);
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d;
	return scratch->data.sch_ptr;
}

struct aml_scratch_ops aml_scratch_seq_ops = {
	aml_scratch_seq_create_request,
	aml_scratch_seq_destroy_request,
	aml_scratch_seq_wait_request,
	aml_scratch_seq_baseptr,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_scratch_seq_create(struct aml_scratch **d, ...)
{
	va_list ap;
	struct aml_scratch *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, d);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_SCRATCH_SEQ_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_scratch);

	ret = (struct aml_scratch *)baseptr;
	ret->data = (struct aml_scratch_data *)dataptr;

	aml_scratch_seq_vinit(ret, ap);

	va_end(ap);
	*d = ret;
	return 0;
}
int aml_scratch_seq_vinit(struct aml_scratch *d, va_list ap)
{
	d->ops = &aml_scratch_seq_ops;
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d->data;

	scratch->ops = aml_scratch_seq_inner_ops;

	scratch->data.sch_area = va_arg(ap, struct aml_area *);
	scratch->data.src_area = va_arg(ap, struct aml_area *);
	scratch->data.dma = va_arg(ap, struct aml_dma *);
	scratch->data.tiling = va_arg(ap, struct aml_tiling *);
	scratch->data.sch_nbtiles = va_arg(ap, size_t);

	/* allocate request array */
	scratch->data.nbrequests = va_arg(ap, size_t);
	scratch->data.requests = calloc(scratch->data.nbrequests,
				     sizeof(struct aml_scratch_request_seq));
	for(int i = 0; i < scratch->data.nbrequests; i++)
		scratch->data.requests[i].type = AML_SCRATCH_REQUEST_TYPE_INVALID;

	/* scratch init */
	scratch->data.tilemap = calloc(scratch->data.sch_nbtiles, sizeof(int));
	for(int i = 0; i < scratch->data.sch_nbtiles; i++)
		scratch->data.tilemap[i] = -1;
	size_t tilesize = aml_tiling_tilesize(scratch->data.tiling, 0);
	scratch->data.sch_ptr = aml_area_calloc(scratch->data.sch_area,
						scratch->data.sch_nbtiles,
						tilesize);
	return 0;
}
int aml_scratch_seq_init(struct aml_scratch *d, ...)
{
	int err;
	va_list ap;
	va_start(ap, d);
	err = aml_scratch_seq_vinit(d, ap);
	va_end(ap);
	return err;
}

int aml_scratch_seq_destroy(struct aml_scratch *d)
{
	struct aml_scratch_seq *scratch = (struct aml_scratch_seq *)d->data;
	free(scratch->data.requests);
	free(scratch->data.tilemap);
	aml_area_free(scratch->data.sch_area, scratch->data.sch_ptr);
	return 0;
}
