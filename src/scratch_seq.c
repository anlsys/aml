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

struct aml_scratch_seq_ops aml_scratch_seq_inner_ops = {
	aml_scratch_seq_doit,
	aml_scratch_seq_add_request,
	aml_scratch_seq_remove_request,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

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
		//scratch->ops.scratch2src(scratch->data, scratchid, srcid);
		*srcid = scratchid;

		/* init request */
		aml_scratch_request_seq_init(req, type, scratch->data.srctiling,
					     srcptr, *srcid,
					     scratch->data.scratchtiling,
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
		//scratch->ops.src2scratch(scratch->data, scratchid, srcid);
		*scratchid = srcid;

		/* init request */
		aml_scratch_request_seq_init(req, type,
					     scratch->data.scratchtiling,
					     scratchptr, *scratchid,
					     scratch->data.srctiling,
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
	return 0;
}

struct aml_scratch_ops aml_scratch_seq_ops = {
	aml_scratch_seq_create_request,
	aml_scratch_seq_destroy_request,
	aml_scratch_seq_wait_request,
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

	scratch->data.scratcharea = va_arg(ap, struct aml_area *);
	scratch->data.scratchtiling = va_arg(ap, struct aml_tiling *);
	scratch->data.srcarea = va_arg(ap, struct aml_area *);
	scratch->data.srctiling = va_arg(ap, struct aml_tiling *);
	scratch->data.dma = va_arg(ap, struct aml_dma *);
	
	/* allocate request array */
	scratch->data.nbrequests = va_arg(ap, size_t);
	scratch->data.requests = calloc(scratch->data.nbrequests,
				     sizeof(struct aml_scratch_request_seq));
	for(int i = 0; i < scratch->data.nbrequests; i++)
		scratch->data.requests[i].type = AML_SCRATCH_REQUEST_TYPE_INVALID;
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
	return 0;
}
