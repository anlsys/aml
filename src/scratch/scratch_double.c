#include "aml.h"
#include <assert.h>

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_scratch_request_double_init(struct aml_scratch_request_double *req,
				    int type, struct aml_dma *dma,
				    struct aml_layout *dl, int dstid,
				    struct aml_layout *sl, int srcid)

{
	assert(req != NULL);
	req->type = type;
	req->dma = dma;
	req->dest = dl;
	req->dstid = dstid;
	req->src = sl;
	req->srcid = srcid;
	return 0;
}

int aml_scratch_request_double_destroy(struct aml_scratch_request_double *r)
{
	assert(r != NULL);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/
void *aml_scratch_double_do_thread(void *arg)
{
	struct aml_scratch_request_double *req =
		(struct aml_scratch_request_double *)arg;

	aml_dma_copy(req->dma, req->dest, req->src);
}

struct aml_scratch_double_ops aml_scratch_double_inner_ops = {
	aml_scratch_double_do_thread,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_scratch_double_create_request(struct aml_scratch_data *d,
				   struct aml_scratch_request **r,
				   int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_double *scratch =
		(struct aml_scratch_double *)d;

	struct aml_scratch_request_double *req;

	pthread_mutex_lock(&scratch->data.lock);
	req = aml_vector_add(&scratch->data.requests);
	/* init the request */
	if(type == AML_SCRATCH_REQUEST_TYPE_PUSH)
	{
		struct aml_layout *scratch_layout;
		struct aml_layout *src_layout;
		int *src_uid;
		int scratch_uid;

		src_layout = va_arg(ap, struct aml_layout *);
		src_uid = va_arg(ap, int *);
		scratch_layout = va_arg(ap, struct aml_layout *);
		scratch_uid = va_arg(ap, int);

		/* find destination tile */
		int *slot = aml_vector_get(&scratch->data.tilemap, scratch_uid);
		assert(slot != NULL);
		*src_uid = *slot;

		/* init request */
		aml_scratch_request_double_init(req, type,
						scratch->data.push_dma,
						src_layout, *src_uid,
						scratch_layout, scratch_uid);
	}
	else if(type == AML_SCRATCH_REQUEST_TYPE_PULL)
	{
		struct aml_layout **scratch_layout;
		struct aml_layout *src_layout;
		int *scratch_uid;
		int src_uid;

		scratch_layout = va_arg(ap, struct aml_layout **);
		scratch_uid  = va_arg(ap, int *);
		src_layout = va_arg(ap, struct aml_layout *);
		src_uid = va_arg(ap, int);

		/* find scratchination tile
		 * We don't use add here because adding a tile means allocating
		 * new tiles on the sch_area too. */
		int slot = aml_vector_find(&scratch->data.tilemap, src_uid);
		if(slot == -1)
		{
			/* create a new request */
			slot = aml_vector_find(&scratch->data.tilemap, -1);
			assert(slot != -1);
			int *tile = aml_vector_get(&scratch->data.tilemap, slot);
			*tile = src_uid;
		}
		else
			type = AML_SCRATCH_REQUEST_TYPE_NOOP;

		/* save the key */
		*scratch_uid = slot;
		// *scratch_layout = aml_tiling_nd_get(scratch->data.scratch_tiling)

		/* init request */
		aml_scratch_request_double_init(req, type,
						scratch->data.pull_dma,
						*scratch_layout, slot,
						src_layout, src_uid);
	}
	pthread_mutex_unlock(&scratch->data.lock);
	/* thread creation */
	if(req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
	{
		pthread_create(&req->thread, NULL, scratch->ops.do_thread, req);
	}
	*r = (struct aml_scratch_request *)req;
	return 0;
}

int aml_scratch_double_destroy_request(struct aml_scratch_data *d,
					 struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_double *scratch =
		(struct aml_scratch_double *)d;

	struct aml_scratch_request_double *req =
		(struct aml_scratch_request_double *)r;
	int *tile;

	if(req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
	{
		pthread_cancel(req->thread);
		pthread_join(req->thread, NULL);
	}

	aml_scratch_request_double_destroy(req);

	/* destroy removes the tile from the scratch */
	pthread_mutex_lock(&scratch->data.lock);
	if(req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
		tile = aml_vector_get(&scratch->data.tilemap,req->srcid);
	else if(req->type == AML_SCRATCH_REQUEST_TYPE_PULL)
		tile = aml_vector_get(&scratch->data.tilemap,req->dstid);
	aml_vector_remove(&scratch->data.tilemap, tile);
	aml_vector_remove(&scratch->data.requests, req);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

int aml_scratch_double_wait_request(struct aml_scratch_data *d,
				   struct aml_scratch_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_scratch_double *scratch = (struct aml_scratch_double *)d;
	struct aml_scratch_request_double *req =
		(struct aml_scratch_request_double *)r;
	int *tile;

	/* wait for completion of the request */
	if(req->type != AML_SCRATCH_REQUEST_TYPE_NOOP)
		pthread_join(req->thread, NULL);

	/* cleanup a completed request. In case of push, free up the tile */
	aml_scratch_request_double_destroy(req);
	pthread_mutex_lock(&scratch->data.lock);
	if(req->type == AML_SCRATCH_REQUEST_TYPE_PUSH)
	{
		tile = aml_vector_get(&scratch->data.tilemap,req->srcid);
		aml_vector_remove(&scratch->data.tilemap, tile);
	}
	aml_vector_remove(&scratch->data.requests, req);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

void *aml_scratch_double_baseptr(const struct aml_scratch_data *d)
{
	assert(d != NULL);
	// don't think this function makes sense for this implementation.
	return NULL;
}

int aml_scratch_double_release(struct aml_scratch_data *d, int scratchid)
{
	assert(d != NULL);
	struct aml_scratch_double *scratch = (struct aml_scratch_double *)d;
	int *tile;

	pthread_mutex_lock(&scratch->data.lock);
	tile = aml_vector_get(&scratch->data.tilemap, scratchid);
	if(tile != NULL)
		aml_vector_remove(&scratch->data.tilemap, tile);
	pthread_mutex_unlock(&scratch->data.lock);
	return 0;
}

struct aml_scratch_ops aml_scratch_double_ops = {
	aml_scratch_double_create_request,
	aml_scratch_double_destroy_request,
	aml_scratch_double_wait_request,
	aml_scratch_double_baseptr,
	aml_scratch_double_release,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_scratch_double_create(struct aml_scratch **d, ...)
{
	va_list ap;
	struct aml_scratch *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, d);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_SCRATCH_DOUBLE_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_scratch);

	ret = (struct aml_scratch *)baseptr;
	ret->data = (struct aml_scratch_data *)dataptr;

	aml_scratch_double_vinit(ret, ap);

	va_end(ap);
	*d = ret;
	return 0;
}
int aml_scratch_double_vinit(struct aml_scratch *d, va_list ap)
{
	d->ops = &aml_scratch_double_ops;
	struct aml_scratch_double *scratch = (struct aml_scratch_double *)d->data;

	scratch->ops = aml_scratch_double_inner_ops;

	scratch->data.dest_tiling = va_arg(ap, struct aml_tiling_nd *);
	scratch->data.src_tiling = va_arg(ap, struct aml_tiling_nd *);
	scratch->data.push_dma = va_arg(ap, struct aml_dma *);
	scratch->data.pull_dma = va_arg(ap, struct aml_dma *);
	size_t nbtiles = va_arg(ap, size_t);
	size_t nbreqs = va_arg(ap, size_t);

	/* allocate request array */
	aml_vector_init(&scratch->data.requests, nbreqs,
			sizeof(struct aml_scratch_request_double),
			offsetof(struct aml_scratch_request_double, type),
			AML_SCRATCH_REQUEST_TYPE_INVALID);

	/* scratch init */
	aml_vector_init(&scratch->data.tilemap, nbtiles, sizeof(int), 0, -1);
	pthread_mutex_init(&scratch->data.lock, NULL);
	return 0;
}
int aml_scratch_double_init(struct aml_scratch *d, ...)
{
	int err;
	va_list ap;
	va_start(ap, d);
	err = aml_scratch_double_vinit(d, ap);
	va_end(ap);
	return err;
}

int aml_scratch_double_destroy(struct aml_scratch *d)
{
	struct aml_scratch_double *scratch = (struct aml_scratch_double *)d->data;
	aml_vector_destroy(&scratch->data.requests);
	aml_vector_destroy(&scratch->data.tilemap);
	pthread_mutex_destroy(&scratch->data.lock);
	return 0;
}
