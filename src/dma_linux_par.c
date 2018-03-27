#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <sys/mman.h>

/*******************************************************************************
 * Linux-backed, paruential dma
 * The dma itself is organized into several different components
 * - request types: copy or move
 * - implementation of the request
 * - user API (i.e. generic request creation and call)
 * - how to init the dma
 ******************************************************************************/

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_dma_request_linux_par_copy_init(struct aml_dma_request_linux_par *req,
					struct aml_tiling *dt,
					void *dptr, int dtid,
					struct aml_tiling *st,
					void *sptr, int stid)
{
	assert(req != NULL);

	req->type = AML_DMA_REQUEST_TYPE_COPY;
	/* figure out pointers */
	req->dest = aml_tiling_tilestart(dt, dptr, dtid);
	req->src = aml_tiling_tilestart(st, sptr, stid);
	req->size = aml_tiling_tilesize(st, stid);
	/* TODO: assert size match */
	return 0;
}

int aml_dma_request_linux_par_copy_destroy(struct aml_dma_request_linux_par *r)
{
	assert(r != NULL);
	return 0;
}

int aml_dma_request_linux_par_move_init(struct aml_dma_request_linux_par *req,
					struct aml_area *darea,
					struct aml_tiling *tiling,
					void *startptr, int tileid)
{
	assert(req != NULL);
	struct aml_binding *binding;

	req->type = AML_DMA_REQUEST_TYPE_MOVE;
	aml_area_binding(darea, &binding);
	req->count = aml_binding_nbpages(binding, tiling, startptr, tileid);
	req->pages = calloc(req->count, sizeof(void *));
	req->nodes = calloc(req->count, sizeof(int));
	aml_binding_pages(binding, req->pages, tiling, startptr, tileid);
	aml_binding_nodes(binding, req->nodes, tiling, startptr, tileid);
	free(binding);
	return 0;
}

int aml_dma_request_linux_par_move_destroy(struct aml_dma_request_linux_par *req)
{
	assert(req != NULL);
	free(req->pages);
	free(req->nodes);
	return 0;
}

/*******************************************************************************
 * Internal functions
 ******************************************************************************/

void *aml_dma_linux_par_do_thread(void *arg)
{
	struct aml_dma_linux_par_thread_data *data =
		(struct aml_dma_linux_par_thread_data *)arg;

	if(data->req->type == AML_DMA_REQUEST_TYPE_COPY)
		data->dma->ops.do_copy(&data->dma->data, data->req, data->tid);
	else if(data->req->type == AML_DMA_REQUEST_TYPE_MOVE)
		data->dma->ops.do_move(&data->dma->data, data->req, data->tid);
	return NULL;
}

int aml_dma_linux_par_do_copy(struct aml_dma_linux_par_data *dma,
			      struct aml_dma_request_linux_par *req, int tid)
{
	assert(dma != NULL);
	assert(req != NULL);

	/* chunk memory */
	size_t nbthreads = dma->nbthreads;
	size_t chunksize = req->size / nbthreads;

	void *dest = (void*)((intptr_t)req->dest + tid * chunksize);
	void *src = (void*)((intptr_t)req->src + tid * chunksize);

	if(tid == nbthreads - 1 && req->size > chunksize * nbthreads)
		chunksize += req->size % nbthreads;

	memcpy(dest, src, chunksize);
	return 0;
}

int aml_dma_linux_par_do_move(struct aml_dma_linux_par_data *dma,
			      struct aml_dma_request_linux_par *req, int tid)
{
	assert(dma != NULL);
	assert(req != NULL);

	size_t nbthreads = dma->nbthreads;
	size_t chunksize = req->count / nbthreads;

	size_t idx = tid * chunksize;

	if(tid == nbthreads - 1 && req->count > chunksize * nbthreads)
		chunksize += req->count % nbthreads;

	int status[chunksize];
	int err;
	err = move_pages(0, chunksize, &req->pages[idx], &req->nodes[idx],
			 status, MPOL_MF_MOVE);
	if(err)
	{
		perror("move_pages:");
		return errno;
	}
	return 0;
}

struct aml_dma_linux_par_ops aml_dma_linux_par_inner_ops = {
	aml_dma_linux_par_do_thread,
	aml_dma_linux_par_do_copy,
	aml_dma_linux_par_do_move,
};

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_dma_linux_par_create_request(struct aml_dma_data *d,
				     struct aml_dma_request **r,
				     int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma =
		(struct aml_dma_linux_par *)d;

	struct aml_dma_request_linux_par *req;

	req = aml_vector_add(&dma->data.requests);

	/* init the request */
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
		aml_dma_request_linux_par_copy_init(req, dt, dptr, dtid,
						    st, sptr, stid);
	}
	else if(type == AML_DMA_REQUEST_TYPE_MOVE)
	{
		struct aml_area *darea = va_arg(ap, struct aml_area *);
		struct aml_tiling *st = va_arg(ap, struct aml_tiling *);
		void *sptr = va_arg(ap, void *);
		int stid = va_arg(ap, int);
		aml_dma_request_linux_par_move_init(req, darea, st, sptr,
						    stid);
	}

	for(int i = 0; i < dma->data.nbthreads; i++)
	{
		struct aml_dma_linux_par_thread_data *rd = &req->thread_data[i];
		rd->req = req;
		rd->dma = dma;
		rd->tid = i;
		pthread_create(&rd->thread, NULL, dma->ops.do_thread, rd);
	}
	*r = (struct aml_dma_request *)req;
	return 0;
}

int aml_dma_linux_par_destroy_request(struct aml_dma_data *d,
				      struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma =
		(struct aml_dma_linux_par *)d;

	struct aml_dma_request_linux_par *req =
		(struct aml_dma_request_linux_par *)r;

	/* we cancel and join, instead of killing, for a cleaner result */
	for(int i = 0; i < dma->data.nbthreads; i++)
	{
		pthread_cancel(req->thread_data[i].thread);
		pthread_join(req->thread_data[i].thread, NULL);
	}

	if(req->type == AML_DMA_REQUEST_TYPE_COPY)
		aml_dma_request_linux_par_copy_destroy(req);
	else if(req->type == AML_DMA_REQUEST_TYPE_MOVE)
		aml_dma_request_linux_par_move_destroy(req);

	aml_vector_remove(&dma->data.requests, req);
	return 0;
}

int aml_dma_linux_par_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_linux_par *dma = (struct aml_dma_linux_par *)d;
	struct aml_dma_request_linux_par *req =
		(struct aml_dma_request_linux_par *)r;

	for(int i = 0; i < dma->data.nbthreads; i++)
		pthread_join(req->thread_data[i].thread, NULL);

	/* destroy a completed request */
	if(req->type == AML_DMA_REQUEST_TYPE_COPY)
		aml_dma_request_linux_par_copy_destroy(req);
	else if(req->type == AML_DMA_REQUEST_TYPE_MOVE)
		aml_dma_request_linux_par_move_destroy(req);

	aml_vector_remove(&dma->data.requests, req);
	return 0;
}

struct aml_dma_ops aml_dma_linux_par_ops = {
	aml_dma_linux_par_create_request,
	aml_dma_linux_par_destroy_request,
	aml_dma_linux_par_wait_request,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_dma_linux_par_create(struct aml_dma **d, ...)
{
	va_list ap;
	struct aml_dma *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, d);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_DMA_LINUX_PAR_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_dma);

	ret = (struct aml_dma *)baseptr;
	ret->data = (struct aml_dma_data *)dataptr;

	aml_dma_linux_par_vinit(ret, ap);

	va_end(ap);
	*d = ret;
	return 0;
}
int aml_dma_linux_par_vinit(struct aml_dma *d, va_list ap)
{
	d->ops = &aml_dma_linux_par_ops;
	struct aml_dma_linux_par *dma = (struct aml_dma_linux_par *)d->data;

	dma->ops = aml_dma_linux_par_inner_ops;
	/* allocate request array */
	size_t nbreqs = va_arg(ap, size_t);
	dma->data.nbthreads = va_arg(ap, size_t);

	aml_vector_init(&dma->data.requests, nbreqs,
			sizeof(struct aml_dma_request_linux_par),
			offsetof(struct aml_dma_request_linux_par, type),
			AML_DMA_REQUEST_TYPE_INVALID);
	for(int i = 0; i < nbreqs; i++)
	{
		struct aml_dma_request_linux_par *req =
			aml_vector_get(&dma->data.requests, i);
		req->thread_data = calloc(dma->data.nbthreads,
					  sizeof(struct aml_dma_linux_par_thread_data));
	}
	return 0;
}
int aml_dma_linux_par_init(struct aml_dma *d, ...)
{
	int err;
	va_list ap;
	va_start(ap, d);
	err = aml_dma_linux_par_vinit(d, ap);
	va_end(ap);
	return err;
}

int aml_dma_linux_par_destroy(struct aml_dma *d)
{
	struct aml_dma_linux_par *dma = (struct aml_dma_linux_par *)d->data;
	for(int i = 0; i < aml_vector_size(&dma->data.requests); i++)
	{
		struct aml_dma_request_linux_par *req =
			aml_vector_get(&dma->data.requests, i);
		free(req->thread_data);
	}
	aml_vector_destroy(&dma->data.requests);
	return 0;
}
