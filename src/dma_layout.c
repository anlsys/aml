#include <aml.h>
#include <assert.h>
#include <errno.h>
#include <sys/mman.h>

/*******************************************************************************
 * Requests:
 ******************************************************************************/

int aml_dma_request_layout_init(struct aml_dma_request_layout *req,
				struct aml_layout *dl,
				struct aml_layout *sl, void *arg)
{
	assert(req != NULL);
	req->type = AML_DMA_REQUEST_TYPE_COPY;
	/* figure out pointers */
	req->dest = dl;
	req->src = sl;
	req->arg = arg;
	return 0;
}

int aml_dma_request_layout_destroy(struct aml_dma_request_layout *r)
{
	assert(r != NULL);
	return 0;
}

/*******************************************************************************
 * Public API
 ******************************************************************************/

int aml_dma_layout_create_request(struct aml_dma_data *d,
				  struct aml_dma_request **r,
				  int type, va_list ap)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_layout *dma =
		(struct aml_dma_layout *)d;

	struct aml_dma_request_layout *req;

	pthread_mutex_lock(&dma->lock);
	req = aml_vector_add(&dma->requests);

	/* we don't support move at this time */
	assert(type == AML_DMA_REQUEST_TYPE_COPY);
	struct aml_layout *dl, *sl;
	void *arg;
	dl = va_arg(ap, struct aml_layout *);
	sl = va_arg(ap, struct aml_layout *);
	arg = va_arg(ap, void *);
	aml_dma_request_layout_init(req, dl, sl, arg);

	pthread_mutex_unlock(&dma->lock);
	*r = (struct aml_dma_request *)req;
	return 0;
}

int aml_dma_layout_destroy_request(struct aml_dma_data *d,
				   struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_layout *dma =
		(struct aml_dma_layout *)d;

	struct aml_dma_request_layout *req =
		(struct aml_dma_request_layout *)r;

	assert(req->type == AML_DMA_REQUEST_TYPE_COPY);
	aml_dma_request_layout_destroy(req);

	/* enough to remove from request vector */
	pthread_mutex_lock(&dma->lock);
	aml_vector_remove(&dma->requests, req);
	pthread_mutex_unlock(&dma->lock);
	return 0;
}

int aml_dma_layout_wait_request(struct aml_dma_data *d,
				   struct aml_dma_request *r)
{
	assert(d != NULL);
	assert(r != NULL);
	struct aml_dma_layout *dma = (struct aml_dma_layout *)d;
	struct aml_dma_request_layout *req =
		(struct aml_dma_request_layout *)r;

	/* execute */
	assert(req->type == AML_DMA_REQUEST_TYPE_COPY);
	dma->do_work(req->dest, req->src, req->arg);

	/* destroy a completed request */
	aml_dma_layout_destroy_request(d, r);
	return 0;
}

struct aml_dma_ops aml_dma_ops_layout = {
	aml_dma_layout_create_request,
	aml_dma_layout_destroy_request,
	aml_dma_layout_wait_request,
};

/*******************************************************************************
 * Init functions:
 ******************************************************************************/

int aml_dma_layout_create(struct aml_dma **d, ...)
{
	va_list ap;
	struct aml_dma *ret = NULL;
	intptr_t baseptr, dataptr;
	va_start(ap, d);

	/* alloc */
	baseptr = (intptr_t) calloc(1, AML_DMA_LAYOUT_ALLOCSIZE);
	dataptr = baseptr + sizeof(struct aml_dma);

	ret = (struct aml_dma *)baseptr;
	ret->data = (struct aml_dma_data *)dataptr;

	aml_dma_layout_vinit(ret, ap);

	va_end(ap);
	*d = ret;
	return 0;
}
int aml_dma_layout_vinit(struct aml_dma *d, va_list ap)
{
	d->ops = &aml_dma_ops_layout;
	struct aml_dma_layout *dma = (struct aml_dma_layout *)d->data;

	/* request vector */
	size_t nbreqs = va_arg(ap, size_t);
	dma->do_work = va_arg(ap, aml_dma_operator);
	aml_vector_init(&dma->requests, nbreqs,
			sizeof(struct aml_dma_request_layout),
			offsetof(struct aml_dma_request_layout, type),
			AML_DMA_REQUEST_TYPE_INVALID);
	pthread_mutex_init(&dma->lock, NULL);
	return 0;
}
int aml_dma_layout_init(struct aml_dma *d, ...)
{
	int err;
	va_list ap;
	va_start(ap, d);
	err = aml_dma_layout_vinit(d, ap);
	va_end(ap);
	return err;
}

int aml_dma_layout_destroy(struct aml_dma *d)
{
	struct aml_dma_layout *dma = (struct aml_dma_layout *)d->data;
	aml_vector_destroy(&dma->requests);
	pthread_mutex_destroy(&dma->lock);
	return 0;
}
