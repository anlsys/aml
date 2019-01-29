#ifndef AML_DMA_LAYOUT_H
#define AML_DMA_LAYOUT_H 1

/*******************************************************************************
 * Layout aware DMA
 * DMA using layouts as source and destination.
 ******************************************************************************/

extern struct aml_dma_ops aml_dma_ops_layout;

struct aml_dma_request_layout {
	int type;
	struct aml_layout *dest;
	struct aml_layout *src;
	void *arg;
};

typedef int (*aml_dma_operator)(struct aml_layout *, struct aml_layout *, void*);
struct aml_dma_layout {
	struct aml_vector requests;
	pthread_mutex_t lock;
	aml_dma_operator do_work;
};

#define AML_DMA_LAYOUT_DECL(name) \
	struct aml_dma_layout __ ##name## _inner_data; \
	struct aml_dma name = { \
		&aml_dma_ops_layout, \
		(struct aml_dma_data *)&__ ## name ## _inner_data, \
	};

#define AML_DMA_LAYOUT_ALLOCSIZE \
	(sizeof(struct aml_dma_layout) + \
	 sizeof(struct aml_dma))

int aml_dma_layout_create(struct aml_dma **dma, ...);
int aml_dma_layout_init(struct aml_dma *dma, ...);
int aml_dma_layout_vinit(struct aml_dma *dma, va_list args);
int aml_dma_layout_destroy(struct aml_dma *dma);

#endif
