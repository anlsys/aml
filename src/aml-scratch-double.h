#ifndef AML_SCRATCH_DOUBLE_H
#define AML_SCRATCH_DOUBLE_H 1

/*******************************************************************************
 * Sequential scratchpad API:
 * Scratchpad uses calling thread to trigger asynchronous dma movements.
 ******************************************************************************/

extern struct aml_scratch_ops aml_scratch_double_ops;

struct aml_scratch_request_double {
	int type;
	struct aml_dma *dma;
	struct aml_layout *src;
	int srcid;
	struct aml_layout *dest;
	int dstid;
	pthread_t thread;
};

struct aml_scratch_double_data {
	struct aml_tiling_nd *src_tiling;
	struct aml_tiling_nd *dest_tiling;
	struct aml_dma *push_dma;
	struct aml_dma *pull_dma;
	struct aml_vector tilemap;
	struct aml_vector requests;
	pthread_mutex_t lock;
};

struct aml_scratch_double_ops {
	void *(*do_thread)(void *);
};

struct aml_scratch_double {
	struct aml_scratch_double_ops ops;
	struct aml_scratch_double_data data;
};

#define AML_SCRATCH_DOUBLE_DECL(name) \
	struct aml_scratch_double __ ##name## _inner_data; \
	struct aml_scratch name = { \
		&aml_scratch_double_ops, \
		(struct aml_scratch_data *)&__ ## name ## _inner_data, \
	};

#define AML_SCRATCH_DOUBLE_ALLOCSIZE \
	(sizeof(struct aml_scratch_double) + \
	 sizeof(struct aml_scratch))

int aml_scratch_double_create(struct aml_scratch **scratch, ...);
int aml_scratch_double_init(struct aml_scratch *scratch, ...);
int aml_scratch_double_vinit(struct aml_scratch *scratch, va_list args);
int aml_scratch_double_destroy(struct aml_scratch *scratch);

#endif
