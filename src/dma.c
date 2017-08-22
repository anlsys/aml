#include <aml.h>
#include <assert.h>

/*******************************************************************************
 * DMA implementation
 * At this point, implement the single threaded synchronous stuff
 ******************************************************************************/

int aml_dma_init(struct aml_dma *dma, unsigned int type)
{
	return 0;
}

int aml_dma_destroy(struct aml_dma *dma)
{
	return 0;
}

int aml_dma_copy(struct aml_dma *dma, void *dest, const void *src, size_t size)
{
	return 0;
}

int aml_dma_move(struct aml_dma *dma, struct aml_area *dest,
		 struct aml_area *src, void *ptr, size_t size)
{
	return 0;
}
