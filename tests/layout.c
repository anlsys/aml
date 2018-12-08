#include <aml.h>
#include <assert.h>

int main(int argc, char *argv[])
{
	struct aml_layout *a;
	AML_LAYOUT_DECL(b, 5);

	/* padd the dims to the closest multiple of 2 */
	float memory[4][4][8][12][16];
	size_t dims[5] = {2,3,7,11,13};
	size_t cpitch[5] = {4, 4*4, 4*4*4, 4*4*4*8, 4*4*4*8*12};
	size_t pitch[4] = {4, 4, 8, 12};
	size_t stride[5] = {1,1,1,1,1};

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize the layouts */
	aml_layout_acreate(&a, AML_TYPE_LAYOUT_ROW_ORDER, (void *)memory,
			   sizeof(float), 5, dims, stride, pitch);
	aml_layout_ainit(&b, AML_TYPE_LAYOUT_ROW_ORDER, (void *)memory,
			 sizeof(float), 5, dims, stride, pitch);

	assert( (intptr_t)(a->data->stride) - (intptr_t)(a->data->dims)
                == 5*sizeof(size_t) );
	assert( (intptr_t)(a->data->pitch) - (intptr_t)(a->data->dims)
                == 10*sizeof(size_t) );

	/* some simple checks */
	assert(!memcmp(a->data->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(a->data->pitch, cpitch, sizeof(size_t)*5));
	assert(!memcmp(a->data->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(b.data->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(b.data->pitch, cpitch, sizeof(size_t)*5));
	assert(!memcmp(b.data->stride, stride, sizeof(size_t)*5));

	free(a);

	aml_finalize();
	return 0;
}
