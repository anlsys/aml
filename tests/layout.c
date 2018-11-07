#include <aml.h>
#include <assert.h>

int main(int argc, char *argv[])
{
	struct aml_layout *a;
	AML_LAYOUT_DECL(b, 5);

	/* padd the dims to the closest multiple of 2 */
	float memory[4][4][8][12][16];
	size_t dims[5] = {2,3,7,11,13};
	size_t pitch[5] = {4, 4*4, 4*4*4, 4*4*4*8, 4*4*4*8*12};
	size_t stride[5] = {1,1,1,1,1};

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize the layouts */
	aml_layout_create(&a, (void *)memory, 5, dims, pitch, stride);
	aml_layout_init(&b, (void *)memory, 5, dims, pitch, stride);

	/* some simple checks */
	assert(!memcmp(a->dims, dims, sizeof(size_t)*5));
	assert(!memcmp(a->pitch, pitch, sizeof(size_t)*5));
	assert(!memcmp(a->stride, stride, sizeof(size_t)*5));
	assert(!memcmp(b.dims, dims, sizeof(size_t)*5));
	assert(!memcmp(b.pitch, pitch, sizeof(size_t)*5));
	assert(!memcmp(b.stride, stride, sizeof(size_t)*5));

	free(a);

	aml_finalize();
	return 0;
}
