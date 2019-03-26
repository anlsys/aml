#include "aml.h"
#include <assert.h>

static const unsigned long is[8] = {
	0,
	AML_BITMAP_NBITS / 4,
	3 * AML_BITMAP_NBITS / 4,
	AML_BITMAP_NBITS - 1, 
	AML_BITMAP_NBITS,
	AML_BITMAP_NBITS + AML_BITMAP_NBITS / 4,
	AML_BITMAP_NBITS + 3 * AML_BITMAP_NBITS / 4,
	AML_BITMAP_NBITS + AML_BITMAP_NBITS - 1
};

static const int nis = sizeof(is) / sizeof(*is);

void test_bitmap_fill(){
	unsigned long i;
	struct aml_bitmap b;
	aml_bitmap_fill(&b);
	for(i = 0; i < nis; i++)
		assert(aml_bitmap_isset(&b, is[i]));
	assert(aml_bitmap_nset(&b) == AML_BITMAP_MAX);
}

void test_bitmap_zero(){
	unsigned long i;
	struct aml_bitmap b;
	aml_bitmap_zero(&b);
	for(i = 0; i < nis; i++)
		assert(!aml_bitmap_isset(&b, is[i]));
	assert(aml_bitmap_nset(&b) == 0);
}

void test_bitmap_set(){
	unsigned long i,j;
	struct aml_bitmap b;

	aml_bitmap_zero(&b);
	for(i = 0; i < nis; i++){
		aml_bitmap_set(&b, is[i]);
		assert(aml_bitmap_isset(&b, is[i]));

		for(j = 0; j < is[i]; j++)
			assert(!aml_bitmap_isset(&b, j));
		for(j = is[i]+1; j < AML_BITMAP_MAX; j++)
			assert(!aml_bitmap_isset(&b, j));
		
		assert(aml_bitmap_nset(&b) == 1);
		aml_bitmap_clear(&b, is[i]);
		assert(!aml_bitmap_isset(&b, is[i]));
	}
}

void test_bitmap_clear(){
	unsigned long i,j;
	struct aml_bitmap b;
	
	aml_bitmap_fill(&b);
	for(i = 0; i < nis; i++){
		aml_bitmap_clear(&b, is[i]);
		assert(!aml_bitmap_isset(&b, is[i]));

		for(j = 0; j < is[i]; j++)
			assert(aml_bitmap_isset(&b, j));
		for(j = is[i]+1; j < AML_BITMAP_MAX; j++)
			assert(aml_bitmap_isset(&b, j));

		assert(aml_bitmap_nset(&b) == (AML_BITMAP_MAX-1));
		aml_bitmap_set(&b, is[i]);
		assert(aml_bitmap_isset(&b, is[i]));
	}
}

void test_bitmap_set_range(){
	unsigned long i, ii, j;
	struct aml_bitmap b;
	aml_bitmap_zero(&b);

	for(i = 0; i < nis; i++){
		for(ii = i; ii < nis; ii++){
			assert(aml_bitmap_set_range(&b, is[i], is[ii]) == 0);
			assert(aml_bitmap_nset(&b) == (1 + is[ii] - is[i]));
			for(j = 0; j < is[i]; j++)
				assert(!aml_bitmap_isset(&b, j));
			for(j = is[i]; j <= is[ii]; j++)
				assert(aml_bitmap_isset(&b, j));
			for(j = is[ii]+1; j < AML_BITMAP_MAX; j++)
				assert(!aml_bitmap_isset(&b, j));
			aml_bitmap_zero(&b);
		}
	}
}

void test_bitmap_clear_range(){
	unsigned long i, ii, j;
	struct aml_bitmap b;
	aml_bitmap_fill(&b);

	for(i = 0; i < nis; i++){
		for(ii = i; ii < nis; ii++){
			assert(aml_bitmap_clear_range(&b, is[i], is[ii]) == 0);
			assert(aml_bitmap_nset(&b) == (AML_BITMAP_MAX - is[ii] + is[i] - 1));
			for(j = 0; j < is[i]; j++)
				assert(aml_bitmap_isset(&b, j));
			for(j = is[i]; j <= is[ii]; j++)
				assert(!aml_bitmap_isset(&b, j));
			for(j = is[ii]+1; j < AML_BITMAP_MAX; j++)
				assert(aml_bitmap_isset(&b, j));
			aml_bitmap_fill(&b);
		}
	}
}

int main(){
	test_bitmap_fill();
	test_bitmap_zero();
	test_bitmap_set();
	test_bitmap_clear();
	test_bitmap_set_range();
	test_bitmap_clear_range();
	return 0;
}

