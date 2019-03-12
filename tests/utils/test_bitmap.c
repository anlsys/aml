#include "aml.h"
#include <assert.h>

void test_bitmap_fill(){
	unsigned long i;
	aml_bitmap_decl(b);
	aml_bitmap_fill(b);
	for(i = 0; i < AML_BITMAP_LEN; i++)
		assert(aml_bitmap_isset(b, i));
	assert(aml_bitmap_nset(b) == AML_BITMAP_LEN);
}

void test_bitmap_zero(){
	unsigned long i;
	aml_bitmap_decl(b);
	aml_bitmap_zero(b);
	for(i = 0; i < AML_BITMAP_LEN; i++)
		assert(!aml_bitmap_isset(b, i));
	assert(aml_bitmap_nset(b) == 0);
}

void test_bitmap_set(){
	unsigned long i,j;
	aml_bitmap_decl(b);

	aml_bitmap_zero(b);
	for(i = 0; i < AML_BITMAP_LEN; i++){
		aml_bitmap_set(b, i);
	        assert(aml_bitmap_isset(b, i));
		
		for(j = 0; j < i; j++)
			assert(!aml_bitmap_isset(b, j));
		for(j = i+1; j < AML_BITMAP_LEN; j++)
			assert(!aml_bitmap_isset(b, j));

		assert(aml_bitmap_nset(b) == 1);
		aml_bitmap_clear(b, i);
		assert(!aml_bitmap_isset(b, i));
	}
}

void test_bitmap_clear(){
	unsigned long i,j;
	aml_bitmap_decl(b);
	aml_bitmap_fill(b);
	for(i = 0; i < AML_BITMAP_LEN; i++){
		aml_bitmap_clear(b, i);
	        assert(!aml_bitmap_isset(b, i));
		
		for(j = 0; j < i; j++)
			assert(aml_bitmap_isset(b, j));
		for(j = i+1; j < AML_BITMAP_LEN; j++)
			assert(aml_bitmap_isset(b, j));

		assert(aml_bitmap_nset(b) == (AML_BITMAP_LEN-1));
		aml_bitmap_set(b, i);
		assert(aml_bitmap_isset(b, i));
	}
}

void test_bitmap_set_range(){
	unsigned long i, ii, j;
	aml_bitmap_decl(b);	
	aml_bitmap_zero(b);
	for(i = 0; i < AML_BITMAP_LEN; i++){
		for(ii = i+1; ii < AML_BITMAP_LEN; ii++){
			assert(aml_bitmap_set_range(b, i, ii) == 0);
			assert(aml_bitmap_nset(b) == (1 + ii - i));
			for(j = 0; j < i; j++)
				assert(!aml_bitmap_isset(b, j));
			for(j = i; j <= ii; j++)
				assert(aml_bitmap_isset(b, j));
			for(j = ii+1; j < AML_BITMAP_LEN; j++)
				assert(!aml_bitmap_isset(b, j));
			aml_bitmap_zero(b);
		}
	}
}

void test_bitmap_clear_range(){
	unsigned long i, ii, j;
	aml_bitmap_decl(b);
	aml_bitmap_fill(b);
	for(i = 0; i < AML_BITMAP_LEN; i++){
		for(ii = i+1; ii < AML_BITMAP_LEN; ii++){
			assert(aml_bitmap_clear_range(b, i, ii) == 0);
			assert(aml_bitmap_nset(b) == (AML_BITMAP_LEN-ii+i-1));
			for(j = 0; j < i; j++)
				assert(aml_bitmap_isset(b, j));
			for(j = i; j <= ii; j++)
				assert(!aml_bitmap_isset(b, j));
			for(j = ii+1; j < AML_BITMAP_LEN; j++)
				assert(aml_bitmap_isset(b, j));
			aml_bitmap_fill(b);
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

