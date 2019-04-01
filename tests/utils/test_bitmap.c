/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

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

void test_bitmap_string_conversion(const char * bitmap_str){
	struct aml_bitmap b, c;
	char * cstr;
	
	assert(aml_bitmap_from_string(&b, bitmap_str) == 0);

	if(bitmap_str == NULL              ||
	   !strcasecmp(bitmap_str, "none") ||
	   !strcasecmp(bitmap_str, "zero") ||
	   !strcasecmp(bitmap_str, "empty")){
		assert(aml_bitmap_iszero(&b));
	}
	else if(!strcasecmp(bitmap_str, "all")  ||
		!strcasecmp(bitmap_str, "full") ||
		!strcasecmp(bitmap_str, "fill")){
		assert(aml_bitmap_isfull(&b));
	}
		
	cstr = aml_bitmap_to_string(&b);
	assert(cstr != NULL);
	assert(aml_bitmap_from_string(&c, cstr) == 0);
	assert(aml_bitmap_isequal(&b,&c));	
	free(cstr);
}

void test_bitmap_string(){
	const size_t int_len = 16;
	char *bstr, *next;
	int i;
	const size_t max_len = int_len * (1+nis);
	size_t len = 0;
	struct aml_bitmap b;
		
	assert(aml_bitmap_from_string(&b, "unapropriate string") == -1);

	test_bitmap_string_conversion("all");
	test_bitmap_string_conversion("full");
	test_bitmap_string_conversion("fill");
	test_bitmap_string_conversion("zero");
	test_bitmap_string_conversion("empty");
	test_bitmap_string_conversion("none");
	test_bitmap_string_conversion(NULL);
	
	bstr = malloc(int_len);
	for(i = 0; i<nis; i++){
		memset(bstr, 0 , int_len);
		snprintf(bstr, int_len, "%lu", is[i]);
		test_bitmap_string_conversion(bstr);
		assert(aml_bitmap_from_string(&b, bstr) == 0);
		assert(aml_bitmap_isset(&b, is[i]));
	}
	free(bstr);
	
	bstr = malloc(max_len);
        next = bstr;	
	memset(bstr, 0 , max_len);
	for(i = 0; i<nis; i++){
		len += snprintf(bstr+len, max_len-len, "%lu", is[i]);
		if(i+1 < nis){
			bstr[len] = ',';
			len++;
		}
	}
	test_bitmap_string_conversion(bstr);
	assert(aml_bitmap_from_string(&b, bstr) == 0);
	for(i = 0; i<nis; i++)
		assert(aml_bitmap_isset(&b, is[i]));
	free(bstr);
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
	test_bitmap_string();
	return 0;
}

