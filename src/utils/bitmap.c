#include "aml.h"

#define AML_BITMAP_EMPTY       (0UL)
#define AML_BITMAP_FULL        (~0UL)
#define AML_BITMAP_TYPELEN     (8 * sizeof(aml_bitmap))
#define AML_BITMAP_NTH(i)      ((i) / AML_BITMAP_TYPELEN)
#define AML_BITMAP_ITH(i)      (((i) % AML_BITMAP_TYPELEN))

aml_bitmap aml_bitmap_alloc()
{
	aml_bitmap b = malloc(sizeof(*b) * AML_BITMAP_NUM);
	if(b == NULL)
		return NULL;
	aml_bitmap_zero(b);
	return b;
}

void aml_bitmap_copy(aml_bitmap dst, const aml_bitmap src)
{
	int i;
	if(dst == NULL || src == NULL)
		return;
	for(i=0; i<AML_BITMAP_NUM; i++)
		dst[i] = src[i];
}

aml_bitmap aml_bitmap_dup(const aml_bitmap a)
{
	aml_bitmap b = malloc(sizeof(*b) * AML_BITMAP_NUM);
	int i;
	if(b == NULL)
		return NULL;
	for(i=0; i<AML_BITMAP_NUM; i++)
		b[i] = a[i];
	return b;
}

void aml_bitmap_free(aml_bitmap bitmap)
{
	free(bitmap);
}

void aml_bitmap_zero(aml_bitmap bitmap)
{
	int i;
	for(i=0; i<AML_BITMAP_NUM; i++)
		bitmap[i] = AML_BITMAP_EMPTY;
}

int aml_bitmap_iszero(const aml_bitmap bitmap){
	int i;
	for(i=0; i<AML_BITMAP_NUM; i++)
		if(bitmap[i] != AML_BITMAP_EMPTY)
			return 0;
	return 1;
}

int aml_bitmap_isfull(const aml_bitmap bitmap)
{
	int i;
	for(i=0; i<AML_BITMAP_NUM; i++)
		if(bitmap[i] != AML_BITMAP_FULL)
			return 0;
	return 1;
}

void aml_bitmap_fill(aml_bitmap bitmap){
	int i;
	for(i=0; i<AML_BITMAP_NUM; i++)
		bitmap[i] = AML_BITMAP_FULL;
}

int aml_bitmap_isset(const aml_bitmap bitmap, const unsigned i)
{
	if(i >= AML_BITMAP_LEN)
		return -1;
	unsigned n = AML_BITMAP_NTH(i);
	unsigned b = AML_BITMAP_ITH(i);
	return (bitmap[AML_BITMAP_NTH(i)] & (1UL << AML_BITMAP_ITH(i))) > 0UL;
}

int aml_bitmap_set(aml_bitmap bitmap, const unsigned i)
{
	if(i >= AML_BITMAP_LEN)
		return -1;
	bitmap[AML_BITMAP_NTH(i)] |= (1UL << AML_BITMAP_ITH(i));
	return 0;
}

int aml_bitmap_isequal(const aml_bitmap a, const aml_bitmap b)
{
	int i;
	for(i=0; i<AML_BITMAP_NUM; i++)
		if(a[i] != b[i])
			return 0;
	return 1;
}

int aml_bitmap_clear(aml_bitmap bitmap, const unsigned i)
{
	if(i >= AML_BITMAP_LEN)
		return -1;
	bitmap[AML_BITMAP_NTH(i)] &= ~(1UL << AML_BITMAP_ITH(i));
	return 0;
}

int aml_bitmap_set_range(aml_bitmap bitmap, const unsigned i, const unsigned ii)
{
	if(i >= AML_BITMAP_LEN || ii >= AML_BITMAP_LEN || i > ii)
		return -1;
	if(i == ii)
		return aml_bitmap_set(bitmap, i);

	unsigned long k    =  AML_BITMAP_ITH(ii+1);
	unsigned long low  =  (AML_BITMAP_FULL << AML_BITMAP_ITH(i));
	unsigned long n    =  AML_BITMAP_NTH(i);
	unsigned long nn   =  AML_BITMAP_NTH(ii);
	unsigned long high =  k == 0 ? AML_BITMAP_FULL : ~(AML_BITMAP_FULL << k);
	
	if(nn>n){
		for(k=n+1; k<=nn-1; k++)
			bitmap[k] = AML_BITMAP_FULL;
		bitmap[n]  |= low;
		bitmap[nn] |= high;
	} else 
		bitmap[n]  |= (low & high);
	
	return 0;
}

int aml_bitmap_clear_range(aml_bitmap bitmap, const unsigned i, const unsigned ii)
{
	if(i >= AML_BITMAP_LEN || ii >= AML_BITMAP_LEN || i > ii)
		return -1;
	if(i == ii)
		return aml_bitmap_set(bitmap, i);

	unsigned long k    =  AML_BITMAP_ITH(ii+1);
	unsigned long low  =  ~(AML_BITMAP_FULL << AML_BITMAP_ITH(i));
	unsigned long n    =  AML_BITMAP_NTH(i);
	unsigned long nn   =  AML_BITMAP_NTH(ii);
	unsigned long high =  k == 0 ? AML_BITMAP_EMPTY : (AML_BITMAP_FULL << k);
	
	if(nn>n){
		for(k=n+1; k<=nn-1; k++)
			bitmap[k] = AML_BITMAP_EMPTY;
		bitmap[n]  &= low;
		bitmap[nn] &= high;
	} else 
		bitmap[n]  &= (low | high);
	
	return 0;
}

unsigned long aml_bitmap_nset(const aml_bitmap bitmap)
{
	unsigned long i, b, n;
	unsigned long test = 1UL;
	unsigned long nset = 0;
		;
	for(n = 0; n < AML_BITMAP_NUM; n++){
		b = bitmap[n];
		for(i = 0; i < AML_BITMAP_TYPELEN; i++){
			nset += b & test ? 1 : 0;
			b = b >> 1;
		}
	}
	return nset;
}

