#include "config.h"
#include "aml.h"

#define AML_BITMAP_EMPTY       (0UL)
#define AML_BITMAP_FULL        (~0UL)
#define AML_BITMAP_NBITS     (8 * sizeof(AML_BITMAP_TYPE))
#define AML_BITMAP_NTH(i)      ((i) / AML_BITMAP_NBITS)
#define AML_BITMAP_ITH(i)      (((i) % AML_BITMAP_NBITS))

struct aml_bitmap *aml_bitmap_alloc(void)
{
	struct aml_bitmap *b = malloc(sizeof(struct aml_bitmap));
	if(b == NULL)
		return NULL;
	aml_bitmap_zero(b);
	return b;
}

void aml_bitmap_copy(struct aml_bitmap *dst, const struct aml_bitmap *src)
{
	if(dst == NULL || src == NULL)
		return;
	memcpy(dst, src, sizeof(struct aml_bitmap));
}

void aml_bitmap_copy_ulong(struct aml_bitmap *dst, unsigned long *src,
			   size_t maxbit)
{
	if(dst == NULL || src == NULL)
		return;
	if(maxbit > AML_BITMAP_MAX)
		maxbit = AML_BITMAP_MAX;
	for(size_t i = 0; i < maxbit; i++)
		if(src[AML_BITMAP_NTH(i)] & (1UL << AML_BITMAP_ITH(i)) != 0)
			aml_bitmap_set(dst, i);
}


struct aml_bitmap *aml_bitmap_dup(const struct aml_bitmap *a)
{
	struct aml_bitmap *b = aml_bitmap_alloc();
	if(b == NULL)
		return NULL;
	aml_bitmap_copy(b, a);
	return b;
}

void aml_bitmap_free(struct aml_bitmap *bitmap)
{
	free(bitmap);
}

void aml_bitmap_zero(struct aml_bitmap *bitmap)
{
	memset(bitmap, 0, sizeof(struct aml_bitmap));
}

int aml_bitmap_iszero(const struct aml_bitmap *bitmap){
	int i;
	for(i=0; i<AML_BITMAP_SIZE; i++)
		if(bitmap->mask[i] != AML_BITMAP_EMPTY)
			return 0;
	return 1;
}

int aml_bitmap_isfull(const struct aml_bitmap *bitmap)
{
	int i;
	for(i=0; i<AML_BITMAP_SIZE; i++)
		if(bitmap->mask[i] != AML_BITMAP_FULL)
			return 0;
	return 1;
}

void aml_bitmap_fill(struct aml_bitmap *bitmap){
	memset(bitmap, AML_BITMAP_FULL, sizeof(struct aml_bitmap));
}

int aml_bitmap_isset(const struct aml_bitmap *bitmap, const unsigned i)
{
	if(i >= AML_BITMAP_MAX)
		return -1;
	return (bitmap->mask[AML_BITMAP_NTH(i)] &
		(1UL << AML_BITMAP_ITH(i))) > 0UL;
}

int aml_bitmap_set(struct aml_bitmap *bitmap, const unsigned i)
{
	if(i >= AML_BITMAP_MAX)
		return -1;
	bitmap->mask[AML_BITMAP_NTH(i)] |= (1UL << AML_BITMAP_ITH(i));
	return 0;
}

int aml_bitmap_isequal(const struct aml_bitmap *a, const struct aml_bitmap *b)
{
	int i;
	for(i = 0; i < AML_BITMAP_SIZE; i++)
		if(a->mask[i] != b->mask[i])
			return 0;
	return 1;
}

int aml_bitmap_clear(struct aml_bitmap *bitmap, const unsigned i)
{
	if(i >= AML_BITMAP_MAX)
		return -1;
	bitmap->mask[AML_BITMAP_NTH(i)] &= ~(1UL << AML_BITMAP_ITH(i));
	return 0;
}

int aml_bitmap_set_range(struct aml_bitmap *bitmap,
			 const unsigned i, const unsigned ii)
{
	if(i >= AML_BITMAP_MAX || ii >= AML_BITMAP_MAX || i > ii)
		return -1;
	if(i == ii)
		return aml_bitmap_set(bitmap, i);

	unsigned long k    =  AML_BITMAP_ITH(ii+1);
	unsigned long low  =  (AML_BITMAP_FULL << AML_BITMAP_ITH(i));
	unsigned long n    =  AML_BITMAP_NTH(i);
	unsigned long nn   =  AML_BITMAP_NTH(ii);
	unsigned long high =  k == 0 ? AML_BITMAP_FULL : ~(AML_BITMAP_FULL << k);

	if(nn>n)
	{
		for(k=n+1; k<=nn-1; k++)
			bitmap->mask[k] = AML_BITMAP_FULL;
		bitmap->mask[n]  |= low;
		bitmap->mask[nn] |= high;
	}
	else
		bitmap->mask[n]  |= (low & high);

	return 0;
}

int aml_bitmap_clear_range(struct aml_bitmap *bitmap,
			   const unsigned i, const unsigned ii)
{
	if(i >= AML_BITMAP_MAX || ii >= AML_BITMAP_MAX || i > ii)
		return -1;
	if(i == ii)
		return aml_bitmap_set(bitmap, i);

	unsigned long k    =  AML_BITMAP_ITH(ii+1);
	unsigned long low  =  ~(AML_BITMAP_FULL << AML_BITMAP_ITH(i));
	unsigned long n    =  AML_BITMAP_NTH(i);
	unsigned long nn   =  AML_BITMAP_NTH(ii);
	unsigned long high =  k == 0 ? AML_BITMAP_EMPTY : (AML_BITMAP_FULL << k);

	if(nn>n)
	{
		for(k=n+1; k<=nn-1; k++)
			bitmap->mask[k] = AML_BITMAP_EMPTY;
		bitmap->mask[n]  &= low;
		bitmap->mask[nn] &= high;
	}
	else
		bitmap->mask[n]  &= (low | high);

	return 0;
}

unsigned long aml_bitmap_nset(const struct aml_bitmap *bitmap)
{
	unsigned long i, b, n;
	unsigned long test = 1UL;
	unsigned long nset = 0;
		;
	for(n = 0; n < AML_BITMAP_SIZE; n++){
		b = bitmap->mask[n];
		for(i = 0; i < AML_BITMAP_NBITS; i++){
			nset += b & test ? 1 : 0;
			b = b >> 1;
		}
	}
	return nset;
}

#ifdef HAVE_HWLOC

#include <hwloc.h>

int hwloc_bitmap_copy_aml_bitmap(hwloc_bitmap_t hb, const struct aml_bitmap *ab)
{
	if(hb == NULL)
		return -1;
	hwloc_bitmap_zero(hb);
	
	if(ab == NULL)
		return 0;

	int i, last = 0;
	for(i = 0; i<AML_BITMAP_LEN; i++)
		if(aml_bitmap_isset(ab, i)){
			hwloc_bitmap_set(hb, i);
			last = i;
		}
	return last;
}

hwloc_bitmap_t hwloc_bitmap_from_aml_bitmap(const struct aml_bitmap *b)
{
	hwloc_bitmap_t hb = hwloc_bitmap_alloc();
	hwloc_bitmap_copy_aml_bitmap(hb, b);
	return hb;
}

int aml_bitmap_copy_hwloc_bitmap(struct aml_bitmap *ab, const hwloc_bitmap_t hb)
{
	if(ab == NULL)
		return -1;

	aml_bitmap_zero(ab);
	if(hb == NULL || hwloc_bitmap_iszero(hb))
		return 0;

	int i = -1;
	while((i = hwloc_bitmap_next(hb, i)) != -1){
		if(i >= AML_BITMAP_LEN)
			break;
		aml_bitmap_set(ab, i);
	}
	return i >= AML_BITMAP_LEN ? i : 0;
}

struct aml_bitmap * aml_bitmap_from_hwloc_bitmap(const hwloc_bitmap_t b)
{
        aml_bitmap ab = aml_bitmap_alloc();	
	aml_bitmap_copy_hwloc_bitmap(ab, b);
	return ab;
}
#else
#pragma message("do not HAVE HWLOC")

#endif //HAVE_HWLOC
