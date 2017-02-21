#include <stdlib.h>
#include <string.h>
#include "allocator.h"


/* an area of free memory inside a node */
struct aml_allocator_area {
	size_t size;
	struct aml_allocator_area *next;
};

/* header management and pointer conversion macros */

#define AREA_HEADER_SIZE (sizeof(size_t))
#define AREA_2_USER(p) (void *)((char *)p + AREA_HEADER_SIZE)
#define USER_2_AREA(p) (struct aml_allocator_area *)((char *)p - AREA_HEADER_SIZE)

/* alignments and allocator overhead.
 * We keep a dummy head at the start of the area, so we need at least one full
 * struct + the overhead of a single allocation.
 */
#define AREA_ALIGN_MASK (sizeof(struct aml_allocator_area)-((size_t)1))
#define AREA_OVERHEAD ((sizeof(struct aml_allocator_area) + AREA_HEADER_SIZE + AREA_ALIGN_MASK) & ~AREA_ALIGN_MASK)


static size_t usz2asz(size_t size)
{
	if(size < AREA_HEADER_SIZE)
		return sizeof(struct aml_allocator_area);
	else
	{
		size += AREA_HEADER_SIZE;
		return (size + AREA_ALIGN_MASK) & ~AREA_ALIGN_MASK;
	}
}

static void aml_allocator_find(struct aml_allocator_area *head, size_t size,
			       struct aml_allocator_area **ret,
			       struct aml_allocator_area **prev)
{
	struct aml_allocator_area *it, *old;
	it = head->next;
	old = head;
	for(it = head->next, old = head; it != NULL; old = it, it = it->next)
	{
		if(it->size >= size)
		{
			*ret = it;
			*prev = old;
			return;
		}
	}
	*ret = NULL;
	*prev = NULL;
}

static struct aml_allocator_area *
	aml_allocator_findprev(struct aml_allocator_area *head,
			       struct aml_allocator_area *area)
{
	struct aml_allocator_area *it, *prev;
	for(it = head->next, prev = head; it != NULL; prev = it, it = it->next)
		if(it > area)
			break;
	return prev;
}

int aml_allocator_init(void *start, size_t memsize)
{
	struct aml_allocator_area *head, *next;
	head = (struct aml_allocator_area *)start;
	next = head+1;
	next->size = memsize - sizeof(*head);
	next->next = NULL;
	head->size = memsize - sizeof(*head);
	head->next = next;
	return 0;
}

void *aml_allocator_alloc(void *start, size_t size)
{
	struct aml_allocator_area *head, *it, *prev, *next;
	void *ret;
	if(size == 0)
		return NULL;
	size = usz2asz(size);
	head = (struct aml_allocator_area*)start;
	if(size > head->size)
		return NULL;
	aml_allocator_find(head, size, &it, &prev);
	if(it == NULL)
		return NULL;

	if(it->size - size < sizeof(struct aml_allocator_area))
	{
		prev->next = it->next;
		ret = AREA_2_USER(it);
	}
	else
	{
		next = (struct aml_allocator_area*)((char *)it + size);
		next->size = it->size - size;
		prev->next = next;
		it->size = size;
		ret = AREA_2_USER(it);
	}
	head->size -= size;
	return ret;
}

void aml_allocator_free(void *start, void *p)
{
	struct aml_allocator_area *area, *prev, *next, *head;
	if(p == NULL)
		return;

	area = USER_2_AREA(p);
	area->next = NULL;
	head = (struct aml_allocator_area*) start;

	prev = aml_allocator_findprev(head, area);
	if(prev != head &&
	   (struct aml_allocator_area*)(prev + prev->size) == area)
	{
		prev->size += area->size;
		area = prev;
	}
	else
	{
		area->next = prev->next;
		prev->next = area;
	}
	next = area->next;
	if(next != NULL &&
	   (struct aml_allocator_area*)((char *)area + area->size) == next)
	{
		area->size += next->size;
		area->next = next->next;
	}
}

void *aml_allocator_realloc(void *start, void *p, size_t size)
{
	void *ret;
	struct aml_allocator_area *area;
	if(p == NULL)
		return aml_allocator_alloc(start, size);
	if(size == 0)
	{
		aml_allocator_free(start, p);
		return NULL;
	}

	ret = aml_allocator_alloc(start, size);
	if(ret != NULL)
	{
		ret = memcpy(ret, p, size);
		aml_allocator_free(start, p);
	}
	return ret;
}
