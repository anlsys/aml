#ifndef AML_ALLOCATOR_H
#define AML_ALLOCATOR_H 1

int aml_allocator_init(void *start, size_t memsize);

void *aml_allocator_alloc(void *start, size_t size);

void aml_allocator_free(void *start, void *p);

void *aml_allocator_realloc(void *start, void *p, size_t size);

#endif
