/* copyright 2019 uchicago argonne, llc.
 * (c.f. authors, license)
 *
 * this file is part of the aml project.
 * for more info, see https://github.com/anlsys/aml
 *
 * spdx-license-identifier: bsd-3-clause
 ******************************************************************************/

#include "aml.h"

#include "aml/higher/mapper.h"
#include "aml/higher/mapper/creator.h"
#include "aml/higher/mapper/visitor.h"

#include "internal/utstack.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))

static int aml_mapper_creator_memcpy(struct aml_mapper_creator *c)
{
	struct aml_mapper_visitor_state *state = STACK_TOP(c->stack);
	const size_t size = state->mapper->size * state->array_size;
	void *src = state->device_ptr;
	void *dst = state->host_copy;

	// We always need to wait on this copy to complete in order to explore
	// children or to finish the mapping.
	if (c->dma_src_host != NULL) {
		int err = aml_dma_copy_custom(c->dma_src_host,
		                              (struct aml_layout *)dst,
		                              (struct aml_layout *)src,
		                              c->memcpy_src_host, (void *)size);
		if (err != AML_SUCCESS)
			return err;
	} else
		memcpy(dst, src, size);

	c->offset += size;
	return AML_SUCCESS;
}

static inline void **
aml_mapper_creator_field_ptr(struct aml_mapper_visitor_state *field,
                             struct aml_mapper_visitor_state *parent)
{
	return (void **)PTR_OFF(parent->host_copy, +,
	                        parent->mapper->offsets[field->field_num]);
}

static inline void aml_mapper_creator_init_field(struct aml_mapper_creator *c,
                                                 const size_t num)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(c->stack);
	struct aml_mapper_visitor_state *parent = head->next;

	head->mapper = parent->mapper->fields[num];
	head->field_num = num;
	head->array_num = 0;
	head->array_size = 1;
	if (parent->mapper->num_elements != NULL &&
	    parent->mapper->num_elements[num] != NULL)
		head->array_size =
		        parent->mapper->num_elements[num](parent->host_copy);

	void **field_ptr = aml_mapper_creator_field_ptr(head, parent);
	head->host_copy = PTR_OFF(c->host_memory, +, c->offset);
	head->device_ptr = *field_ptr;
	*field_ptr = PTR_OFF(c->device_memory, +, c->offset);
}

static int aml_mapper_creator_next_field(struct aml_mapper_creator *c)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(c->stack);
	struct aml_mapper_visitor_state *parent = head->next;

	if (parent == NULL)
		return -AML_EDOM;

	size_t num = head->field_num + 1;
	if (num >= parent->mapper->n_fields)
		return -AML_EDOM;
	aml_mapper_creator_init_field(c, num);
	return AML_SUCCESS;
}

static int aml_mapper_creator_first_field(struct aml_mapper_creator *c)
{
	const struct aml_mapper_visitor_state *parent = STACK_TOP(c->stack);
	if (parent->mapper->n_fields == 0)
		return -AML_EDOM;

	// Create new state.
	struct aml_mapper_visitor_state *state = malloc(sizeof *state);
	if (state == NULL)
		return -AML_ENOMEM;
	STACK_PUSH(c->stack, state);
	aml_mapper_creator_init_field(c, 0);
	return AML_SUCCESS;
}

static int aml_mapper_creator_parent(struct aml_mapper_creator *c)
{
	struct aml_mapper_visitor_state *head = c->stack;
	if (head == NULL)
		return -AML_EDOM;
	STACK_POP(c->stack, head);
	free(head);
	return c->stack == NULL ? -AML_EDOM : AML_SUCCESS;
}

static int aml_mapper_creator_next_array_element(struct aml_mapper_creator *c)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(c->stack);
	if (head->array_num + 1 >= head->array_size)
		return -AML_EDOM;
	head->device_ptr = PTR_OFF(head->device_ptr, +, head->mapper->size);
	head->host_copy = PTR_OFF(head->host_copy, +, head->mapper->size);
	head->array_num++;
	return AML_SUCCESS;
}

int aml_mapper_creator_branch(struct aml_mapper_creator **out,
                              struct aml_mapper_creator *c,
                              struct aml_area *area,
                              struct aml_area_mmap_options *area_opts,
                              struct aml_dma *dma_host_dst,
                              aml_dma_operator memcpy_host_dst)
{
	int err;

	// Get size to map.
	size_t tot_size = 0;
	for (size_t i = 0; i < c->stack->array_size; i++) {
		struct aml_mapper_visitor *visitor;
		size_t size;
		void *src_ptr = PTR_OFF(c->stack->device_ptr, +,
		                        c->stack->mapper->size);
		err = aml_mapper_visitor_create(&visitor, src_ptr,
		                                c->stack->mapper,
		                                c->dma_src_host,
		                                c->memcpy_src_host);
		if (err != AML_SUCCESS)
			return err;
		err = aml_mapper_size(visitor, &size);
		aml_mapper_visitor_destroy(visitor);
		if (err != AML_SUCCESS)
			return err;
		tot_size += size;
	}

	// Create a mapper creator for the field we just allocated.
	err = aml_mapper_creator_create(out, c->stack->device_ptr, tot_size,
	                                c->stack->mapper, area, area_opts,
	                                c->dma_src_host, dma_host_dst,
	                                c->memcpy_src_host, memcpy_host_dst);
	if (err != AML_SUCCESS)
		return err;
	// Set correct array size.
	(*out)->stack->array_size = c->stack->array_size;

	// Set pointer in parent structure to point to this new memory area.
	*aml_mapper_creator_field_ptr(c->stack, c->stack->next) =
	        (*out)->device_memory;

	// Move current creator to next field.
next_field:
	err = aml_mapper_creator_next_field(c);
	if (err != -AML_EDOM)
		return err;
	err = aml_mapper_creator_parent(c);
	if (err != AML_SUCCESS)
		return err;
	if (c->stack->mapper->n_fields == 0)
		goto next_field;
	err = aml_mapper_creator_next_array_element(c);
	if (err == -AML_EDOM)
		goto next_field;
	if (err != AML_SUCCESS)
		return err;
	err = aml_mapper_creator_first_field(c);
	if (err != -AML_EDOM)
		return err;
	goto next_field;
}

int aml_mapper_creator_next(struct aml_mapper_creator *c)
{
	int err;
	if (c->stack == NULL)
		return -AML_EDOM;
	if (c->stack->mapper->flags & AML_MAPPER_FLAG_SPLIT &&
	    c->stack->next != NULL)
		return -AML_EINVAL;

	// Copy to host.
	// If the current state is an array element. The whole array has already
	// been copied on first element.
	if (c->stack->array_num == 0) {
		err = aml_mapper_creator_memcpy(c);
		if (err != AML_SUCCESS)
			return err;
	}

	// Move to next field.
first_field:
	err = aml_mapper_creator_first_field(c);
	if (err != -AML_EDOM)
		return err;
next_array_element:
	if (c->stack->mapper->n_fields == 0)
		goto next_field;
	err = aml_mapper_creator_next_array_element(c);
	if (err == AML_SUCCESS)
		goto first_field;
	if (err != -AML_EDOM)
		return err;
next_field:
	err = aml_mapper_creator_next_field(c);
	if (err != -AML_EDOM)
		return err;
	err = aml_mapper_creator_parent(c);
	if (err != AML_SUCCESS)
		return err;
	goto next_array_element;
}

int aml_mapper_creator_create(struct aml_mapper_creator **out,
                              void *src_ptr,
                              size_t size,
                              struct aml_mapper *mapper,
                              struct aml_area *area,
                              struct aml_area_mmap_options *area_opts,
                              struct aml_dma *dma_src_host,
                              struct aml_dma *dma_host_dst,
                              aml_dma_operator memcpy_src_host,
                              aml_dma_operator memcpy_host_dst)
{
	if (out == NULL || src_ptr == NULL || mapper == NULL || area == NULL ||
	    dma_host_dst == NULL || memcpy_host_dst == NULL)
		return -AML_EINVAL;

	int err;
	struct aml_mapper_visitor *visitor;

	// Get the total size to map.
	if (size == 0) {
		err = aml_mapper_visitor_create(&visitor, src_ptr, mapper,
		                                dma_src_host, memcpy_src_host);
		if (err != AML_SUCCESS)
			return err;
		err = aml_mapper_size(visitor, &size);
		aml_mapper_visitor_destroy(visitor);
		if (err != AML_SUCCESS)
			return err;
	}

	// Allocate buffer in device to contain the total size to map.
	void *dst_ptr = aml_area_mmap(area, size, area_opts);
	if (dst_ptr == NULL)
		return aml_errno;

	// Allocate buffer on host to contain the total size to map.
	void *host_memory = malloc(size);
	if (host_memory == NULL)
		goto err_with_device_ptr;

	// Initialize mapper creator with enough host memory to copy source
	// structure.
	err = -AML_ENOMEM;
	struct aml_mapper_creator *c = malloc(sizeof(*c));
	if (c == NULL)
		goto err_with_host_memory;
	c->device_memory = dst_ptr;
	c->host_memory = host_memory;
	c->offset = 0;
	c->dma_src_host = dma_src_host;
	c->dma_host_dst = dma_host_dst;
	c->memcpy_src_host = memcpy_src_host;
	c->memcpy_host_dst = memcpy_host_dst;

	struct aml_mapper_visitor_state *head = malloc(sizeof(*head));
	if (head == NULL)
		goto err_with_creator;
	// This the original pointer to copy, not the target device pointer.
	head->device_ptr = src_ptr;
	head->host_copy = c->host_memory;
	head->mapper = mapper;
	head->field_num = 0;
	head->array_size = 1;
	head->array_num = 0;
	head->next = NULL;
	c->stack = head;
	*out = c;
	return AML_SUCCESS;

err_with_creator:
	free(c);
err_with_host_memory:
	free(host_memory);
err_with_device_ptr:
	aml_area_munmap(area, dst_ptr, size);
	return err;
}

int aml_mapper_creator_finish(struct aml_mapper_creator *c,
                              void **ptr,
                              size_t *size)
{
	int err;
	err = aml_dma_copy_custom(c->dma_host_dst, c->device_memory,
	                          c->host_memory, c->memcpy_host_dst,
	                          (void *)c->offset);
	if (err != AML_SUCCESS)
		return err;
	if (ptr != NULL)
		*ptr = c->device_memory;
	if (size != NULL)
		*size = c->offset;
	while (aml_mapper_creator_parent(c) == AML_SUCCESS)
		;
	free(c->host_memory);
	free(c);
	return AML_SUCCESS;
}
