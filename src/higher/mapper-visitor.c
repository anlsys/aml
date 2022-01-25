/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "aml.h"

#include "aml/higher/mapper.h"
#include "aml/higher/mapper/visitor.h"
#include "aml/layout/dense.h"
#include "aml/utils/inner-malloc.h"
#include "aml/utils/queue.h"

#include "internal/utstack.h"

#define PTR_OFF(ptr, sign, off) (void *)((intptr_t)(ptr)sign(intptr_t)(off))

// Create a new instance of visitor over ptr described by mapper where
// dma and its memcpy operator copy from area where ptr is allocated to host.
int aml_mapper_visitor_create(struct aml_mapper_visitor **out,
                              void *ptr,
                              struct aml_mapper *mapper,
                              struct aml_dma *dma,
                              aml_dma_operator memcpy_op)
{
	if (out == NULL || ptr == NULL || mapper == NULL)
		return -AML_EINVAL;

	struct aml_mapper_visitor *it = malloc(sizeof(*it));
	if (it == NULL)
		return -AML_ENOMEM;
	it->stack = NULL;
	it->dma = dma;
	it->memcpy_op = memcpy_op;

	// Create first state representing the first iteration and push
	// it to the state stack.
	struct aml_mapper_visitor_state *head = malloc(sizeof(*head));
	if (head == NULL) {
		free(it);
		return -AML_ENOMEM;
	}
	head->device_ptr = ptr;
	head->host_copy = NULL;
	head->mapper = mapper;
	head->field_num = 0;
	head->array_size = 1;
	head->array_num = 0;
	head->next = NULL;
	STACK_PUSH(it->stack, head);

	*out = it;
	return AML_SUCCESS;
}

int aml_mapper_visitor_subtree(const struct aml_mapper_visitor *it,
                               struct aml_mapper_visitor **subtree_visitor)
{
	// Cut the stack into the top host_copy.
	// Since there is no parent in the stack, the visit is limited to
	// child fields.
	return aml_mapper_visitor_create(subtree_visitor, it->stack->device_ptr,
	                                 it->stack->mapper, it->dma,
	                                 it->memcpy_op);
}

int aml_mapper_visitor_destroy(struct aml_mapper_visitor *it)
{
	if (it == NULL)
		return -AML_EINVAL;
	while (aml_mapper_visitor_parent(it) == AML_SUCCESS)
		;
	free(it);
	return AML_SUCCESS;
}

// Attempt to visit next element of currently visited array of elements.
// If there is no more element to visit, -AML_EDOM is returned, else
// visitor head state is updated to the state of its next array element.
int aml_mapper_visitor_next_array_element(struct aml_mapper_visitor *it)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(it->stack);
	assert(head != NULL);
	// Check if we reached array bounds
	if ((head->array_num + 1) >= head->array_size)
		return -AML_EDOM;
	if (head->device_ptr == NULL)
		return -AML_EINVAL;
	// The device pointer of next array element is the same device pointer
	// offseted by the element size.
	head->device_ptr = PTR_OFF(head->device_ptr, +, head->mapper->size);
	// Update index.
	head->array_num++;
	return AML_SUCCESS;
}

int aml_mapper_visitor_prev_array_element(struct aml_mapper_visitor *it)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(it->stack);
	assert(head != NULL);
	if (head->array_num == 0)
		return -AML_EDOM;
	if (head->device_ptr == NULL)
		return -AML_EINVAL;
	head->device_ptr = PTR_OFF(head->device_ptr, -, head->mapper->size);
	head->array_num--;
	return AML_SUCCESS;
}

static int aml_mapper_visitor_field(struct aml_mapper_visitor *it,
                                    const size_t num)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(it->stack);
	struct aml_mapper_visitor_state *parent = head->next;

	// The new device pointer can be read in the parent byte copy.
	// And it is the pointer at the offset of the next field offset.
	head->device_ptr = *(void **)PTR_OFF(parent->host_copy, +,
	                                     parent->mapper->offsets[num]);
	// Update the other fields.
	head->field_num = num;
	head->mapper = parent->mapper->fields[num];
	head->array_num = 0;
	head->array_size = 1;
	if (parent->mapper->num_elements != NULL &&
	    parent->mapper->num_elements[num] != NULL)
		head->array_size =
		        parent->mapper->num_elements[num](parent->host_copy);

	return AML_SUCCESS;
}

// Attempt to visit next field of the parent of the element of currently
// visited. If there is no more sibling field to visit, -AML_EDOM is returned,
// else visitor head state is updated to the state of its next sibling.
int aml_mapper_visitor_next_field(struct aml_mapper_visitor *it)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(it->stack);
	assert(head != NULL);
	struct aml_mapper_visitor_state *parent = head->next;

	// If this has no parent, there is no next field in the parent.
	if (parent == NULL)
		return -AML_EDOM;

	// Check bounds of the number of fields.
	const size_t num = head->field_num + 1;
	if (num >= parent->mapper->n_fields)
		return -AML_EDOM;

	return aml_mapper_visitor_field(it, num);
}

int aml_mapper_visitor_prev_field(struct aml_mapper_visitor *it)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(it->stack);
	assert(head != NULL);
	if (head->next == NULL)
		return -AML_EDOM;
	if (head->field_num == 0)
		return -AML_EDOM;

	return aml_mapper_visitor_field(it, head->field_num - 1);
}

// Attempt to visit first child of the element of currently visited.
// If there no child to visit, -AML_EDOM is returned, else a new state
// representing first child element is pushed on top of the state stack.
int aml_mapper_visitor_first_field(struct aml_mapper_visitor *it)
{
	struct aml_mapper_visitor_state *head = STACK_TOP(it->stack);
	assert(head != NULL);

	if (head->mapper->n_fields == 0)
		return -AML_EDOM;
	if (head->device_ptr == NULL)
		return -AML_EINVAL;

	// Byte copy current field structure to easily have access to number
	// of fields and fields value later.
	if (it->dma != NULL) {
		const size_t size = head->mapper->size;
		void *element = malloc(size);
		if (element == NULL)
			return -AML_ENOMEM;

		int err = aml_dma_copy_custom(
		        it->dma, (struct aml_layout *)element,
		        (struct aml_layout *)head->device_ptr, it->memcpy_op,
		        (void *)size);
		if (err != AML_SUCCESS) {
			free(element);
			return err;
		}
		if (head->host_copy != NULL)
			free(head->host_copy);
		head->host_copy = element;
	} else
		head->host_copy = head->device_ptr;

	// Create the child state.
	struct aml_mapper_visitor_state *next = malloc(sizeof *next);
	if (next == NULL)
		return -AML_ENOMEM;

	// Device ptr is the pointer in current state host copy at the
	// offset of the first field.
	next->device_ptr =
	        *(void **)PTR_OFF(head->host_copy, +, head->mapper->offsets[0]);
	// Set next time this field is descended.
	next->host_copy = NULL;
	// Initialization for stack primitives
	next->next = NULL;
	// Mapper of new visited field.
	next->mapper = head->mapper->fields[0];
	// The new field is the first field.
	next->field_num = 0;
	// Start at first element.
	next->array_num = 0;
	// Default to not an array.
	next->array_size = 1;
	// Set array size if the state is an array.
	if (head->mapper->num_elements != NULL &&
	    head->mapper->num_elements[0] != NULL)
		next->array_size =
		        head->mapper->num_elements[0](head->host_copy);
	// Push the new state at the top of the stack. This is the current
	// state now.
	STACK_PUSH(it->stack, next);

	return AML_SUCCESS;
}

int aml_mapper_visitor_parent(struct aml_mapper_visitor *it)
{
	struct aml_mapper_visitor_state *head = it->stack;
	if (head == NULL)
		return -AML_EDOM;
	STACK_POP(it->stack, head);
	if (head->host_copy != NULL && head->host_copy != head->device_ptr)
		free(head->host_copy);
	free(head);
	return it->stack == NULL ? -AML_EDOM : AML_SUCCESS;
}

void *aml_mapper_visitor_ptr(struct aml_mapper_visitor *it)
{
	return it->stack->device_ptr;
}

size_t aml_mapper_visitor_size(struct aml_mapper_visitor *it)
{
	return it->stack->mapper->size * it->stack->array_size;
}

int aml_mapper_visitor_is_array(struct aml_mapper_visitor *it)
{
	return it->stack->array_size > 1;
}

size_t aml_mapper_visitor_array_len(struct aml_mapper_visitor *it)
{
	return it->stack->array_size;
}

int aml_mapper_size(const struct aml_mapper_visitor *visitor, size_t *size)
{
	struct aml_mapper_visitor *v;
	size_t tot = 0;
	int err;

	err = aml_mapper_visitor_subtree(visitor, &v);
	if (err != AML_SUCCESS)
		return err;

add_size:
	tot += aml_mapper_visitor_size(v);
first_field:
	err = aml_mapper_visitor_first_field(v);
	if (err == AML_SUCCESS)
		goto check_split;
	if (err != -AML_EDOM)
		goto error;
next_array_element:
	if (v->stack->mapper->n_fields == 0)
		goto next_field;
	err = aml_mapper_visitor_next_array_element(v);
	if (err == AML_SUCCESS)
		goto first_field;
	if (err != -AML_EDOM)
		goto error;
next_field:
	err = aml_mapper_visitor_next_field(v);
	if (err == AML_SUCCESS)
		goto check_split;
	if (err != -AML_EDOM)
		goto error;
	err = aml_mapper_visitor_parent(v);
	if (err == AML_SUCCESS)
		goto next_array_element;
	if (err != -AML_EDOM)
		goto error;
	goto success;
check_split:
	// Skip nodes that will be split in a different allocation.
	if (v->stack->mapper->flags & AML_MAPPER_FLAG_SPLIT)
		goto next_field;
	else
		goto add_size;
success:
	aml_mapper_visitor_destroy(v);
	*size = tot;
	return AML_SUCCESS;
error:
	aml_mapper_visitor_destroy(v);
	return err;
}
