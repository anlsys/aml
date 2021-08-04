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

//------------------------------------------------------------------------//
// Utils
//------------------------------------------------------------------------//

struct aml_tree_node_generic {
	struct aml_tree_node *parent;
	size_t num_children;
	size_t max_children;
	struct aml_tree_node **children;
};

static int aml_tree_node_generic_extend(struct aml_tree_node_generic *n)
{
	if (n == NULL)
		return -AML_EINVAL;
	if (n->num_children < n->max_children)
		return AML_SUCCESS;

	const size_t max_children = n->max_children * 2;
	struct aml_tree_node **children =
	        realloc(n->children, max_children * sizeof(*children));
	if (children == NULL)
		return -AML_ENOMEM;
	n->children = children;
	n->max_children = max_children;
	return AML_SUCCESS;
}

struct aml_tree_node *
aml_tree_node_generic_next_sibling(struct aml_tree_node_data *data)
{
	struct aml_tree_node_generic *node = (struct aml_tree_node_generic*)data;
	if (node->parent == NULL)
		return NULL;
	for (size_t i=0; i<node->parent->num_children; i++)
		if (node->parent->children[i] == node) {
			if (i + 1 < node->parent->num_children)
				return node->parent->children[i+1];
			else
				return NULL;
		}
	return NULL;
}

struct aml_tree_node *
aml_tree_node_generic_prev_sibling(struct aml_tree_node_data *data)
{
	struct aml_tree_node_generic *node = (struct aml_tree_node_generic*)data;
	if (node->parent == NULL)
		return NULL;
	for (size_t i=0; i<node->parent->num_children; i++)
		if (node->parent->children[i] == node) {
			if (i > 0)
				return node->parent->children[i-1];
			else
				return NULL;
		}
	return NULL;
}

struct aml_tree_node *(*parent)(struct aml_tree_node_data *data) {
	struct aml_tree_node_generic *node = (struct aml_tree_node_generic*)data;
	return node->parent;
}

struct aml_tree_node *
aml_tree_node_generic_first_child(struct aml_tree_node_data *data)
{
	struct aml_tree_node_generic *node =
	        (struct aml_tree_node_generic *)data;
	if (node->num_children > 0)
		return node->children[0];
	return NULL;
}

struct aml_tree_node *(*last_child)(struct aml_tree_node_data *data);
void *(*destroy)(struct aml_tree_node *node);

int (*append)(struct aml_tree_node_data *data,
              size_t num_children,
              struct aml_tree_node **children);

static struct aml_tree_node **
aml_tree_node_find_in_parent(struct aml_tree_node *p)
{
	struct aml_tree_node **node = root->parent->children;
	while ((*node) != p) {
		node++;
		if (node >= p->children + p->num_children)
			return NULL;
	}
	return p;
}

//------------------------------------------------------------------------//
// Header Implementation
//------------------------------------------------------------------------//

struct aml_tree_node *aml_tree_node_create()
{
	const size_t max_children = 4;
	struct aml_tree_node *ret = malloc(sizeof(*ret));
	if (ret == NULL)
		return NULL;
	ret->parent = NULL;
	ret->num_children = 0;
	ret->max_children = max_children;
	ret->data = NULL;

	ret->children = malloc(max_children * sizeof(struct aml_tree_node *));
	if (ret->children == NULL) {
		free(ret);
		return NULL;
	}
	return ret;
}

void *aml_tree_node_destroy(struct aml_tree_node *node)
{
	if (node == NULL)
		return NULL;
	void *ret = node->data;
	free(node->children);
	free(node);
	return ret;
}

int aml_tree_node_destroy_recursive(struct aml_tree_node *root,
                                    aml_tree_node_destroy_fn delete)
{
	int err;
	struct aml_tree_node **node = &root, *n = root;

	// Iterate until the whole tree is deleted.
	while (1) {
		// If current node has no child, we can free it.
		if (n->num_children == 0) {
			// Cleanup node.
			if (delete != NULL) {
				err = delete (n->data);
				if (err != AML_SUCCESS)
					return err;
			}
			free(n->children);
			free(n);

			if (n->parent != NULL)
				n->parent->num_children--;

			// If we just freed root, we stop.
			// root is the last node to free.
			if (n == root)
				break;

			// Else we move on to next sibling
			if (n->parent->num_children > 0)
				node++;

			// If no sibling is left, move to parent.
			// We store address in children array.
			// If no such array exists,
			// we store address on stack, i.e root.
			else if (n->parent->parent == NULL) {
				assert(n->parent == root);
				node = &root;
			} else
				node = aml_tree_node_find_in_parent(n->parent);
		}
		// If we did not free anything, there are children.
		// Therefore we descend them.
		else
			node = n->children;
		n = *node;
	}
}

struct aml_tree_node *aml_tree_node_insert_leaf(struct aml_tree_node *node,
                                                void *data)
{
	// Make sure there is enough space in node children array.
	int err = aml_tree_node_extend(node);
	if (err != AML_SUCCESS)
		goto error;

	// Create leaf node
	struct aml_tree_node *n = aml_tree_node_create(data);
	if (n == NULL) {
		err = AML_ENOMEM;
		goto error;
	}

	// Connect leaf node
	n->parent = node;
	node->children[node->num_children] = n;
	node->num_children++;

	return n;
error:
	aml_errno = err;
	return NULL;
}

struct aml_tree_node *
aml_tree_node_connect_children(struct aml_tree_node *node,
                               size_t num_children,
                               struct aml_tree_node **children)
{
	// Make sure there is enough space in node children array.
	int err;

	while (node->max_children - node->num_children <= num_children) {
		err = aml_tree_node_extend(node);
		if (err != AML_SUCCESS)
			return err;
	}

	// Connect parent to children
	memcpy(node->children + node->num_children, children,
	       num_children * sizeof(*children));
	node->num_children += num_children;

	// Connect children to parent
	for (size_t i = 0; i < num_children; i++)
		children[i]->parent = node;

	return AML_SUCCESS;
}

int aml_tree_node_remove_child(struct aml_tree_node *node,
                               struct aml_tree_node **child)
{
	if (node == NULL || child == NULL || *child == NULL)
		return -AML_EINVAL;

	// Find child position in parent children array.
	if (child < node->children ||
	    child >= node->children + node->num_children)
		child = aml_tree_node_find_in_parent(*child);
	if (child == NULL)
		return -AML_EINVAL;

	size_t len = ((intptr_t)(child + 1) - (intptr_t)node->children);
	len = (sizeof(node) * node->num_children) - len;
	memmove(child, child + 1, len);
	node->num_children--;
	return AML_SUCCESS;
}

int aml_tree_node_coords(struct aml_tree_node *node,
                         size_t *coords,
                         const size_t max_coords)
{
	struct aml_tree_node **n;
	ssize_t i = max_coords;

	while (--i >= 0 && node->parent != NULL) {
		n = aml_tree_node_find_in_parent(node);
		coords[i] = (n - node->parent->children) / sizeof(node);
		node = node->parent;
	}

	// If we did not reach root, the coordinate set is incomplete
	// due to not enough max_coords.
	if (node->parent != NULL)
		return -AML_EINVAL;
	// If we reached root and we had more coordinate than needed,
	// we shift coordinates to the left.
	if (i > 0)
		memmove(coords, coords + i, max_coords - i);

	return AML_SUCCESS;
}

struct aml_tree_node *aml_tree_node_get(struct aml_tree_node *root,
                                        size_t *coords,
                                        const size_t num_coords)
{
	if (root == NULL)
		return NULL;

	for (size_t i = 0; i < num_coords; i++) {
		if (coords[i] >= root->num_children)
			return NULL;
		root = root->children[i];
	}
	return root;
}

struct aml_tree_node *aml_tree_node_next(struct aml_tree_node **node,
                                         struct aml_tree_node *key,
                                         aml_tree_node_cmp_fn cmp,
                                         aml_tree_node_walker_fn walker) {
	do {
		if (cmp(*node, key) == 0)
		        break;
		node = walker(node);
	} while (node != NULL);
	return node;
}

struct aml_tree_node **
aml_tree_node_depth_first_walker(struct aml_tree_node **node)
{
	if (node == NULL)
		return NULL;

	struct aml_tree_node *n = *node;

	// If there are children, descend first one.
	if (n->num_children != 0)
		return n->children;
	// Node does not have children nor parent then stop iteration.
	if (n->parent == NULL)
		return NULL;

	// Climb parent as long as node is the right most node.
	while (aml_tree_node_is_last_child(node)) {
		n = n->parent;
		// We reached back to root (last node) then stop iteration.
		if (n->parent == NULL)
			return NULL;
		node = aml_tree_node_find_in_parent(n);
	}

	return node + 1;
}
