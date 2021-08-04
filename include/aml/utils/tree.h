/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#ifndef AML_TREE_H
#define AML_TREE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup aml_tree "AML Tree API"
 * @brief AML Tree API
 *
 * Generic tree implementation:
 * Serial tree for pushing and poping pointers.
 * @{
 **/

struct aml_tree_node_data;
struct aml_tree_node_ops;

struct aml_tree_node {
	struct aml_tree_node_data *data;
	struct aml_tree_node_ops *ops;
	void *user_data;
};

struct aml_tree_node_ops {
	struct aml_tree_node *(*next_sibling)(struct aml_tree_node_data *data);
	struct aml_tree_node *(*prev_sibling)(struct aml_tree_node_data *data);
	struct aml_tree_node *(*parent)(struct aml_tree_node_data *data);
	struct aml_tree_node *(*first_child)(struct aml_tree_node_data *data);
	struct aml_tree_node *(*last_child)(struct aml_tree_node_data *data);
	void *(*destroy)(struct aml_tree_node *node);

	int (*append)(struct aml_tree_node_data *data,
		      size_t num_children,
		      struct aml_tree_node **children);
}


//------------------------------------------------------------------------//
// Tree Walk
//------------------------------------------------------------------------//

typedef struct aml_tree_node **(*aml_tree_node_walker_fn)(
        struct aml_tree_node **);

struct aml_tree_node **
aml_tree_node_depth_first_walker(struct aml_tree_node **node);

typedef int (*aml_tree_node_cmp_fn)(struct aml_tree_node *,
                                    struct aml_tree_node *);

struct aml_tree_node *aml_tree_node_next(struct aml_tree_node **node,
                                         struct aml_tree_node *key,
                                         aml_tree_node_cmp_fn cmp,
                                         aml_tree_node_walker_fn walker);

struct aml_tree_node *aml_tree_node_get(struct aml_tree_node *root,
                                        size_t *coords,
                                        const size_t num_coords);

//------------------------------------------------------------------------//
// Creation / Destruction
//------------------------------------------------------------------------//

struct aml_tree_node *aml_tree_node_create(void *data);
void *aml_tree_node_destroy(struct aml_tree_node *node);

typedef int (*aml_tree_node_destroy_fn)(void *);
int aml_tree_node_destroy_recursive(struct aml_tree_node *root,
                                    aml_tree_node_destroy_fn delete);

//------------------------------------------------------------------------//
// Insertion / Removal
//------------------------------------------------------------------------//

struct aml_tree_node *aml_tree_node_insert_leaf(struct aml_tree_node *node,
                                                void *data);

int aml_tree_node_connect_children(struct aml_tree_node *node,
                                   size_t num_children,
                                   struct aml_tree_node **children);

int aml_tree_node_remove_child(struct aml_tree_node *node,
                               struct aml_tree_node **child);

//------------------------------------------------------------------------//
// Attributes
//------------------------------------------------------------------------//

inline size_t aml_tree_node_depth(struct aml_tree_node *node)
{
	size_t depth = 0;
	while ((node = node->parent) != NULL)
		depth++;
	return depth;
}

inline struct aml_tree_node *aml_tree_node_root(struct aml_tree_node *node)
{
	while (node->parent != NULL)
		node = node->parent;
	return node;
}

inline struct aml_tree_node *
aml_tree_node_first_leaf(struct aml_tree_node *root)
{
	while (node->num_children != 0)
		node = node->children[0];
	return node;
}

inline struct aml_tree_node *aml_tree_node_last_leaf(struct aml_tree_node *root)
{
	while (node->num_children != 0)
		node = node->children[node->num_children - 1];
	return node;
}

inline int aml_tree_node_is_leaf(struct aml_tree_node *node)
{
	return node->num_children == 0;
}

inline int aml_tree_node_is_root(struct aml_tree_node *node)
{
	return node->parent == NULL;
}

int aml_tree_node_coords(struct aml_tree_node *node,
                         size_t *coords,
                         const size_t max_coords);

/**
 * @}
 **/

#ifdef __cplusplus
}
#endif

#endif // AML_TREE_H
