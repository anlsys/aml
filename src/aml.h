#ifndef AML_H
#define AML_H 1

#include<numa.h>
#include <stdlib.h>

/* An allocation.
 *
 * Contains information about an allocation.
 */
struct aml_node;
struct aml_alloc;

struct aml_alloc {
	void *start;
	size_t memsize;
	size_t blocksize;
	size_t numblocks;
	struct aml_node **nodemap;
};

struct aml_node {
	struct bitmask *mask;
	int numaid;
};

int aml_init(int *argc, char **argv[]);
int aml_finalize(void);

int aml_node_init(struct aml_node *, unsigned int);
int aml_node_destroy(struct aml_node *);

int aml_malloc(struct aml_alloc *, size_t, size_t, struct aml_node *);
int aml_free(struct aml_alloc *);

inline size_t aml_block_size(struct aml_alloc *a) {
	return a->blocksize;
}

int aml_block_address(struct aml_alloc *, size_t, void **);

int aml_block_move(struct aml_alloc *, size_t, struct aml_node *);
int aml_block_copy(struct aml_alloc *, size_t, struct aml_alloc *, size_t);
#endif
