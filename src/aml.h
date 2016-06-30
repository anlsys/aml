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
	char *path;
	int fd;
	size_t maxsize;
};

int aml_init(int *argc, char **argv[]);
int aml_finalize(void);

int aml_node_init(struct aml_node *, struct bitmask *, size_t);
int aml_node_destroy(struct aml_node *);

int aml_malloc(struct aml_alloc *, size_t, size_t, struct aml_node *);
int aml_free(struct aml_alloc *);

int aml_pull_sync(struct aml_alloc *, unsigned long, struct aml_node *);
int aml_push_sync(struct aml_alloc *, unsigned long, struct aml_node *);
#endif
