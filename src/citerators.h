#ifndef CITERATORS_H
#define CITERATORS_H 1

enum citerator_type_e {
	CITERATOR_RANGE,
	CITERATOR_CONS,
	CITERATOR_REPEAT,
	CITERATOR_HILBERT2D,
	CITERATOR_PRODUCT,
	CITERATOR_SLICE
};

typedef struct citerator_s *citerator_t;

typedef int citerator_index_t;

citerator_t citerator_alloc(enum citerator_type_e type);
void citerator_free(citerator_t iterator);
citerator_t citerator_dup(const citerator_t iterator);

int citerator_type(citerator_t iterator, enum citerator_type_e *type);
int citerator_dimension(citerator_t iterator, citerator_index_t *dimension);

int citerator_next(citerator_t iterator, citerator_index_t *indexes);
int citerator_peek(const citerator_t iterator, citerator_index_t *indexes);
int citerator_size(const citerator_t iterator, citerator_index_t *size);
int citerator_rewind(citerator_t iterator);
int citerator_split(const citerator_t iterator, citerator_index_t n,
		    citerator_t *results);
int citerator_nth(const citerator_t iterator, citerator_index_t n,
		  citerator_index_t *indexes);
int citerator_n(const citerator_t iterator, const citerator_index_t *indexes,
		citerator_index_t *n);
int citerator_pos(const citerator_t iterator, citerator_index_t *n);

int citerator_skip(citerator_t iterator);
int citerator_cyclic_next(citerator_t iterator, citerator_index_t *indexes,
			  int *looped);

int citerator_range_init(citerator_t iterator, citerator_index_t first,
			 citerator_index_t last, citerator_index_t step);

int citerator_cons_init(citerator_t iterator, citerator_t src,
			citerator_index_t n);

int citerator_repeat_init(citerator_t iterator, citerator_t src,
			  citerator_index_t n);

int citerator_hilbert2d_init(citerator_t iterator, citerator_index_t order);

int citerator_product_add(citerator_t iterator, citerator_t added_iterator);
int citerator_product_add_copy(citerator_t iterator,
			       citerator_t added_iterator);
int citerator_product_count(const citerator_t iterator,
			    citerator_index_t *count);
int citerator_product_split_dim(const citerator_t iterator,
				citerator_index_t dim, citerator_index_t n,
				citerator_t *results);

int citerator_slice_init(citerator_t iterator, citerator_t src,
			 citerator_t indexer);
#endif
