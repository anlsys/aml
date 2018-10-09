#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include "citerators.h"

void test_range_iterator(void)
{
	citerator_t it;
	citerator_index_t dim;
	citerator_index_t size;
	citerator_index_t indexes[1];
	enum citerator_type_e type;
	citerator_t its[3];
	int looped;
	int i;
	citerator_index_t ith;

	it = citerator_alloc(CITERATOR_RANGE);
	assert(it != NULL);
	assert(citerator_type(it, &type) == 0);
	assert(type == CITERATOR_RANGE);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 0);

	assert(citerator_range_init(it, 0, 3, 1) == 0);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 1);
	assert(citerator_size(it, &size) == 0);
	assert(size == 4);

	for (i = 0; i < 4; i++) {
		assert(citerator_nth(it, i, indexes) == 0);
		assert(indexes[0] == i);
		assert(citerator_n(it, indexes, &ith) == 0);
		assert(ith == i);
		ith = -1;
		assert(citerator_pos(it, &ith) == 0);
		assert(ith == i);
		assert(citerator_peek(it, indexes) == 0);
		assert(indexes[0] == i);
		assert(citerator_next(it, indexes) == 0);
		assert(indexes[0] == i);
	}
	assert(citerator_next(it, indexes) == -ERANGE);
	assert(citerator_peek(it, indexes) == -ERANGE);

	assert(citerator_rewind(it) == 0);
	looped = 0;
	i = 0;
	while (!looped) {
		assert(citerator_cyclic_next(it, indexes, &looped) == 0);
		assert(indexes[0] == i);
		i++;
	}
	assert(citerator_peek(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(citerator_cyclic_next(it, indexes, &looped) == 0);
	assert(indexes[0] == 0);
	assert(looped == 0);

	for (i = 1; i < 4; i++) {
		assert(citerator_next(it, indexes) == 0);
		assert(indexes[0] == i);
	}
	assert(citerator_next(it, indexes) == -ERANGE);
	assert(citerator_next(it, indexes) == -ERANGE);
	assert(citerator_cyclic_next(it, indexes, &looped) == 0);
	assert(indexes[0] == 0);
	assert(looped == 1);

	assert(citerator_range_init(it, 3, 0, -1) == 0);
	assert(citerator_size(it, &size) == 0);
	assert(size == 4);

	for (i = 3; i >= 0; i--) {
		assert(citerator_next(it, indexes) == 0);
		assert(indexes[0] == i);
	}
	assert(citerator_next(it, indexes) == -ERANGE);

	assert(citerator_range_init(it, 3, 0, 1) == 0);
	assert(citerator_size(it, &size) == 0);
	assert(size == 0);

	assert(citerator_range_init(it, 0, 9, 1) == 0);
	assert(citerator_size(it, &size) == 0);
	assert(size == 10);
	assert(citerator_split(it, 3, its) == 0);
	assert(citerator_size(its[0], &size) == 0);
	assert(size == 4);
	for (int i = 0; i < 4; i++) {
		assert(citerator_next(its[0], indexes) == 0);
		assert(indexes[0] == i);
	}
	assert(citerator_size(its[1], &size) == 0);
	assert(size == 3);
	for (int i = 4; i < 7; i++) {
		assert(citerator_next(its[1], indexes) == 0);
		assert(indexes[0] == i);
	}
	assert(citerator_size(its[2], &size) == 0);
	assert(size == 3);
	for (int i = 7; i < 10; i++) {
		assert(citerator_next(its[2], indexes) == 0);
		assert(indexes[0] == i);
	}
	citerator_free(its[0]);
	citerator_free(its[1]);
	citerator_free(its[2]);

	assert(citerator_split(it, 24, its) == -EDOM);
	assert(citerator_split(it, 0, its) == -EDOM);
	assert(citerator_split(it, -1, its) == -EDOM);
	citerator_free(it);
}

void test_product_iterators(void)
{
	int i, j, k;
	int looped;
	citerator_t it;
	citerator_t tmp;
	citerator_index_t dim;
	citerator_index_t count;
	citerator_index_t size;
	citerator_index_t indexes[3];
	citerator_index_t indexes2[3];
	enum citerator_type_e type;
	citerator_t its[5];
	citerator_index_t ith;

	it = citerator_alloc(CITERATOR_PRODUCT);
	assert(it != NULL);
	assert(citerator_type(it, &type) == 0);
	assert(type == CITERATOR_PRODUCT);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 0);
	assert(citerator_product_count(it, &count) == 0);
	assert(count == 0);
	assert(citerator_size(it, &size) == 0);
	assert(size == 0);
	assert(citerator_peek(it, indexes) == -EINVAL);

	tmp = citerator_alloc(CITERATOR_RANGE);
	citerator_range_init(tmp, 0, 3, 1);
	assert(citerator_product_add(it, tmp) == 0);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 1);
	assert(citerator_product_count(it, &count) == 0);
	assert(count == 1);
	assert(citerator_size(it, &size) == 0);
	assert(size == 4);
	assert(citerator_peek(it, indexes) == 0);
	assert(indexes[0] == 0);

	tmp = citerator_alloc(CITERATOR_RANGE);
	citerator_range_init(tmp, 1, -1, -1);
	assert(citerator_product_add(it, tmp) == 0);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 2);
	assert(citerator_product_count(it, &count) == 0);
	assert(count == 2);
	assert(citerator_size(it, &size) == 0);
	assert(size == 3 * 4);
	assert(citerator_peek(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 1);

	k = 0;
	for (i = 0; i <= 3; i++) {
		for (j = 1; j >= -1; j--, k++) {
			assert(citerator_nth(it, k, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == j);
			assert(citerator_pos(it, &ith) == 0);
			assert(ith == k);
			ith = -1;
			assert(citerator_n(it, indexes, &ith) == 0);
			assert(ith == k);
			assert(citerator_peek(it, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == j);
			assert(citerator_next(it, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == j);
		}
	}
	assert(citerator_peek(it, indexes) == -ERANGE);
	assert(citerator_next(it, indexes) == -ERANGE);
	assert(citerator_next(it, indexes) == -ERANGE);
	assert(citerator_cyclic_next(it, indexes, &looped) == 0);
	assert(looped == 1);
	assert(indexes[0] == 0);
	assert(indexes[1] == 1);

	assert(citerator_rewind(it) == 0);
	for (k = 0; k < 3; k++) {
		for (i = 0; i <= 3; i++) {
			for (j = 1; j >= -1; j--) {
				assert(citerator_cyclic_next
				       (it, indexes, &looped) == 0);
				assert(indexes[0] == i);
				assert(indexes[1] == j);
			}
		}
		assert(looped == 1);
	}

	citerator_rewind(it);

	tmp = citerator_alloc(CITERATOR_RANGE);
	citerator_range_init(tmp, -5, 5, 1);
	assert(citerator_product_add(it, tmp) == 0);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 3);
	assert(citerator_size(it, &size) == 0);
	assert(size == 3 * 4 * 11);

	//split first dimension
	assert(citerator_product_split_dim(it, 0, 2, its) == 0);
	for (i = 0; i <= 1; i++) {
		for (j = 1; j >= -1; j--) {
			for (k = -5; k <= 5; k++) {
				assert(citerator_next(its[0], indexes) == 0);
				assert(indexes[0] == i);
				assert(indexes[1] == j);
				assert(indexes[2] == k);
			}
		}
	}
	assert(citerator_next(its[0], indexes) == -ERANGE);
	for (i = 2; i <= 3; i++) {
		for (j = 1; j >= -1; j--) {
			for (k = -5; k <= 5; k++) {
				assert(citerator_next(its[1], indexes) == 0);
				assert(indexes[0] == i);
				assert(indexes[1] == j);
				assert(indexes[2] == k);
			}
		}
	}
	assert(citerator_next(its[1], indexes) == -ERANGE);
	citerator_free(its[0]);
	citerator_free(its[1]);

	//split second dimension
	assert(citerator_product_split_dim(it, 1, 2, its) == 0);
	for (i = 0; i <= 3; i++) {
		for (j = 1; j >= 0; j--) {
			for (k = -5; k <= 5; k++) {
				assert(citerator_next(its[0], indexes) == 0);
				assert(indexes[0] == i);
				assert(indexes[1] == j);
				assert(indexes[2] == k);
			}
		}
	}
	assert(citerator_next(its[0], indexes) == -ERANGE);
	for (i = 0; i <= 3; i++) {
		for (j = -1; j >= -1; j--) {
			for (k = -5; k <= 5; k++) {
				assert(citerator_next(its[1], indexes) == 0);
				assert(indexes[0] == i);
				assert(indexes[1] == j);
				assert(indexes[2] == k);
			}
		}
	}
	assert(citerator_next(its[1], indexes) == -ERANGE);
	citerator_free(its[0]);
	citerator_free(its[1]);

	//split third dimension
	assert(citerator_product_split_dim(it, 2, 2, its) == 0);
	for (i = 0; i <= 3; i++) {
		for (j = 1; j >= -1; j--) {
			for (k = -5; k <= 0; k++) {
				assert(citerator_next(its[0], indexes) == 0);
				assert(indexes[0] == i);
				assert(indexes[1] == j);
				assert(indexes[2] == k);
			}
		}
	}
	assert(citerator_next(its[0], indexes) == -ERANGE);
	for (i = 0; i <= 3; i++) {
		for (j = 1; j >= -1; j--) {
			for (k = 1; k <= 5; k++) {
				assert(citerator_next(its[1], indexes) == 0);
				assert(indexes[0] == i);
				assert(indexes[1] == j);
				assert(indexes[2] == k);
			}
		}
	}
	assert(citerator_next(its[1], indexes) == -ERANGE);
	citerator_free(its[0]);
	citerator_free(its[1]);

	assert(citerator_product_split_dim(it, 2, 15, its) == -EDOM);

	assert(citerator_split(it, 5, its) == 0);

	assert(citerator_rewind(it) == 0);
	for (i = 0; i < 5; i++) {
		while (citerator_next(its[i], indexes)) {
			assert(citerator_next(it, indexes2) == 0);
			assert(indexes[0] == indexes2[0]);
			assert(indexes[1] == indexes2[1]);
			assert(indexes[2] == indexes2[2]);
		}
	}

	for (i = 0; i < 5; i++)
		citerator_free(its[i]);
	citerator_free(it);

}

void test_repeat_iterator(void)
{
	citerator_t it;
	citerator_t tmp;
	citerator_index_t dim;
	citerator_index_t size;
	enum citerator_type_e type;
	citerator_index_t indexes[1];
	citerator_t its[2];
	citerator_index_t ith;

	it = citerator_alloc(CITERATOR_REPEAT);
	assert(it != NULL);
	assert(citerator_type(it, &type) == 0);
	assert(type == CITERATOR_REPEAT);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 0);

	tmp = citerator_alloc(CITERATOR_RANGE);
	citerator_range_init(tmp, 0, 2, 1);
	citerator_repeat_init(it, tmp, 2);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 1);
	assert(citerator_size(it, &size) == 0);
	assert(size == 6);

	for (int i = 0, k = 0; i <= 2; i++) {
		for (int j = 0; j < 2; j++, k++) {
			assert(citerator_nth(it, k, indexes) == 0);
			assert(indexes[0] == i);
			assert(citerator_pos(it, &ith) == 0);
			assert(ith == k);
			assert(citerator_peek(it, indexes) == 0);
			assert(indexes[0] == i);
			assert(citerator_next(it, indexes) == 0);
			assert(indexes[0] == i);
		}
	}
	assert(citerator_peek(it, indexes) == -ERANGE);
	assert(citerator_next(it, indexes) == -ERANGE);

	assert(citerator_rewind(it) == 0);
	for (int i = 0; i <= 2; i++) {
		for (int j = 0; j < 2; j++) {
			assert(citerator_peek(it, indexes) == 0);
			assert(indexes[0] == i);
			assert(citerator_next(it, indexes) == 0);
			assert(indexes[0] == i);
		}
	}
	assert(citerator_peek(it, indexes) == -ERANGE);
	assert(citerator_next(it, indexes) == -ERANGE);

	assert(citerator_split(it, 2, its) == 0);
	for (int i = 0; i <= 1; i++) {
		for (int j = 0; j < 2; j++) {
			assert(citerator_peek(its[0], indexes) == 0);
			assert(indexes[0] == i);
			assert(citerator_next(its[0], indexes) == 0);
			assert(indexes[0] == i);
		}
	}
	assert(citerator_peek(its[0], indexes) == -ERANGE);
	assert(citerator_next(its[0], indexes) == -ERANGE);
	for (int i = 2; i <= 2; i++) {
		for (int j = 0; j < 2; j++) {
			assert(citerator_peek(its[1], indexes) == 0);
			assert(indexes[0] == i);
			assert(citerator_next(its[1], indexes) == 0);
			assert(indexes[0] == i);
		}
	}
	assert(citerator_peek(its[1], indexes) == -ERANGE);
	assert(citerator_next(its[1], indexes) == -ERANGE);

	citerator_free(its[0]);
	citerator_free(its[1]);
	citerator_free(it);
}

void test_cons_iterator(void)
{
	citerator_t it;
	citerator_t tmp;
	citerator_index_t dim;
	citerator_index_t size;
	enum citerator_type_e type;
	citerator_index_t indexes[4];
	citerator_t its[2];
	citerator_index_t ith;

	it = citerator_alloc(CITERATOR_CONS);
	assert(it != NULL);
	assert(citerator_type(it, &type) == 0);
	assert(type == CITERATOR_CONS);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 0);

	tmp = citerator_alloc(CITERATOR_RANGE);
	citerator_range_init(tmp, 0, 4, 1);
	assert(citerator_cons_init(it, tmp, 3) == 0);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 3);
	assert(citerator_size(it, &size) == 0);
	assert(size == 3);
	for (int j = 0; j < 2; j++) {
		for (int i = 0, k = 0; i < 3; i++, k++) {
			assert(citerator_nth(it, k, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
			assert(citerator_pos(it, &ith) == 0);
			assert(ith == k);
			ith = -1;
			assert(citerator_n(it, indexes, &ith) == 0);
			assert(ith == k);
			assert(citerator_peek(it, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
			assert(citerator_next(it, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
		}
		assert(citerator_peek(it, indexes) == -ERANGE);
		assert(citerator_next(it, indexes) == -ERANGE);
		assert(citerator_rewind(it) == 0);
	}

	assert(citerator_split(it, 2, its) == 0);

	for (int j = 0; j < 2; j++) {
		for (int i = 0, k = 0; i < 2; i++, k++) {
			assert(citerator_nth(its[0], k, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
			assert(citerator_peek(its[0], indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
			assert(citerator_next(its[0], indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
		}
		assert(citerator_peek(its[0], indexes) == -ERANGE);
		assert(citerator_next(its[0], indexes) == -ERANGE);
		assert(citerator_rewind(its[0]) == 0);
	}

	for (int j = 0; j < 2; j++) {
		for (int i = 2, k = 0; i < 3; i++, k++) {
			assert(citerator_nth(its[1], k, indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
			assert(citerator_peek(its[1], indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
			assert(citerator_next(its[1], indexes) == 0);
			assert(indexes[0] == i);
			assert(indexes[1] == i + 1);
			assert(indexes[2] == i + 2);
		}
		assert(citerator_peek(its[1], indexes) == -ERANGE);
		assert(citerator_next(its[1], indexes) == -ERANGE);
		assert(citerator_rewind(its[1]) == 0);
	}
	citerator_free(its[0]);
	citerator_free(its[1]);
	citerator_free(it);

	it = citerator_alloc(CITERATOR_PRODUCT);
	tmp = citerator_alloc(CITERATOR_RANGE);
	citerator_range_init(tmp, 0, 3, 1);
	citerator_product_add(it, tmp);
	tmp = citerator_alloc(CITERATOR_RANGE);
	citerator_range_init(tmp, 1, -1, -1);
	citerator_product_add(it, tmp);

	tmp = citerator_alloc(CITERATOR_CONS);
	assert(citerator_cons_init(tmp, it, 2) == 0);
	assert(citerator_dimension(tmp, &dim) == 0);
	assert(dim == 4);
	assert(citerator_size(tmp, &size) == 0);
	assert(size == 11);
	assert(citerator_peek(tmp, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 1);
	assert(indexes[2] == 0);
	assert(indexes[3] == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(indexes[0] == 3);
	assert(indexes[1] == 1);
	assert(indexes[2] == 3);
	assert(indexes[3] == 0);
	assert(citerator_next(tmp, indexes) == 0);
	assert(indexes[0] == 3);
	assert(indexes[1] == 0);
	assert(indexes[2] == 3);
	assert(indexes[3] == -1);
	assert(citerator_next(tmp, indexes) == -ERANGE);

	citerator_free(tmp);

}

void test_hilbert2d_iterator(void)
{
	citerator_index_t dim;
	citerator_index_t size;
	citerator_index_t indexes[2];
	citerator_index_t indexes2[2];
	enum citerator_type_e type;
	citerator_t it;
	citerator_t its[3];
	citerator_index_t ith;

	it = citerator_alloc(CITERATOR_HILBERT2D);
	assert(it != NULL);
	assert(citerator_type(it, &type) == 0);
	assert(type == CITERATOR_HILBERT2D);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 0);
	assert(citerator_hilbert2d_init(it, 2) == 0);
	assert(citerator_dimension(it, &dim) == 0);
	assert(dim == 2);
	assert(citerator_size(it, &size) == 0);
	assert(size == 16);

	assert(citerator_peek(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 0);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 0);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 1);
	assert(indexes[1] == 0);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 1);
	assert(indexes[1] == 1);
	assert(citerator_pos(it, &ith) == 0);
	assert(ith == 3);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 1);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 2);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 3);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 1);
	assert(indexes[1] == 3);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 1);
	assert(indexes[1] == 2);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 2);
	assert(indexes[1] == 2);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 2);
	assert(indexes[1] == 3);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 3);
	assert(indexes[1] == 3);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 3);
	assert(indexes[1] == 2);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 3);
	assert(indexes[1] == 1);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 2);
	assert(indexes[1] == 1);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 2);
	assert(indexes[1] == 0);
	assert(citerator_next(it, indexes) == 0);
	assert(indexes[0] == 3);
	assert(indexes[1] == 0);
	assert(citerator_peek(it, indexes) == -ERANGE);
	assert(citerator_next(it, indexes) == -ERANGE);
	assert(citerator_rewind(it) == 0);
	assert(citerator_peek(it, indexes) == 0);
	assert(indexes[0] == 0);
	assert(indexes[1] == 0);
	assert(citerator_peek(it, NULL) == 0);
	assert(citerator_skip(it) == 0);
	assert(citerator_peek(it, indexes) == 0);
	assert(indexes[0] == 1);
	assert(indexes[1] == 0);

	assert(citerator_rewind(it) == 0);
	int j = 0;

	while (citerator_next(it, indexes) == 0) {
		assert(citerator_nth(it, j, indexes2) == 0);
		assert(indexes[0] == indexes2[0]);
		assert(indexes[1] == indexes2[1]);
		assert(citerator_n(it, indexes, &ith) == 0);
		assert(j == ith);
		j++;
	}

	assert(citerator_split(it, 3, its) == 0);
	assert(citerator_rewind(it) == 0);
	for (int i = 0; i < 3; i++) {
		while (citerator_next(its[i], indexes) == 0) {
			assert(citerator_next(it, indexes2) == 0);
			assert(indexes[0] == indexes2[0]);
			assert(indexes[1] == indexes2[1]);
		}
	}

	citerator_free(its[0]);
	citerator_free(its[1]);
	citerator_free(its[2]);

	assert(citerator_split(it, 17, its) == -EDOM);

	citerator_free(it);
}

int main(int argc, char *argv[])
{
	test_range_iterator();
	test_product_iterators();
	test_cons_iterator();
	test_repeat_iterator();
	test_hilbert2d_iterator();
	return 0;
}
