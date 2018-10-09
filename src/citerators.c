#include <errno.h>
#include <stdlib.h>
#include <citerators.h>

struct citerator_func_table_s {
	int (*alloc)(citerator_t data);
	void (*free)(citerator_t data);
	int (*copy)(citerator_t dst, const citerator_t src);
	int (*next)(citerator_t data, citerator_index_t *indexes);
	int (*peek)(const citerator_t data, citerator_index_t *indexes);
	int (*size)(const citerator_t data, citerator_index_t *size);
	int (*rewind)(citerator_t data);
	int (*split)(const citerator_t data, citerator_index_t n,
		     citerator_t *results);
	int (*nth)(const citerator_t data, citerator_index_t n,
		   citerator_index_t *indexes);
	int (*n)(const citerator_t data, const citerator_index_t *indexes,
		 citerator_index_t *n);
	int (*pos)(const citerator_t iterator, citerator_index_t *n);
};

struct citerator_s {
	const struct citerator_func_table_s *functions;
	citerator_index_t dimension;
	enum citerator_type_e type;
	void *data;
};

/*--------------------------------------------------------------------*/

struct slice_iterator_s {
	citerator_t src;
	citerator_t indexer;
};

static int slice_iterator_alloc(citerator_t data)
{
	data->data = malloc(sizeof(struct slice_iterator_s));
	if (!data->data)
		return -ENOMEM;
	struct slice_iterator_s *iterator =
	    (struct slice_iterator_s *) data->data;

	iterator->src = NULL;
	iterator->indexer = NULL;
	return 0;
}

static void slice_iterator_free(citerator_t data)
{
	struct slice_iterator_s *iterator =
	    (struct slice_iterator_s *) data->data;

	citerator_free(iterator->src);
	citerator_free(iterator->indexer);
	free(data->data);
}

static int slice_iterator_copy(citerator_t dst, const citerator_t src)
{
	const struct slice_iterator_s *iterator =
	    (const struct slice_iterator_s *) src->data;
	struct slice_iterator_s *result = (struct slice_iterator_s *) dst->data;

	result->src = citerator_dup(iterator->src);
	if (!result->src)
		return -ENOMEM;
	result->indexer = citerator_dup(iterator->indexer);
	if (!result->indexer) {
		citerator_free(iterator->src);
		return -ENOMEM;
	}
	return 0;
}

static int slice_iterator_next(citerator_t data, citerator_index_t *indexes)
{
	struct slice_iterator_s *iterator =
	    (struct slice_iterator_s *) data->data;
	citerator_index_t n;
	int err = citerator_next(iterator->indexer, &n);

	if (err)
		return err;
	return citerator_nth(iterator->src, n, indexes);
}

static int slice_iterator_peek(const citerator_t data,
			       citerator_index_t *indexes)
{
	const struct slice_iterator_s *iterator =
	    (const struct slice_iterator_s *)data->data;
	citerator_index_t n;
	int err = citerator_peek(iterator->indexer, &n);

	if (err)
		return err;
	return citerator_nth(iterator->src, n, indexes);
}

static int slice_iterator_size(const citerator_t data, citerator_index_t *size)
{
	const struct slice_iterator_s *iterator =
	    (const struct slice_iterator_s *)data->data;

	return citerator_size(iterator->indexer, size);
}

static int slice_iterator_rewind(citerator_t data)
{
	struct slice_iterator_s *iterator =
	    (struct slice_iterator_s *) data->data;

	return citerator_rewind(iterator->indexer);
}

static int slice_iterator_nth(const citerator_t data, citerator_index_t n,
			      citerator_index_t *indexes)
{
	const struct slice_iterator_s *iterator =
	    (const struct slice_iterator_s *)data->data;
	citerator_index_t p;
	int err = citerator_nth(iterator->indexer, n, &p);

	if (err)
		return err;
	return citerator_nth(iterator->src, p, indexes);
}

static int slice_iterator_n(const citerator_t data,
			    const citerator_index_t *indexes,
			    citerator_index_t *n)
{
	const struct slice_iterator_s *iterator =
	    (const struct slice_iterator_s *)data->data;
	citerator_index_t inner_n;
	int err = citerator_n(iterator->src, indexes, &inner_n);

	if (err)
		return err;
	return citerator_n(iterator->indexer, &inner_n, n);
}

static int slice_iterator_pos(const citerator_t data, citerator_index_t *n)
{
	const struct slice_iterator_s *iterator =
	    (const struct slice_iterator_s *)data->data;

	return citerator_pos(iterator->indexer, n);
}

static int slice_iterator_split(const citerator_t data, citerator_index_t n,
				citerator_t *results)
{
	const struct slice_iterator_s *iterator =
	    (const struct slice_iterator_s *)data->data;
	int err = citerator_split(iterator->indexer, n, results);

	if (err)
		return err;
	if (!results)
		return 0;
	for (int i = 0; i < n; i++) {
		citerator_t tmp;
		citerator_t tmp2;

		tmp = results[i];
		results[i] = citerator_alloc(CITERATOR_SLICE);
		if (!results[i]) {
			citerator_free(tmp);
			err = -ENOMEM;
			goto error;
		}
		tmp2 = citerator_dup(iterator->src);
		if (!tmp2) {
			citerator_free(tmp);
			err = -ENOMEM;
			goto error;
		}
		err = citerator_slice_init(results[i], tmp, tmp2);
		if (err) {
			citerator_free(tmp);
			citerator_free(tmp2);
			goto error;
		}
	}
	return 0;
error:
	for (int i = 0; i < n; i++)
		citerator_free(results[i]);
	return err;
}

static const struct citerator_func_table_s citerator_slice_func_table = {
	slice_iterator_alloc,
	slice_iterator_free,
	slice_iterator_copy,
	slice_iterator_next,
	slice_iterator_peek,
	slice_iterator_size,
	slice_iterator_rewind,
	slice_iterator_split,
	slice_iterator_nth,
	slice_iterator_n,
	slice_iterator_pos
};

int citerator_slice_init(citerator_t iterator, citerator_t src,
			 citerator_t indexer)
{
	if (!iterator || iterator->type != CITERATOR_SLICE || !src || !indexer
	    || indexer->dimension != 1)
		return -EINVAL;
	struct slice_iterator_s *it =
	    (struct slice_iterator_s *) iterator->data;
	citerator_index_t size_src;
	citerator_index_t size_indexer;
	int err = citerator_size(src, &size_src);

	if (err)
		return err;
	err = citerator_size(indexer, &size_indexer);
	if (err)
		return err;
	if (size_indexer > size_src)
		return -EDOM;
	it->src = src;
	it->indexer = indexer;
	iterator->dimension = src->dimension;
	return 0;
}

/*--------------------------------------------------------------------*/

struct product_iterator_s {
	citerator_index_t count;
	citerator_t *iterators;
};

static int product_iterator_alloc(citerator_t data)
{
	data->data = malloc(sizeof(struct product_iterator_s));
	if (!data->data)
		return -ENOMEM;
	struct product_iterator_s *iterator =
	    (struct product_iterator_s *) data->data;

	iterator->count = 0;
	iterator->iterators = NULL;
	return 0;
}

static void product_iterator_free(citerator_t data)
{
	struct product_iterator_s *iterator =
	    (struct product_iterator_s *) data->data;
	if (iterator->iterators) {
		for (int i = 0; i < iterator->count; i++)
			citerator_free(iterator->iterators[i]);
		free(iterator->iterators);
	}
	free(data->data);
}

static int product_iterator_copy(citerator_t dst, const citerator_t src)
{
	const struct product_iterator_s *iterator =
	    (const struct product_iterator_s *)src->data;
	struct product_iterator_s *result =
	    (struct product_iterator_s *) dst->data;

	result->iterators =
	    (citerator_t *) malloc(iterator->count * sizeof(citerator_t));
	if (!result->iterators)
		return -ENOMEM;
	int i;

	for (i = 0; i < iterator->count; i++) {
		result->iterators[i] = citerator_dup(iterator->iterators[i]);
		if (!result->iterators[i]) {
			i--;
			goto error;
		}
	}
	result->count = iterator->count;
	return 0;
error:
	while (i >= 0) {
		free(result->iterators[i]);
		i--;
	}
	free(result->iterators);
	return -ENOMEM;
}

static int product_iterator_rewind(citerator_t data)
{
	struct product_iterator_s *iterator =
	    (struct product_iterator_s *) data->data;

	for (int i = 0; i < iterator->count; i++) {
		int err = citerator_rewind(iterator->iterators[i]);

		if (err)
			return err;
	}
	return 0;
}

static int product_iterator_size(const citerator_t data,
				 citerator_index_t *size)
{
	const struct product_iterator_s *iterator =
	    (const struct product_iterator_s *) data->data;
	citerator_index_t tmp_size = 0;

	if (!size)
		return -EINVAL;
	if (iterator->count == 0)
		*size = 0;
	else {
		*size = 1;
		for (int i = 0; i < iterator->count; i++) {
			int err =
			    citerator_size(iterator->iterators[i], &tmp_size);

			if (err) {
				*size = 0;
				return err;
			}
			*size *= tmp_size;
		}
	}
	return 0;
}

static int product_iterator_nth(const citerator_t data, citerator_index_t n,
				citerator_index_t *indexes)
{
	int size;
	int err = product_iterator_size(data, &size);

	if (err)
		return err;
	if (n < 0 || n >= size)
		return -EDOM;
	const struct product_iterator_s *iterator =
	    (const struct product_iterator_s *) data->data;

	if (indexes) {
		citerator_index_t subsize = 0;
		citerator_index_t offset = data->dimension;

		for (int i = iterator->count - 1; i >= 0; i--) {
			offset -= iterator->iterators[i]->dimension;
			err = citerator_size(iterator->iterators[i], &subsize);
			if (err)
				return err;
			err =
			    citerator_nth(iterator->iterators[i], n % subsize,
					  indexes + offset);
			if (err)
				return err;
			n /= subsize;
		}
	}
	return 0;
}

static int product_iterator_n(const citerator_t data,
			      const citerator_index_t *indexes,
			      citerator_index_t *n)
{
	const struct product_iterator_s *iterator =
	    (const struct product_iterator_s *) data->data;

	if (iterator->count == 0)
		return -EINVAL;
	citerator_index_t offset = 0;
	citerator_index_t product = 0;
	citerator_index_t inner_n;
	citerator_index_t subsize;

	for (int i = 0; i < iterator->count; i++) {
		int err =
		    citerator_n(iterator->iterators[i], indexes + offset,
				&inner_n);
		if (err)
			return err;
		err = citerator_size(iterator->iterators[i], &subsize);
		if (err)
			return err;
		product *= subsize;
		product += inner_n;
		offset += iterator->iterators[i]->dimension;
	}
	if (n)
		*n = product;
	return 0;
}

static int product_iterator_pos(const citerator_t data, citerator_index_t *n)
{
	const struct product_iterator_s *iterator =
	    (const struct product_iterator_s *) data->data;

	if (iterator->count == 0)
		return -EINVAL;
	citerator_index_t product = 0;
	citerator_index_t inner_n;
	citerator_index_t subsize;

	for (int i = 0; i < iterator->count; i++) {
		int err = citerator_pos(iterator->iterators[i], &inner_n);

		if (err)
			return err;
		err = citerator_size(iterator->iterators[i], &subsize);
		if (err)
			return err;
		product *= subsize;
		product += inner_n;
	}
	if (n)
		*n = product;
	return 0;
}

static inline int product_iterator_peeknext_helper(citerator_t data,
						   citerator_index_t *indexes,
						   int next)
{
	struct product_iterator_s *iterator =
	    (struct product_iterator_s *) data->data;
	int err;
	int looped;
	int i;
	citerator_index_t *next_indexes;
	citerator_index_t offset = data->dimension;

	if (iterator->count == 0)
		return -EINVAL;
	looped = next;
	for (i = iterator->count - 1; i > 0; i--) {
		if (indexes) {
			offset -= iterator->iterators[i]->dimension;
			next_indexes = indexes + offset;
		} else
			next_indexes = NULL;
		if (looped)
			err =
			    citerator_cyclic_next(iterator->iterators[i],
						  next_indexes, &looped);
		else
			err =
			    citerator_peek(iterator->iterators[i],
					   next_indexes);
		if (err)
			return err;
	}
	if (indexes) {
		offset -= iterator->iterators[i]->dimension;
		next_indexes = indexes + offset;
	} else
		next_indexes = NULL;
	if (looped)
		err = citerator_next(iterator->iterators[0], next_indexes);
	else
		err = citerator_peek(iterator->iterators[0], next_indexes);
	if (err)
		return err;
	return 0;
}

static int product_iterator_peek(const citerator_t data,
				 citerator_index_t *indexes)
{
	return product_iterator_peeknext_helper(data, indexes, 0);
}

static int product_iterator_next(citerator_t data, citerator_index_t *indexes)
{
	return product_iterator_peeknext_helper(data, indexes, 1);
}

static int product_iterator_split(const citerator_t data, citerator_index_t n,
				  citerator_t *results)
{
	int size;
	int err = product_iterator_size(data, &size);

	if (err)
		return err;
	if (size < n)
		return -EDOM;
	if (!results)
		return 0;
	citerator_t range = citerator_alloc(CITERATOR_RANGE);

	if (!range)
		return -ENOMEM;
	err = citerator_range_init(range, 0, size - 1, 1);
	if (err)
		goto error1;
	err = citerator_split(range, n, results);
	if (err)
		goto error1;
	for (int i = 0; i < n; i++) {
		citerator_t tmp, tmp2;

		tmp = citerator_dup(data);
		if (!tmp)
			goto error2;
		tmp2 = results[i];
		results[i] = citerator_alloc(CITERATOR_SLICE);
		if (!results[i]) {
			citerator_free(tmp2);
			goto error2;
		}
		err = citerator_slice_init(results[i], tmp, tmp2);
		if (err) {
			citerator_free(tmp2);
			goto error2;
		}
	}
	citerator_free(range);
	return 0;
error2:
	for (int i = 0; i < n; i++)
		citerator_free(results[i]);
error1:
	citerator_free(range);
	return err;
}

int citerator_product_count(const citerator_t iterator,
			    citerator_index_t *count)
{
	if (!iterator || iterator->type != CITERATOR_PRODUCT || !count)
		return -EINVAL;
	*count = ((struct product_iterator_s *) iterator->data)->count;
	return 0;
}

int citerator_product_split_dim(const citerator_t iterator,
				citerator_index_t dim, citerator_index_t n,
				citerator_t *results)
{
	if (!iterator || iterator->type != CITERATOR_PRODUCT)
		return -EINVAL;
	if (n <= 0)
		return -EDOM;
	int count;
	int err = citerator_product_count(iterator, &count);

	if (err)
		return err;
	if (dim >= count)
		return -EDOM;
	struct product_iterator_s *product_iterator =
	    (struct product_iterator_s *) iterator->data;

	err = citerator_split(product_iterator->iterators[dim], n, results);
	if (err)
		return err;
	if (!results)
		return 0;
	for (int i = 0; i < n; i++) {
		citerator_t tmp = results[i];

		results[i] = citerator_dup(iterator);
		if (!tmp) {
			citerator_free(tmp);
			err = -ENOMEM;
			goto error;
		}
		struct product_iterator_s *new_product_iterator =
		    (struct product_iterator_s *) results[i]->data;
		citerator_free(new_product_iterator->iterators[dim]);
		new_product_iterator->iterators[dim] = tmp;
	}
	return 0;
error:
	for (int i = 0; i < n; i++)
		citerator_free(results[i]);
	return err;
}

int citerator_product_add_copy(citerator_t iterator, citerator_t added_iterator)
{
	int err = 0;
	citerator_t copy = citerator_dup(added_iterator);

	if (!copy)
		return -EINVAL;
	err = citerator_product_add(iterator, copy);
	if (err) {
		citerator_free(added_iterator);
		return err;
	}
	return 0;
}

int citerator_product_add(citerator_t iterator, citerator_t added_iterator)
{
	if (!iterator || iterator->type != CITERATOR_PRODUCT || !iterator->data
	    || !added_iterator)
		return -EINVAL;

	struct product_iterator_s *product_iterator =
	    (struct product_iterator_s *) iterator->data;
	int mew_count = product_iterator->count + 1;

	citerator_t *new_its =
	    (citerator_t *) realloc(product_iterator->iterators,
				    mew_count * sizeof(citerator_t));
	if (!new_its)
		return -ENOMEM;
	product_iterator->iterators = new_its;
	product_iterator->iterators[product_iterator->count] = added_iterator;
	product_iterator->count = mew_count;
	iterator->dimension += added_iterator->dimension;
	return 0;
}

static const struct citerator_func_table_s citerator_product_func_table = {
	product_iterator_alloc,
	product_iterator_free,
	product_iterator_copy,
	product_iterator_next,
	product_iterator_peek,
	product_iterator_size,
	product_iterator_rewind,
	product_iterator_split,
	product_iterator_nth,
	product_iterator_n,
	product_iterator_pos
};

/*--------------------------------------------------------------------*/

struct circular_fifo_s {
	citerator_index_t length;
	citerator_index_t start;
	citerator_index_t end;
	citerator_index_t size;
	citerator_index_t *buffer;
};

static void circular_fifo_add(struct circular_fifo_s *fifo,
			      citerator_index_t elem)
{
	if (fifo->size == fifo->length) {
		fifo->start = (fifo->start + 1) % fifo->length;
		fifo->end = (fifo->end + 1) % fifo->length;
	} else {
		fifo->end = (fifo->end + 1) % fifo->length;
		fifo->size++;
	}
	fifo->buffer[fifo->end] = elem;
}

static void circular_fifo_dump(const struct circular_fifo_s *fifo,
			       citerator_index_t *vals)
{
	citerator_index_t i;
	citerator_index_t j;

	for (i = 0, j = fifo->start; i < fifo->size; i++) {
		vals[i] = fifo->buffer[j];
		j = (j + 1) % fifo->length;
	}
}

struct cons_iterator_s {
	citerator_t iterator;
	citerator_index_t n;
	struct circular_fifo_s fifo;
};

static int cons_iterator_alloc(citerator_t data)
{
	data->data = malloc(sizeof(struct cons_iterator_s));
	if (!data->data)
		return -ENOMEM;
	struct cons_iterator_s *iterator =
	    (struct cons_iterator_s *) data->data;

	iterator->iterator = NULL;
	iterator->n = 0;
	iterator->fifo.length = 0;
	iterator->fifo.start = 0;
	iterator->fifo.end = -1;
	iterator->fifo.size = 0;
	iterator->fifo.buffer = NULL;
	return 0;
}

static void cons_iterator_free(citerator_t data)
{
	struct cons_iterator_s *iterator =
	    (struct cons_iterator_s *) data->data;

	citerator_free(iterator->iterator);
	free(iterator->fifo.buffer);
	free(data->data);
}

static int cons_iterator_copy(citerator_t ddst, const citerator_t dsrc)
{
	struct cons_iterator_s *dst = (struct cons_iterator_s *) ddst->data;
	const struct cons_iterator_s *src =
	    (const struct cons_iterator_s *)dsrc->data;
	citerator_t copy = citerator_dup(src->iterator);

	if (!copy)
		return -EINVAL;
	dst->iterator = copy;
	dst->n = src->n;
	dst->fifo.length = src->fifo.length;
	dst->fifo.start = src->fifo.start;
	dst->fifo.end = src->fifo.end;
	dst->fifo.size = src->fifo.size;
	dst->fifo.buffer =
	    (citerator_index_t *) malloc(src->fifo.length *
					 sizeof(citerator_index_t));
	if (!dst->fifo.buffer) {
		citerator_free(copy);
		return -ENOMEM;
	}
	for (int i = 0; i < dst->fifo.length; i++)
		dst->fifo.buffer[i] = src->fifo.buffer[i];
	return 0;
}

static int cons_iterator_size(const citerator_t data, citerator_index_t *size)
{
	const struct cons_iterator_s *iterator =
	    (const struct cons_iterator_s *)data->data;
	int tmp_size = 0;
	int err = citerator_size(iterator->iterator, &tmp_size);

	if (err)
		return err;
	*size = tmp_size - (iterator->n - 1);
	return 0;
}

static int cons_iterator_split(const citerator_t data, citerator_index_t n,
			       citerator_t *results)
{
	int size;
	int err = cons_iterator_size(data, &size);

	if (err)
		return err;
	if (size < n)
		return -EDOM;
	if (!results)
		return 0;
	citerator_t range = citerator_alloc(CITERATOR_RANGE);

	if (!range)
		return -ENOMEM;
	err = citerator_range_init(range, 0, size - 1, 1);
	if (err)
		goto error1;
	err = citerator_split(range, n, results);
	if (err)
		goto error1;
	int i;

	for (i = 0; i < n; i++) {
		citerator_t tmp, tmp2;

		tmp = citerator_dup(data);
		if (!tmp)
			goto error2;
		tmp2 = results[i];
		results[i] = citerator_alloc(CITERATOR_SLICE);
		if (!results[i]) {
			citerator_free(tmp2);
			goto error2;
		}
		err = citerator_slice_init(results[i], tmp, tmp2);
		if (err) {
			citerator_free(tmp2);
			goto error2;
		}
	}
	citerator_free(range);
	return 0;
error2:
	for (; i >= 0; i--)
		citerator_free(results[i]);
error1:
	citerator_free(range);
	return err;
}

static int cons_iterator_nth(const citerator_t data, citerator_index_t n,
			     citerator_index_t *indexes)
{
	int size;
	int err = cons_iterator_size(data, &size);

	if (err)
		return err;
	if (n < 0 || n >= size)
		return -EDOM;
	const struct cons_iterator_s *iterator =
	    (const struct cons_iterator_s *) data->data;
	int dim = iterator->iterator->dimension;

	if (indexes) {
		for (int i = 0; i < iterator->n; i++) {
			err =
			    citerator_nth(iterator->iterator, n + i,
					  indexes + dim * i);
			if (err)
				return err;
		}
	}
	return 0;
}

static int cons_iterator_n(const citerator_t data,
			   const citerator_index_t *indexes,
			   citerator_index_t *n)
{
	const struct cons_iterator_s *iterator =
	    (const struct cons_iterator_s *) data->data;
	citerator_index_t inner_n, inner_n_tmp;
	int err = citerator_n(iterator->iterator, indexes, &inner_n);

	if (err)
		return err;
	int dim = iterator->iterator->dimension;

	for (int i = 1; i < iterator->n; i++) {
		err =
		    citerator_n(iterator->iterator, indexes + dim * i,
				&inner_n_tmp);
		if (err)
			return err;
		if (inner_n_tmp != inner_n + 1)
			return -EINVAL;
		inner_n = inner_n_tmp;
	}
	if (n)
		*n = inner_n - (iterator->n - 1);
	return 0;
}

static int cons_iterator_pos(const citerator_t data, citerator_index_t *n)
{
	citerator_index_t inner_n;
	const struct cons_iterator_s *iterator =
	    (const struct cons_iterator_s *) data->data;
	int err = citerator_pos(iterator->iterator, &inner_n);

	if (err)
		return err;
	if (n)
		*n = inner_n - (iterator->n - 1);
	return 0;
}

static int cons_iterator_peek(const citerator_t data,
			      citerator_index_t *indexes)
{
	const struct cons_iterator_s *iterator =
	    (const struct cons_iterator_s *) data->data;
	int err;
	int dim = iterator->iterator->dimension;
	int n = iterator->n;

	if (indexes) {
		circular_fifo_dump(&iterator->fifo, indexes);
		err =
		    citerator_peek(iterator->iterator, indexes + dim * (n - 1));
	} else
		err = citerator_peek(iterator->iterator, NULL);
	if (err)
		return err;
	return 0;
}

static int cons_iterator_next(citerator_t data, citerator_index_t *indexes)
{
	struct cons_iterator_s *iterator =
	    (struct cons_iterator_s *) data->data;
	int err;
	int dim = iterator->iterator->dimension;
	int n = iterator->n;

	if (indexes) {
		circular_fifo_dump(&iterator->fifo, indexes);
		err =
		    citerator_next(iterator->iterator, indexes + dim * (n - 1));
	} else
		err = citerator_next(iterator->iterator, NULL);
	if (err)
		return err;
	if (indexes)
		for (int i = dim * (n - 1); i < dim * n; i++)
			circular_fifo_add(&iterator->fifo, indexes[i]);
	return 0;
}

static int cons_iterator_rewind(citerator_t data)
{
	struct cons_iterator_s *iterator =
	    (struct cons_iterator_s *) data->data;
	int err = citerator_rewind(iterator->iterator);

	if (err)
		return err;
	iterator->fifo.start = 0;
	iterator->fifo.end = -1;
	iterator->fifo.size = 0;

	for (int i = 0; i < iterator->n - 1; i++) {
		int err;

		err =
		    citerator_next(iterator->iterator,
				   iterator->fifo.buffer +
				   iterator->iterator->dimension * i);
		if (err)
			return err;
		iterator->fifo.size += iterator->iterator->dimension;
		iterator->fifo.end += iterator->iterator->dimension;
	}
	return 0;
}

int citerator_cons_init(citerator_t iterator, citerator_t src,
			citerator_index_t n)
{
	int src_size;
	int err;

	if (!iterator || iterator->type != CITERATOR_CONS || !src || n <= 0)
		return -EINVAL;
	err = citerator_size(src, &src_size);
	if (err)
		return err;
	if (src_size < n)
		return -EINVAL;
	struct cons_iterator_s *cons_iterator =
	    (struct cons_iterator_s *) iterator->data;

	free(cons_iterator->fifo.buffer);
	citerator_free(cons_iterator->iterator);
	iterator->dimension = n * src->dimension;
	cons_iterator->iterator = src;
	cons_iterator->n = n;
	cons_iterator->fifo.length = src->dimension * (n - 1);
	cons_iterator->fifo.buffer =
	    (citerator_index_t *) malloc(cons_iterator->fifo.length *
					 sizeof(citerator_index_t));
	if (!cons_iterator->fifo.buffer)
		return -ENOMEM;
	err = cons_iterator_rewind(iterator);
	if (err) {
		free(cons_iterator->fifo.buffer);
		return err;
	}
	return 0;
}

static const struct citerator_func_table_s citerator_cons_func_table = {
	cons_iterator_alloc,
	cons_iterator_free,
	cons_iterator_copy,
	cons_iterator_next,
	cons_iterator_peek,
	cons_iterator_size,
	cons_iterator_rewind,
	cons_iterator_split,
	cons_iterator_nth,
	cons_iterator_n,
	cons_iterator_pos
};

/*--------------------------------------------------------------------*/

struct repeat_iterator_s {
	citerator_t iterator;
	citerator_index_t n;
	citerator_index_t counter;
};

static int repeat_iterator_alloc(citerator_t data)
{
	data->data = malloc(sizeof(struct repeat_iterator_s));
	if (!data->data)
		return -ENOMEM;
	struct repeat_iterator_s *iterator =
	    (struct repeat_iterator_s *) data->data;

	iterator->iterator = NULL;
	iterator->n = 0;
	iterator->counter = 0;
	return 0;
}

static void repeat_iterator_free(citerator_t data)
{
	struct repeat_iterator_s *iterator =
	    (struct repeat_iterator_s *) data->data;

	citerator_free(iterator->iterator);
	free(data->data);
}

static int repeat_iterator_copy(citerator_t ddst, const citerator_t dsrc)
{
	struct repeat_iterator_s *dst = (struct repeat_iterator_s *) ddst->data;
	const struct repeat_iterator_s *src =
	    (const struct repeat_iterator_s *)dsrc->data;
	citerator_t copy = citerator_dup(src->iterator);

	if (!copy)
		return -EINVAL;
	dst->iterator = copy;
	dst->n = src->n;
	dst->counter = src->counter;
	return 0;
}

static int repeat_iterator_peek(const citerator_t data,
				citerator_index_t *indexes)
{
	const struct repeat_iterator_s *iterator =
	    (const struct repeat_iterator_s *) data->data;

	return citerator_peek(iterator->iterator, indexes);
}

static int repeat_iterator_next(citerator_t data, citerator_index_t *indexes)
{
	struct repeat_iterator_s *iterator =
	    (struct repeat_iterator_s *) data->data;

	iterator->counter++;
	if (iterator->counter < iterator->n)
		return citerator_peek(iterator->iterator, indexes);
	iterator->counter = 0;
	return citerator_next(iterator->iterator, indexes);
}

static int repeat_iterator_size(const citerator_t data,
				citerator_index_t *size)
{
	const struct repeat_iterator_s *iterator =
	    (const struct repeat_iterator_s *) data->data;
	int err = citerator_size(iterator->iterator, size);

	if (err)
		return err;
	*size *= iterator->n;
	return 0;
}

static int repeat_iterator_rewind(citerator_t data)
{
	struct repeat_iterator_s *iterator =
	    (struct repeat_iterator_s *) data->data;

	iterator->counter = 0;
	return citerator_rewind(iterator->iterator);
}

static int repeat_iterator_nth(const citerator_t data, citerator_index_t n,
			       citerator_index_t *val)
{
	int size;
	int err = repeat_iterator_size(data, &size);

	if (err)
		return err;
	if (n < 0 || n >= size)
		return -EDOM;
	const struct repeat_iterator_s *iterator =
	    (const struct repeat_iterator_s *) data->data;

	return citerator_nth(iterator->iterator, n / iterator->n, val);
}

static int repeat_iterator_pos(const citerator_t data, citerator_index_t *n)
{
	citerator_index_t inner_n;
	const struct repeat_iterator_s *iterator =
	    (const struct repeat_iterator_s *) data->data;
	int err = citerator_pos(iterator->iterator, &inner_n);

	if (err)
		return err;
	if (n)
		*n = inner_n * iterator->n + iterator->counter;
	return 0;
}

static int repeat_iterator_split(const citerator_t data, citerator_index_t n,
				 citerator_t *results)
{
	const struct repeat_iterator_s *iterator =
	    (const struct repeat_iterator_s *) data->data;
	int err = citerator_split(iterator->iterator, n, results);

	if (err)
		return err;
	if (!results)
		return 0;
	for (int i = 0; i < n; i++) {
		citerator_t tmp;

		tmp = results[i];
		results[i] = citerator_alloc(CITERATOR_REPEAT);
		if (!results[i]) {
			citerator_free(tmp);
			err = -ENOMEM;
			goto error;
		}
		err = citerator_repeat_init(results[i], tmp, iterator->n);
		if (err)
			goto error;
	}
	return 0;
error:
	for (int i = 0; i < n; i++)
		citerator_free(results[i]);
	return err;
}

static const struct citerator_func_table_s citerator_repeat_func_table = {
	repeat_iterator_alloc,
	repeat_iterator_free,
	repeat_iterator_copy,
	repeat_iterator_next,
	repeat_iterator_peek,
	repeat_iterator_size,
	repeat_iterator_rewind,
	repeat_iterator_split,
	repeat_iterator_nth,
	NULL,
	repeat_iterator_pos
};

int citerator_repeat_init(citerator_t iterator, citerator_t src,
			  citerator_index_t n)
{
	if (!iterator || iterator->type != CITERATOR_REPEAT || !src || n <= 0)
		return -EINVAL;
	struct repeat_iterator_s *repeat_iterator =
	    (struct repeat_iterator_s *) iterator->data;
	citerator_free(repeat_iterator->iterator);
	iterator->dimension = src->dimension;
	repeat_iterator->iterator = src;
	repeat_iterator->n = n;
	repeat_iterator->counter = 0;
	return 0;
}

/*--------------------------------------------------------------------*/

struct hilbert2d_iterator_s {
	citerator_index_t n;
	citerator_t range_iterator;
};

/* Helper functions from: https://en.wikipedia.org/wiki/Hilbert_curve */

//rotate/flip a quadrant appropriately
static void rot(citerator_index_t n, citerator_index_t *x,
		citerator_index_t *y, citerator_index_t rx,
		citerator_index_t ry)
{
	if (ry == 0) {
		if (rx == 1) {
			*x = n - 1 - *x;
			*y = n - 1 - *y;
		}
		//Swap x and y
		citerator_index_t t = *x;

		*x = *y;
		*y = t;
	}
}

//convert (x,y) to d
static citerator_index_t xy2d(citerator_index_t n, citerator_index_t x,
			      citerator_index_t y)
{
	citerator_index_t rx, ry, s, d = 0;

	for (s = n / 2; s > 0; s /= 2) {
		rx = (x & s) > 0;
		ry = (y & s) > 0;
		d += s * s * ((3 * rx) ^ ry);
		rot(s, &x, &y, rx, ry);
	}
	return d;
}

//convert d to (x,y)
static void d2xy(citerator_index_t n, citerator_index_t d,
		 citerator_index_t *x, citerator_index_t *y)
{
	citerator_index_t rx, ry, s, t = d;

	*x = *y = 0;
	for (s = 1; s < n; s *= 2) {
		rx = 1 & (t / 2);
		ry = 1 & (t ^ rx);
		rot(s, x, y, rx, ry);
		*x += s * rx;
		*y += s * ry;
		t /= 4;
	}
}

/* End helper functions */

static int hilbert2d_iterator_alloc(citerator_t data)
{
	data->data = malloc(sizeof(struct hilbert2d_iterator_s));
	if (!data->data)
		return -ENOMEM;
	struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;

	iterator->n = 0;
	iterator->range_iterator = NULL;
	return 0;
}

static void hilbert2d_iterator_free(citerator_t data)
{
	struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;

	citerator_free(iterator->range_iterator);
	free(data->data);
}

static int hilbert2d_iterator_copy(citerator_t ddst, const citerator_t dsrc)
{
	struct hilbert2d_iterator_s *dst =
	    (struct hilbert2d_iterator_s *) ddst->data;
	const struct hilbert2d_iterator_s *src =
	    (const struct hilbert2d_iterator_s *)dsrc->data;
	citerator_t copy = citerator_dup(src->range_iterator);

	if (!copy)
		return -EINVAL;
	dst->range_iterator = copy;
	dst->n = src->n;
	return 0;
}

static int hilbert2d_iterator_rewind(citerator_t data)
{
	struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;

	return citerator_rewind(iterator->range_iterator);
}

static int hilbert2d_iterator_peek(const citerator_t data,
				   citerator_index_t *val)
{
	struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;
	citerator_index_t d;
	int err;

	err = citerator_peek(iterator->range_iterator, &d);
	if (err)
		return err;
	if (val)
		d2xy(iterator->n, d, val, val + 1);
	return 0;
}

static int hilbert2d_iterator_next(citerator_t data, citerator_index_t *val)
{
	struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;
	citerator_index_t d;
	int err = citerator_next(iterator->range_iterator, &d);

	if (err)
		return err;
	if (val)
		d2xy(iterator->n, d, val, val + 1);
	return 0;
}

static int hilbert2d_iterator_size(const citerator_t data,
				   citerator_index_t *size)
{
	const struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;
	return citerator_size(iterator->range_iterator, size);
}

static int hilbert2d_iterator_nth(const citerator_t data, citerator_index_t n,
				  citerator_index_t *val)
{
	citerator_index_t d;
	struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;
	int err = citerator_nth(iterator->range_iterator, n, &d);

	if (err)
		return err;
	if (val)
		d2xy(iterator->n, d, val, val + 1);
	return 0;
}

static int hilbert2d_iterator_n(const citerator_t data,
				const citerator_index_t *indexes,
				citerator_index_t *n)
{
	struct hilbert2d_iterator_s *it =
	    (struct hilbert2d_iterator_s *) data->data;

	if (indexes[0] < 0 || indexes[0] >= it->n || indexes[1] < 0
	    || indexes[1] >= it->n)
		return -EINVAL;
	citerator_index_t d = xy2d(it->n, indexes[0], indexes[1]);

	return citerator_n(it->range_iterator, &d, n);
}

static int hilbert2d_iterator_pos(const citerator_t data, citerator_index_t *n)
{
	struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;

	return citerator_pos(iterator->range_iterator, n);
}

static int hilbert2d_iterator_split(const citerator_t data, citerator_index_t n,
				    citerator_t *results)
{
	const struct hilbert2d_iterator_s *iterator =
	    (struct hilbert2d_iterator_s *) data->data;
	int err = citerator_split(iterator->range_iterator, n, results);

	if (err)
		return err;
	if (!results)
		return 0;
	for (int i = 0; i < n; i++) {
		citerator_t tmp;

		tmp = results[i];
		results[i] = citerator_alloc(CITERATOR_HILBERT2D);
		if (!results[i]) {
			citerator_free(tmp);
			err = -ENOMEM;
			goto error;
		}
		results[i]->dimension = 2;
		struct hilbert2d_iterator_s *it =
		    (struct hilbert2d_iterator_s *) results[i]->data;
		it->n = iterator->n;
		it->range_iterator = tmp;
	}
	return 0;
error:
	for (int i = 0; i < n; i++)
		citerator_free(results[i]);
	return err;
}

int citerator_hilbert2d_init(citerator_t iterator, citerator_index_t order)
{
	struct hilbert2d_iterator_s *hilbert2d_iterator;

	if (!iterator || iterator->type != CITERATOR_HILBERT2D || order <= 0)
		return -EINVAL;
	iterator->dimension = 2;
	hilbert2d_iterator = (struct hilbert2d_iterator_s *) iterator->data;
	hilbert2d_iterator->range_iterator = citerator_alloc(CITERATOR_RANGE);
	if (!hilbert2d_iterator->range_iterator)
		return -ENOMEM;
	int n = 1 << order;
	int err =
	    citerator_range_init(hilbert2d_iterator->range_iterator, 0,
				 n * n - 1, 1);

	if (err)
		return err;
	hilbert2d_iterator->n = n;
	return 0;
}

static const struct citerator_func_table_s citerator_hilbert2d_func_table = {
	hilbert2d_iterator_alloc,
	hilbert2d_iterator_free,
	hilbert2d_iterator_copy,
	hilbert2d_iterator_next,
	hilbert2d_iterator_peek,
	hilbert2d_iterator_size,
	hilbert2d_iterator_rewind,
	hilbert2d_iterator_split,
	hilbert2d_iterator_nth,
	hilbert2d_iterator_n,
	hilbert2d_iterator_pos
};

/*--------------------------------------------------------------------*/

struct range_iterator_s {
	citerator_index_t v;
	citerator_index_t first;
	citerator_index_t last;
	citerator_index_t step;
};

static int range_iterator_alloc(citerator_t data)
{
	data->data = malloc(sizeof(struct range_iterator_s));
	if (!data->data)
		return -ENOMEM;
	struct range_iterator_s *iterator =
	    (struct range_iterator_s *) data->data;

	iterator->v = 0;
	iterator->first = 0;
	iterator->last = -1;
	iterator->step = 1;
	return 0;
}

static void range_iterator_free(citerator_t data)
{
	free(data->data);
}

static int range_iterator_copy(citerator_t ddst, const citerator_t dsrc)
{
	struct range_iterator_s *dst = (struct range_iterator_s *) ddst->data;
	const struct range_iterator_s *src =
	    (const struct range_iterator_s *)dsrc->data;

	dst->v = src->v;
	dst->first = src->first;
	dst->last = src->last;
	dst->step = src->step;
	return 0;
}

static int range_iterator_rewind(citerator_t data)
{
	struct range_iterator_s *iterator =
	    (struct range_iterator_s *) data->data;

	iterator->v = iterator->first;
	return 0;
}

static int range_iterator_peek(const citerator_t data, citerator_index_t *val)
{
	struct range_iterator_s *iterator =
	    (struct range_iterator_s *) data->data;

	if (iterator->step < 0) {
		if (iterator->v < iterator->last)
			return -ERANGE;
	} else if (iterator->step > 0) {
		if (iterator->v > iterator->last)
			return -ERANGE;
	} else
		return -EINVAL;
	if (val)
		*val = iterator->v;
	return 0;
}

static int range_iterator_next(citerator_t data, citerator_index_t *val)
{
	struct range_iterator_s *iterator =
	    (struct range_iterator_s *) data->data;
	int err = range_iterator_peek(data, val);

	if (err)
		return err;
	iterator->v += iterator->step;
	return 0;
}

static int range_iterator_size(const citerator_t data, citerator_index_t *size)
{
	const struct range_iterator_s *it =
	    (struct range_iterator_s *) data->data;

	if (it->step < 0)
		if (it->first < it->last)
			*size = 0;
		else
			*size = 1 + (it->first - it->last) / (-it->step);
	else if (it->step > 0)
		if (it->first > it->last)
			*size = 0;
		else
			*size = 1 + (it->last - it->first) / (it->step);
	else
		return -EINVAL;
	return 0;
}

static int range_iterator_nth(const citerator_t data, citerator_index_t n,
			      citerator_index_t *val)
{
	int size;
	int err = range_iterator_size(data, &size);

	if (err)
		return err;
	if (n < 0 || n >= size)
		return -EDOM;
	if (val) {
		struct range_iterator_s *iterator =
		    (struct range_iterator_s *) data->data;
		*val = iterator->first + n * iterator->step;
	}
	return 0;
}

static int range_iterator_n(const citerator_t data,
			    const citerator_index_t *val,
			    citerator_index_t *n)
{
	int size;
	int err = range_iterator_size(data, &size);

	if (err)
		return err;
	const struct range_iterator_s *it =
	    (struct range_iterator_s *) data->data;
	citerator_index_t pos = (*val - it->first) / it->step;

	if (pos < 0 || pos >= size || it->first + pos * it->step != *val)
		return -EINVAL;
	if (n)
		*n = pos;
	return 0;
}

static int range_iterator_pos(const citerator_t data, citerator_index_t *n)
{
	citerator_index_t val;
	int err = range_iterator_peek(data, &val);

	if (err)
		return err;
	const struct range_iterator_s *it =
	    (struct range_iterator_s *) data->data;

	if (n)
		*n = (val - it->first) / it->step;
	return 0;
}

static int range_iterator_split(const citerator_t data, citerator_index_t n,
				citerator_t *results)
{
	const struct range_iterator_s *iterator =
	    (struct range_iterator_s *) data->data;
	citerator_index_t size;
	int err = range_iterator_size(data, &size);

	if (err)
		return err;
	int block_size = size / n;

	if (block_size <= 0)
		return -EDOM;
	if (!results)
		return 0;
	citerator_index_t new_last = iterator->last;
	citerator_index_t new_first;
	int i;

	for (i = n - 1; i >= 0; i--) {
		block_size = size / (i + 1);
		results[i] = citerator_alloc(CITERATOR_RANGE);
		if (!results[i]) {
			err = -ENOMEM;
			goto error;
		}
		new_first = new_last - (block_size - 1) * iterator->step;
		err =
		    citerator_range_init(results[i], new_first, new_last,
					 iterator->step);
		if (err) {
			citerator_free(results[i]);
			goto error;
		}
		new_last = new_first - iterator->step;
		size = size - block_size;
	}
	return 0;
error:
	i += 1;
	for (; i < n; i++)
		citerator_free(results[i]);
	return err;
}

int citerator_range_init(citerator_t iterator, citerator_index_t first,
			 citerator_index_t last, citerator_index_t step)
{
	struct range_iterator_s *range_iterator;

	if (!iterator || iterator->type != CITERATOR_RANGE)
		return -EINVAL;
	iterator->dimension = 1;
	range_iterator = (struct range_iterator_s *) iterator->data;
	range_iterator->first = first;
	range_iterator->v = first;
	range_iterator->last = last;
	range_iterator->step = step;
	return 0;
}

static const struct citerator_func_table_s citerator_range_func_table = {
	range_iterator_alloc,
	range_iterator_free,
	range_iterator_copy,
	range_iterator_next,
	range_iterator_peek,
	range_iterator_size,
	range_iterator_rewind,
	range_iterator_split,
	range_iterator_nth,
	range_iterator_n,
	range_iterator_pos
};

/*--------------------------------------------------------------------*/

#define ALLOC_CITERATOR(it, func_table) {\
	if (!func_table.alloc)\
		goto error;\
	it->functions = &func_table;\
	iterator->dimension = 0;\
	if (func_table.alloc(iterator))\
		goto error;\
}

citerator_t citerator_alloc(enum citerator_type_e type)
{
	citerator_t iterator;

	iterator = malloc(sizeof(const struct citerator_s));
	if (!iterator)
		return NULL;
	switch (type) {
	case CITERATOR_RANGE:
		ALLOC_CITERATOR(iterator, citerator_range_func_table);
		break;
	case CITERATOR_CONS:
		ALLOC_CITERATOR(iterator, citerator_cons_func_table);
		break;
	case CITERATOR_REPEAT:
		ALLOC_CITERATOR(iterator, citerator_repeat_func_table);
		break;
	case CITERATOR_HILBERT2D:
		ALLOC_CITERATOR(iterator, citerator_hilbert2d_func_table);
		break;
	case CITERATOR_PRODUCT:
		ALLOC_CITERATOR(iterator, citerator_product_func_table);
		break;
	case CITERATOR_SLICE:
		ALLOC_CITERATOR(iterator, citerator_slice_func_table);
		break;
	default:
		goto error;
	}
	iterator->type = type;
	return iterator;
error:
	free(iterator);
	return NULL;
}

citerator_t citerator_dup(citerator_t iterator)
{
	citerator_t result = NULL;

	if (!iterator || !iterator->data || !iterator->functions
	    || !iterator->functions->copy)
		return NULL;
	result = citerator_alloc(iterator->type);
	if (!result)
		return NULL;
	result->dimension = iterator->dimension;
	if (iterator->functions->copy(result, iterator))
		goto error;
	return result;
error:
	citerator_free(result);
	return NULL;
}

void citerator_free(citerator_t iterator)
{
	if (!iterator || !iterator->functions)
		return;
	if (iterator->functions->free)
		iterator->functions->free(iterator);
	free(iterator);
}

int citerator_dimension(citerator_t iterator, citerator_index_t *dimension)
{
	if (!iterator || !dimension)
		return -EINVAL;
	*dimension = iterator->dimension;
	return 0;
}

int citerator_type(citerator_t iterator, enum citerator_type_e *type)
{
	if (!iterator || !type)
		return -EINVAL;
	*type = iterator->type;
	return 0;
}

int citerator_next(citerator_t iterator, citerator_index_t *indexes)
{
	if (!iterator || !iterator->functions)
		return -EINVAL;
	if (!iterator->functions->next)
		return -ENOTSUP;
	return iterator->functions->next(iterator, indexes);
}

int citerator_peek(const citerator_t iterator, citerator_index_t *indexes)
{
	if (!iterator || !iterator->functions)
		return -EINVAL;
	if (!iterator->functions->peek)
		return -ENOTSUP;
	return iterator->functions->peek(iterator, indexes);
}

int citerator_size(const citerator_t iterator, citerator_index_t *size)
{
	if (!iterator || !iterator->functions || !size)
		return -EINVAL;
	if (!iterator->functions->size)
		return -ENOTSUP;
	return iterator->functions->size(iterator, size);
}

int citerator_rewind(citerator_t iterator)
{
	if (!iterator || !iterator->functions)
		return -EINVAL;
	if (!iterator->functions->rewind)
		return -ENOTSUP;
	return iterator->functions->rewind(iterator);
}

int citerator_split(const citerator_t iterator, citerator_index_t n,
		    citerator_t *results)
{
	if (!iterator || !iterator->functions)
		return -EINVAL;
	if (n <= 0)
		return -EDOM;
	if (!iterator->functions->split)
		return -ENOTSUP;
	return iterator->functions->split(iterator, n, results);
}

int citerator_nth(const citerator_t iterator, citerator_index_t n,
		  citerator_index_t *indexes)
{
	if (!iterator || !iterator->functions)
		return -EINVAL;
	if (!iterator->functions->nth)
		return -ENOTSUP;
	return iterator->functions->nth(iterator, n, indexes);
}

int citerator_n(const citerator_t iterator, const citerator_index_t *indexes,
		citerator_index_t *n)
{
	if (!iterator || !iterator->functions || !indexes)
		return -EINVAL;
	if (!iterator->functions->n)
		return -ENOTSUP;
	return iterator->functions->n(iterator, indexes, n);
}

int citerator_pos(const citerator_t iterator, citerator_index_t *n)
{
	if (!iterator || !iterator->functions)
		return -EINVAL;
	if (!iterator->functions->pos)
		return -ENOTSUP;
	return iterator->functions->pos(iterator, n);
}

int citerator_cyclic_next(citerator_t iterator, citerator_index_t *indexes,
			  int *looped)
{
	int err;

	if (!iterator)
		return -EINVAL;
	*looped = 0;
	err = citerator_next(iterator, indexes);
	switch (err) {
	case 0:
		break;
	case -ERANGE:
		err = citerator_rewind(iterator);
		if (err)
			return err;
		*looped = 1;
		err = citerator_next(iterator, indexes);
		if (err)
			return err;
		break;
	default:
		return err;
	}
	if (citerator_peek(iterator, NULL) == -ERANGE) {
		err = citerator_rewind(iterator);
		if (err)
			return err;
		*looped = 1;
	}
	return 0;
}

int citerator_skip(citerator_t iterator)
{
	return citerator_next(iterator, NULL);
}
