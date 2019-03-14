#ifndef AML_TILING_H
#define AML_TILING_H

/* opaque handle to all tilings */
struct aml_tiling_data;
struct aml_tiling_iterator_data;

/*forward declarations */
struct aml_tiling_iterator_ops;
struct aml_tiling_iterator;


struct aml_tiling_ops {
	int (*create_iterator)(struct aml_tiling_data *tiling,
			       struct aml_tiling_iterator **iterator,
			       int flags);
	int (*init_iterator)(struct aml_tiling_data *tiling,
			     struct aml_tiling_iterator *iterator, int flags);
	int (*destroy_iterator)(struct aml_tiling_data *tiling,
				struct aml_tiling_iterator *iterator);
	int (*tileid)(const struct aml_tiling_data *tiling, va_list);
	size_t (*tilesize)(const struct aml_tiling_data *tiling, int tileid);
	void* (*tilestart)(const struct aml_tiling_data *tiling,
			   const void *ptr, int tileid);
	int (*ndims)(const struct aml_tiling_data *tiling, va_list);
};


struct aml_tiling {
	struct aml_tiling_ops *ops;
	struct aml_tiling_data *data;
};

struct aml_tiling_iterator_ops {
	int (*reset)(struct aml_tiling_iterator_data *iterator);
	int (*next)(struct aml_tiling_iterator_data *iterator);
	int (*end)(const struct aml_tiling_iterator_data *iterator);
	int (*get)(const struct aml_tiling_iterator_data *iterator,
		   va_list args);
};

struct aml_tiling_iterator {
	struct aml_tiling_iterator_ops *ops;
	struct aml_tiling_iterator_data *data;
};

/*******************************************************************************
 * Tiling 1D:
 ******************************************************************************/

extern struct aml_tiling_ops aml_tiling_1d_ops;
extern struct aml_tiling_iterator_ops aml_tiling_iterator_1d_ops;

struct aml_tiling_1d_data {
	size_t blocksize;
	size_t totalsize;
};

struct aml_tiling_iterator_1d_data {
	size_t i;
	struct aml_tiling_1d_data *tiling;
};

#define AML_TILING_1D_DECL(name) \
	struct aml_tiling_1d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_1d_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_ITERATOR_1D_DECL(name) \
	struct aml_tiling_iterator_1d_data __ ##name## _inner_data; \
	struct aml_tiling_iterator name = { \
		&aml_tiling_iterator_1d_ops, \
		(struct aml_tiling_iterator_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_1D_ALLOCSIZE (sizeof(struct aml_tiling_1d_data) + \
				 sizeof(struct aml_tiling))

#define AML_TILING_ITERATOR_1D_ALLOCSIZE \
	(sizeof(struct aml_tiling_iterator_1d_data) + \
	 sizeof(struct aml_tiling_iterator))


/*******************************************************************************
 * Tiling 2D:
 * a contiguous memory area composed of contiguous tiles arranged in 2D grid.
 ******************************************************************************/

extern struct aml_tiling_ops aml_tiling_2d_rowmajor_ops;
extern struct aml_tiling_ops aml_tiling_2d_colmajor_ops;
extern struct aml_tiling_iterator_ops aml_tiling_iterator_2d_ops;

struct aml_tiling_2d_data {
	size_t blocksize;
	size_t totalsize;
	size_t ndims[2]; /* # number of rows, # number of cols (in tiles) */
};

struct aml_tiling_iterator_2d_data {
	size_t i;
	struct aml_tiling_2d_data *tiling;
};

#define AML_TILING_2D_ROWMAJOR_DECL(name) \
	struct aml_tiling_2d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_2d_rowmajor_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_2D_COLMAJOR_DECL(name) \
	struct aml_tiling_2d_data __ ##name## _inner_data; \
	struct aml_tiling name = { \
		&aml_tiling_2d_colmajor_ops, \
		(struct aml_tiling_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_ITERATOR_2D_DECL(name) \
	struct aml_tiling_iterator_2d_data __ ##name## _inner_data; \
	struct aml_tiling_iterator name = { \
		&aml_tiling_iterator_2d_ops, \
		(struct aml_tiling_iterator_data *)&__ ## name ## _inner_data, \
	};

#define AML_TILING_2D_ALLOCSIZE (sizeof(struct aml_tiling_2d_data) + \
				 sizeof(struct aml_tiling))

#define AML_TILING_ITERATOR_2D_ALLOCSIZE \
	(sizeof(struct aml_tiling_iterator_2d_data) + \
	 sizeof(struct aml_tiling_iterator))


#endif //AML_TILING_H
