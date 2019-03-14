#ifndef AML_AREA_H
#define AML_AREA_H

#include <aml/utils/bitmap.h>

/*********************************************************************************
 * Lower level area management.
 *********************************************************************************/

struct aml_area;

/** Implementation specific operations. **/
struct aml_area_ops {
	/**
	 * Initialisation of userdata inside area. 
	 * "area": Cannot be NULL.
	 * Returns AML_AREA_* error code.
	 **/
        int (*create)(struct aml_area*);
	/**
	 * Destruction of userdata inside area. 
	 * "area": Cannot be NULL.
	 **/
	void (*destroy)(struct aml_area*);
	/**
	 * Bind area to a specific set of memories.
	 * "area": Cannot be NULL.
	 * "binding": May be NULL which means "do not bind".
	 *            Special function may be included in area header for populating 
	 *            the bitmap.
	 * "flags": Can be any value. 
	 *          Special values may be included in area header.
	 * Returns AML_AREA_* error code.
	 **/
	int (*bind)(struct aml_area *,
		    const aml_bitmap binding,
		    const unsigned long flags);
	/**
	 * Optional function to check a binding has effectively been applied.
	 * The binding to enforce is the one defining area.
	 * "area": Cannot be NULL.
	 * "ptr": The data to check for binding. Cannot be NULL.
	 * "size" The data size. data is greater than 0.
	 * Returns 0 if ptr binding does not match area settings, 
	 * else a positive value or an error code on error. 
	 **/
	int (*check_binding)(struct aml_area *, void * ptr, const size_t size);
	/**
	 * Coarse grain allocator of virtual memory.
	 *
	 * "area": The area operations to use. Cannot be NULL.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 * "size": The size of data. Is greater than 0.
	 *
	 * Returns AML_AREA_* error code.
	 **/
        int (*map)(const struct aml_area*, void **ptr, size_t size);
	/**
	 * Unmapping of virtual memory mapped with map().
	 *
	 * "area": The area operations to use. Cannot be NULL.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 * "size": The size of data. Cannot be 0.
	 *
	 * Returns AML_AREA_* error code.
	 **/
        int (*unmap)(const struct aml_area *, void *ptr, size_t size);
	/**
	 * Fine grain allocator of virtual memory.
	 * Memory allocated with malloc() is released with free().
	 *
	 * "area": The area operations to use. Cannot be NULL.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 * "size": The size of data. Is greater than 0.
	 * "alignement": If 0 no alignement needs to be enforced. If greater than 0,
	 *               but implementation does not support alignement,
	 *               AML_AREA_ENOTSUP must be returned.
	 *
	 * Returns AML_AREA_* error code.
	 **/
	int (*malloc)(struct aml_area *, void **ptr, size_t size, size_t alignement);
	/**
	 * Free memory allocated with malloc().
	 *
	 * "area": The area operations to use. Cannot be NULL.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 *
	 * Returns AML_AREA_* error code.
	 **/
	int (*free)(struct aml_area * area, void *ptr);
};

struct aml_area {
	/* Basic memory operations implementation */
	struct aml_area_ops *ops;
	/* Implmentation specific data. Set to NULL at creation. */
	void *data;
};

#endif //AML_AREA_H