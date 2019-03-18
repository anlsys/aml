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
        int (*create)(struct aml_area *area);
	
	/**
	 * Destruction of userdata inside area. 
	 * "area": Cannot be NULL.
	 **/
	void (*destroy)(struct aml_area *area);
	
	/**
	 * Bind area to a specific set of memories.
	 * "area": Cannot be NULL.
	 * "binding": May be NULL which means "do change binding".
	 *            Special function may be included in area header for populating 
	 *            the bitmap.
	 * "flags": Can be any value. 
	 *          Special values may be included in area header.
	 * Returns AML_AREA_* error code. AML_AREA_EDOM in particular if binding is not a valid binding.
	 **/
	int (*bind)(struct aml_area         *area,
		    const struct aml_bitmap *binding,
		    const unsigned long      flags);
	
	/**
	 * Optional function to check a binding has effectively been applied.
	 * The binding to enforce is the one defining area.
	 * "area": Cannot be NULL.
	 * "ptr": The data to check for binding. Cannot be NULL.
	 * "size" The data size. data is greater than 0.
	 * Returns 0 if ptr binding does not match area settings, 
	 * else a positive value or an error code on error. 
	 **/
	int (*check_binding)(struct aml_area *area,
			     void            *ptr,
			     const size_t     size);
	
	/**
	 * Coarse grain allocator of virtual memory.
	 *
	 * "area": The area operations to use. Cannot be NULL.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 * "size": The size of data. Is greater than 0.
	 *
	 * Returns AML_AREA_* error code.
	 **/
        int (*map)(const struct aml_area *area,
		   void                 **ptr,
		   size_t                 size);
	
	/**
	 * Unmapping of virtual memory mapped with map().
	 *
	 * "area": The area operations to use. Cannot be NULL.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 * "size": The size of data. Cannot be 0.
	 *
	 * Returns AML_AREA_* error code.
	 **/
        int (*unmap)(const struct aml_area *area,
		     void                  *ptr,
		     size_t                 size);
	
	/**
	 * Fine grain allocator of virtual memory.
	 * Memory allocated with malloc() is released with free().
	 * (Optional) Fallback on map() is done if not implemented.
	 * If not implemented, free must be unimplemented.
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
	int (*malloc)(struct aml_area *area,
		      void           **ptr,
		      size_t           size,
		      size_t           alignement);
	
	/**
	 * Free memory allocated with malloc().
	 * (Optional) Fallback on unmap() is done if not implemented.
	 * If not implemented, malloc() must be unimplemented.
	 *
	 * "area": The area operations to use. Cannot be NULL.
	 * "ptr": Pointer to data mapped in physical memory. Cannot be NULL.
	 *
	 * Returns AML_AREA_* error code.
	 **/
	int (*free)(struct aml_area *area,
		    void            *ptr);
};

struct aml_area {
	/* Basic memory operations implementation */
	struct aml_area_ops *ops;
	/* Implmentation specific data. Set to NULL at creation. */
	void *data;
};

/*
 * Implementation specific functions used in several areas
 */

int
linux_mmap_private(const struct aml_area* area,
		   void **ptr,
		   size_t size);

int
linux_mmap_shared(const struct aml_area* area,
		  void **ptr,
		  size_t size);

int
linux_munmap(const struct aml_area* area,
	     void *ptr,
	     const size_t size);

int
linux_malloc(struct aml_area *area,
	     void           **ptr,
	     size_t           size,
	     size_t           alignement);
	
int
linux_free(struct aml_area *area,
	   void *ptr);

#endif //AML_AREA_H
