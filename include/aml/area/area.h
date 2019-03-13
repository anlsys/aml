#ifndef AML_AREA_H
#define AML_AREA_H

#include <aml/utils/bitmap.h>

/*********************************************************************************
 * Lower level area management.
 *********************************************************************************/

struct aml_area;

/** Implementation specific operations. **/
struct aml_area_ops {
	/** Initialisation of user data inside area. **/
        int (*create)(struct aml_area*);
	/** Destruction of user data inside area. **/
	void (*destroy)(struct aml_area*);
	/** Bind area to a specific set of memories. 
	    Return AML_AREA_ENOTSUP if not supported **/
	int (*bind)(struct aml_area *,
		    const aml_bitmap binding,
		    const unsigned long flags);
	/** Returns 0 if ptr binding does not match area settings, 
	    else a positive value or an error code on error. **/
	int (*check_binding)(struct aml_area *, void * ptr, const size_t size);
	/** See aml_area_mmap() **/
        int (*map)(const struct aml_area*, void **ptr, size_t size);
	/** See aml_area_munmap() **/
        int (*unmap)(const struct aml_area *, void *ptr, size_t size);
	/** See aml_area_malloc() **/
	int (*malloc)(struct aml_area *, void **ptr, size_t size, size_t alignement);
	/** See aml_area_free() **/
	int (*free)(struct aml_area *, void *ptr);
};

struct aml_area {
	/* Basic memory operations implementation */
	struct aml_area_ops *ops;
	/* Implmentation specific data */
	void *data;
};

#endif //AML_AREA_H
