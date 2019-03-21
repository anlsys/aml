#ifndef AML_AREA_HWLOC_H
#define AML_AREA_HWLOC_H

#include <hwloc.h>

/** 
 * Use linux areas allocation with hwloc binding capabilities for process wide
 * memory operations.
 **/
extern struct aml_area *aml_area_hwloc_private;

/** 
 * Use linux areas allocation with hwloc binding capabilities for cross process
 * memory operations.
 **/
extern struct aml_area *aml_area_hwloc_shared;

/* Bind memory on given nodeset with HWLOC_MEMBIND_BIND policy */
const extern unsigned long aml_area_hwloc_flag_bind;
/* Bind memory on given nodeset with HWLOC_MEMBIND_INTERLEAVE policy */
const extern unsigned long aml_area_hwloc_flag_interleave;
/* Bind memory on given nodeset with HWLOC_MEMBIND_FIRSTTOUCH policy */
const extern unsigned long aml_area_hwloc_flag_firsttouch;
/* Bind memory on given nodeset with HWLOC_MEMBIND_NEXTTTOUCH policy */
const extern unsigned long aml_area_hwloc_flag_nexttouch;

/************************************************************************************
 * Low level implementation details
 ************************************************************************************/

struct aml_area_ops; /* definition in <aml/area/area.h> */

/** User data stored inside aml area.**/
struct hwloc_binding{
	struct aml_area_ops    ops;     //Used for malloc/free and mmap/munmap
	hwloc_bitmap_t         nodeset; //Nodeset where to bind. Can be NULL.
        hwloc_membind_policy_t policy;  //The memory binding policy.
};

int
aml_area_hwloc_create(struct aml_area* area);

void
aml_area_hwloc_destroy(struct aml_area* area);

int
aml_area_hwloc_bind(struct aml_area         *area,
		    const struct aml_bitmap *nodeset,
		    const unsigned long      policy);

int
aml_area_hwloc_bind(struct aml_area *area,
		    const struct aml_bitmap *bitmap,
		    const unsigned long hwloc_policy);

int
aml_area_hwloc_check_binding(struct aml_area *area,
			     void            *ptr,
			     size_t           size);

int
aml_area_hwloc_mmap(const struct aml_area* area,
		    void **ptr,
		    size_t size);

int
aml_area_hwloc_munmap(const struct aml_area* area,
		      void *ptr,
		      const size_t size);

int
aml_area_hwloc_malloc(const struct aml_area* area,
		      void **ptr,
		      size_t size,
		      size_t alignement);

int
aml_area_hwloc_free(const struct aml_area *area,
		    void                  *ptr);
#endif
