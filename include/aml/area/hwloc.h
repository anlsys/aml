#ifndef AML_AREA_HWLOC_H
#define AML_AREA_HWLOC_H

/** struct aml_bitmap can be translated to hwloc_nodeset_t and hwloc_bitmap_t **/
#include <aml/utils/hwloc.h>

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

#endif
