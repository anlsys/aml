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

#endif
