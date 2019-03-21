/**
 * AML area linux building blocks
 * Additional linux building blocks in <aml/area/linux_numa.h> if supported.
 **/

/**
 * Call to linux mmap().
 * mode is set to PROT_READ|PROT_WRITE.
 * flags is set to MAP_ANONYMOUS or'd with flags argument.
 **/
int
aml_area_linux_mmap_generic(void **ptr,
		       size_t size,
		       int    flags);

/**
 * Call to aml_area_linux_mmap_generic() with flag MAP_PRIVATE.
 **/
int
aml_area_linux_mmap_private(const struct aml_area* area,
		       void **ptr,
		       size_t size);

/**
 * Call to aml_area_linux_mmap_generic() with flag MAP_SHARED.
 **/
int
aml_area_linux_mmap_shared(const struct aml_area* area,
		      void **ptr,
		      size_t size);

/**
 * Call to linux munmap()
 **/
int
aml_area_linux_munmap(const struct aml_area* area,
		 void *ptr,
		 const size_t size);

/**
 * Call to linux malloc() or posix_memalign() depending on alignement argument.
 **/
int
aml_area_linux_malloc(const struct aml_area *area,
		 void                 **ptr,
		 size_t                 size,
		 size_t                 alignement);

/**
 * Call to linux free().
 **/
int
aml_area_linux_free(const struct aml_area *area,
	       void                  *ptr);

