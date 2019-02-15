/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include <aml.h>
#include <assert.h>

#define TILESIZE (2)
#define NBTILES (4)

int main(int argc, char *argv[])
{
	struct aml_binding *a;
	AML_BINDING_SINGLE_DECL(b);
	
	AML_TILING_1D_DECL(t);

	void *ptr;

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize the bindings */
	aml_binding_create(&a, AML_BINDING_TYPE_SINGLE, 0);
	aml_binding_init(&b, AML_BINDING_TYPE_SINGLE, 0);

	/* create a tiling scheme */
	aml_tiling_init(&t, AML_TILING_TYPE_1D, TILESIZE*PAGE_SIZE,
			TILESIZE*PAGE_SIZE*NBTILES);

	/* allocate some memory */
	ptr = mmap(NULL, 1<<15, PROT_READ|PROT_WRITE,
		   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	assert(ptr != MAP_FAILED);

	/* now translate a binding into paging info */
	int nbpages = aml_binding_nbpages(a, &t, ptr, 0);
	void *pages[nbpages];
	int nodes[nbpages];
	aml_binding_pages(a, pages, &t, ptr, 0);
	aml_binding_nodes(a, nodes, &t, ptr, 0);

	assert(nbpages == TILESIZE);
	for(int i = 0; i < TILESIZE; i++)
	{
		assert(pages[i] == (char *)ptr + i*PAGE_SIZE);
		assert(nodes[i] == 0);
	}

	/* delete the bindings */
	aml_binding_destroy(a, AML_BINDING_TYPE_SINGLE);
	aml_binding_destroy(&b, AML_BINDING_TYPE_SINGLE);
	free(a);

	aml_finalize();
	return 0;
}
