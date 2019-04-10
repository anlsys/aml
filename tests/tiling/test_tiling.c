/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
*******************************************************************************/

#include "aml.h"
#include "aml/tiling/1d.h"
#include "aml/tiling/2d.h"
#include <assert.h>

#define TILESIZE 8192
#define NBTILES 4

int doit(struct aml_tiling *t, struct aml_tiling_iterator *it)
{
	size_t tilesize;
	unsigned long i;
	intptr_t ptr;
	tilesize = aml_tiling_tilesize(t, 0);
	assert(tilesize == TILESIZE);

	/* actualy use the iterators */
	for(aml_tiling_iterator_reset(it);
	    !aml_tiling_iterator_end(it);
	    aml_tiling_iterator_next(it))
	{
		aml_tiling_iterator_get(it, &i);
	}
	assert(i == NBTILES -1);

	for(i = 0; i < NBTILES; i++)
	{
		ptr = (intptr_t) aml_tiling_tilestart(t, NULL, i);
		assert(ptr == i*TILESIZE);
	}
}

int main(int argc, char *argv[])
{
	struct aml_tiling *a;
	AML_TILING_1D_DECL(b);
	
	struct aml_tiling_iterator *ita;
	AML_TILING_ITERATOR_1D_DECL(itb);

	/* library initialization */
	aml_init(&argc, &argv);

	/* initialize the tilings */
	aml_tiling_1d_create(&a, TILESIZE, TILESIZE*NBTILES);
	aml_tiling_1d_init(&b, TILESIZE, TILESIZE*NBTILES);

	/* initialize the iterators */
	aml_tiling_create_iterator(a, &ita, 0);
	aml_tiling_init_iterator(&b, &itb, 0);

	doit(a, ita);
	doit(&b, &itb);

	/* delete the iterators */
	aml_tiling_destroy_iterator(a, &ita);
	aml_tiling_fini_iterator(&b, &itb);


	/* delete the tilings */
	aml_tiling_1d_destroy(&a);
	aml_tiling_1d_fini(&b);

	aml_finalize();
	return 0;
}
