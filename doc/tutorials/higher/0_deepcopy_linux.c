/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include <string.h>

#include "aml.h"

// Mapper includes
#include "aml/higher/mapper.h"

// Allocation (linux)
#include "aml/area/linux.h"

// Copy (linux)
#include "aml/dma/linux-seq.h"

//- Mapper for standard type --------------------------------------------------

aml_final_mapper_decl(double_mapper, double);

//- Mapper for simple struct --------------------------------------------------

struct A {
	size_t val;
};
aml_final_mapper_decl(struct_A_mapper, struct A);

//- Mapper for struct with descendants ----------------------------------------

struct B {
	int dummy_int;
	double dummy_double;
	struct A *a;
};
aml_mapper_decl(struct_B_mapper, struct B, a, &struct_A_mapper);

//- Mapper for struct with arrays ---------------------------------------------

struct C {
	size_t nb;
	struct B *b;
	size_t nx;
	double *x;
};
aml_mapper_decl(struct_C_mapper,
                struct C,
                b,
                nb,
                &struct_B_mapper,
                x,
                nx,
                &double_mapper);

//- Mapping args ---------------------------------------------------------------

struct aml_mapper_args linux_mapper_args;

//- Application struct initialization ------------------------------------------

struct C *init_struct()
{
	const size_t nb = 8;
	const size_t nx = 16;
	struct C *c;
	struct A *a;

	c = malloc(sizeof(*c) + nx * sizeof(double) +
	           nb * (sizeof(struct B) + sizeof(struct A)));

	c->nb = nb;
	c->nx = nx;
	c->x = (double *)((char *)c + sizeof(*c));
	c->b = (struct B *)((char *)c->x + nx * sizeof(double));
	a = (struct A *)((char *)c->b + nb * sizeof(struct B));

	for (size_t i = 0; i < nx; i++)
		c->x[i] = -1.0 * i;

	for (size_t i = 0; i < nb; i++) {
		c->b[i].dummy_int = i;
		c->b[i].dummy_double = i * 1e3;
		c->b[i].a = &a[i];
		c->b[i].a->val = i;
	}

	return c;
}

//- Compare structures --------------------------------------------------------

int eq_struct(const struct C *c, const struct C *_c)
{
	if (c->nx != _c->nx)
		return 0;
	if (c->nb != _c->nb)
		return 0;
	if (memcmp(c->x, _c->x, c->nx * sizeof(double)))
		return 0;
	for (size_t i = 0; i < c->nb; i++) {
		if (c->b[i].dummy_int != _c->b[i].dummy_int)
			return 0;
		if (c->b[i].dummy_double != _c->b[i].dummy_double)
			return 0;
		if (c->b[i].a->val != _c->b[i].a->val)
			return 0;
	}
	return 1;
}

//- Deep copy -----------------------------------------------------------------

#ifndef DEEP_COPY_CUDA // Disable in next tutorial
int main(int argc, char **argv)
{
	struct C *c, *_c;

	// Init
	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	linux_mapper_args.area = &aml_area_linux;
	linux_mapper_args.area_opts = NULL;
	linux_mapper_args.dma = aml_dma_linux_sequential;
	linux_mapper_args.dma_op = NULL;
	linux_mapper_args.dma_op_arg = NULL;

	c = init_struct();

	// deepcopy
	assert(aml_mapper_mmap(&struct_C_mapper, &linux_mapper_args, c, &_c,
	                       1) == AML_SUCCESS);
	assert(eq_struct(c, _c));

	// Cleanup
	aml_mapper_munmap(&struct_C_mapper, &linux_mapper_args, _c);
	free(c);
	aml_finalize();
	return 0;
}
#endif
