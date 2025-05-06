/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/
#include <stdlib.h>
#include <string.h>

#include "aml.h"
#include "aml/area/mpi.h"

int main(void)
{
    // init runtimes
	assert(aml_init(NULL, NULL) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_MPI))
		return 77;
    assert(MPI_Init(NULL, NULL) == MPI_SUCCESS);

    // test MPI area
    {
        struct aml_area * area;
        assert(aml_area_mpi_create(&area, ...) == AML_SUCCESS);

        void * ptr = aml_area_mmap(area, size, NULL);
        assert(ptr);

        assert(aml_area_munmap(area, ptr, size) == AML_SUCCESS);
        aml_area_mpi_destroy(&area);
    }

    // finalize runtimes
    assert(MPI_Finalize == MPI_SUCCESS);
	aml_finalize();

	return 0;
}
