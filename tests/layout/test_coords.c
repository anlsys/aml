/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

#include "test_layout.h"

void test_pack_unpack_coords(const size_t ndims,
			     const size_t *dims,
			     const size_t *coords)
{
	size_t unpacked[ndims];
	size_t bits[ndims];

	dims_nbits(ndims, dims, bits);

	uint64_t packed = pack_coords(ndims, coords, bits);

	unpack_coords(packed, ndims, bits, unpacked);
	assert(!memcmp(coords, unpacked, sizeof(unpacked)));
}

void test_increment_coords(const size_t ndims, const size_t *dims)
{
	size_t num = 1;
	size_t coords[ndims];
	size_t prev[ndims];

	for (size_t i = 0; i < ndims; i++) {
		num = num * dims[i];
		coords[i] = 0;
		prev[i] = 0;
	}

	for (size_t i = 0; i < num; i++) {
		increment_coords(ndims, dims, coords, 1);

		assert((coords[ndims - 1] == 0 &&
			(prev[ndims - 1] == 0 ||
			 prev[ndims - 1] == dims[ndims - 1] - 1)) ||
		       coords[ndims - 1] == prev[ndims - 1] ||
		       coords[ndims - 1] == prev[ndims - 1] + 1);
		prev[ndims - 1] = coords[ndims - 1];

		for (size_t j = 0; j < ndims - 1; j++) {
			assert((coords[j] == 0 &&
				(prev[j] == 0 ||
				 prev[j] == dims[j] - 1)) ||
			       coords[j] == prev[j] ||
			       (coords[j] == prev[j] + 1 &&
				coords[j + 1] == 0));
			prev[j] = coords[j];
		}
	}

	for (size_t i = 0; i < ndims; i++)
		assert(coords[i] == 0);
}

int main(void)
{
	size_t dims0[4] = { 5, 7, 2, 8 };
	size_t coords0[4] = { 3, 2, 0, 6 };

	test_increment_coords(4, dims0);
	test_pack_unpack_coords(4, dims0, coords0);

	return 0;
}
