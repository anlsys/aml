/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://xgitlab.cels.anl.gov/argo/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/

/**
 * This file contains utils function to set coordinates inside the elements
 * of a layout.
 **/

#include "aml.h"
#include <assert.h>

/** Count the number of bits required to store a value between 0 and n. **/
static size_t count_bits(size_t n)
{
	size_t b = 0;

	while (n > 0) {
		n = n / 2;
		b++;
	};
	return b;
}

void dims_nbits(const size_t ndims,
		const size_t *dims,
		size_t *bits)
{
	for (size_t i = 0; i < ndims; i++)
		bits[i] = count_bits(dims[i]);
}

uint64_t pack_coords(const size_t ndims,
		     const size_t *coords, const size_t *bits)
{
	size_t nbit = 0;
	uint64_t packed = 0;

	// Start from last coord then shift left each new coord
	for (size_t i = ndims - 1; i > 0; i--) {
		nbit += bits[i];
		assert(nbit <= 8 * sizeof(uint64_t));
		packed = packed | coords[i];	// store coord into the bitmask
		packed = packed << bits[i - 1];	// Shift for next coord
	}
	packed = packed | coords[0];	// store last coord into the bitmask
	return packed;
}

void unpack_coords(uint64_t coords,
		   const size_t ndims, const size_t *bits, size_t *out)
{
	size_t nbit = 0;

	// Start from first coord then shift right to unpack next coords.
	for (size_t i = 0; i < ndims; i++) {
		nbit += bits[i];
		assert(nbit <= 8 * sizeof(uint64_t));
		// Fill a bitmask of bits[i] bits then use it to unpack the
		// good bits.
		out[i] = coords & ((1 << bits[i]) - 1);
		// shift right to unpack next coords.
		coords = coords >> bits[i];
	}
}

void increment_coords(const size_t ndims,
		      const size_t *dims,
		      size_t *coords,
		      size_t n)
{
	for (size_t c, j = ndims - 1; j < ndims && n > 0; j--) {
		// Save the value of current coordinate.
		c = coords[j];
		// Update value in current coordinate.
		coords[j] = (c + n) % dims[j];
		// How much do we increment next coordinate.
		n = (c + n) / dims[j];
	}
}
