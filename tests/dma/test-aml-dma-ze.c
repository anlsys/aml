/*******************************************************************************
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *******************************************************************************/

#include <level_zero/ze_api.h>
#include <time.h>

#include <aml.h>

#include <aml/area/ze.h>
#include <aml/dma/ze.h>

#define min(a, b)                                                              \
	({                                                                     \
		__typeof__(a) _a = (a);                                        \
		__typeof__(b) _b = (b);                                        \
		_a < _b ? _a : _b;                                             \
	})

static int64_t timediff(struct timespec end, struct timespec start)
{
	int64_t e = end.tv_sec * 1e9 + end.tv_nsec;
	int64_t s = start.tv_sec * 1e9 + start.tv_nsec;
	return e - s;
}

static double bw(size_t size, int64_t time)
{
	return size / (double)time;
}

int do_copy_1way(
        struct aml_dma *dma, void *dest, void *src, size_t size, int repeat)
{
	int64_t mintime = INT64_MAX;
	struct timespec start, end;
	for (int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		aml_dma_copy_custom(dma, dest, src, aml_dma_ze_memcpy_op,
		                    (void *)size);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	printf("BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int do_copy_chunks(struct aml_dma *dma,
                   void *dest,
                   void *src,
                   size_t size,
                   size_t chunks,
                   int repeat)
{
	int64_t mintime = INT64_MAX;
	struct timespec start, end;
	size_t cksz = size / chunks;
	for (int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		for (int j = 0; j < chunks; j++) {
			aml_dma_async_copy_custom(dma, NULL, dest + (j * cksz),
			                          src + (j * cksz),
			                          aml_dma_ze_memcpy_op,
			                          (void *)cksz);
		}
		aml_dma_barrier(dma);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	printf("BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int do_copy_2barrier(struct aml_dma *dma,
                     void *dest1,
                     void *src1,
                     void *dest2,
                     void *src2,
                     size_t size,
                     int repeat)
{
	int64_t mintime = INT64_MAX;
	struct timespec start, end;
	for (int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		aml_dma_async_copy_custom(dma, NULL, dest1, src1,
		                          aml_dma_ze_memcpy_op, (void *)size);
		aml_dma_async_copy_custom(dma, NULL, dest2, src2,
		                          aml_dma_ze_memcpy_op, (void *)size);
		aml_dma_barrier(dma);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	printf("BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int do_copy_2chunks(struct aml_dma *dma,
                    void *dest1,
                    void *src1,
                    void *dest2,
                    void *src2,
                    size_t size,
                    size_t chunks,
                    int repeat)
{
	int64_t mintime = INT64_MAX;
	struct timespec start, end;
	size_t cksz = size / chunks;
	for (int i = 0; i < repeat; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		for (int j = 0; j < chunks; j++) {
			aml_dma_async_copy_custom(dma, NULL, dest1 + (j * cksz),
			                          src1 + (j * cksz),
			                          aml_dma_ze_memcpy_op,
			                          (void *)cksz);
			aml_dma_async_copy_custom(dma, NULL, dest2 + (j * cksz),
			                          src2 + (j * cksz),
			                          aml_dma_ze_memcpy_op,
			                          (void *)cksz);
		}
		aml_dma_barrier(dma);
		clock_gettime(CLOCK_MONOTONIC, &end);
		mintime = min(mintime, timediff(end, start));
	}
	printf("BW: %gGBytes/s\n", bw(size, mintime));
	return 0;
}

int main(int argc, char *argv[])
{
	const size_t bytes = 1e9;
	const size_t N = bytes / sizeof(double);
	const int repeats = 100;

	assert(aml_init(&argc, &argv) == AML_SUCCESS);
	if (!aml_support_backends(AML_BACKEND_ZE))
		return 77;

	double *host =
	        aml_area_mmap(aml_area_ze_host, N * sizeof(double), NULL);
	double *device =
	        aml_area_mmap(aml_area_ze_device, N * sizeof(double), NULL);

	printf("Default ZE DMA\n");
	printf("H2D: 1C-wait\n");
	do_copy_1way(aml_dma_ze_default, device, host, bytes, repeats);

	for (size_t i = 1; i < 5; i++) {
		printf("Default ZE DMA\n");
		printf("H2D: Chunks: %zu\n", i);
		do_copy_chunks(aml_dma_ze_default, device, host, bytes, i,
		               repeats);
	}

	printf("Default ZE DMA\n");
	printf("D2H: 1C-wait\n");
	do_copy_1way(aml_dma_ze_default, host, device, bytes, repeats);

	for (size_t i = 1; i < 5; i++) {
		printf("Default ZE DMA\n");
		printf("D2H: Chunks: %zu\n", i);
		do_copy_chunks(aml_dma_ze_default, host, device, bytes, i,
		               repeats);
	}

	double *host2 =
	        aml_area_mmap(aml_area_ze_host, N * sizeof(double), NULL);
	double *device2 =
	        aml_area_mmap(aml_area_ze_device, N * sizeof(double), NULL);

	printf("Default ZE DMA\n");
	printf("2Way: 1C-barrier\n");
	do_copy_2barrier(aml_dma_ze_default, host, device, device2, host2,
	                 bytes, repeats);

	for (size_t i = 1; i < 5; i++) {
		printf("Default ZE DMA\n");
		printf("2Way: Chunks: %zu\n", i);
		do_copy_2chunks(aml_dma_ze_default, host, device, device2,
		                host2, bytes, i, repeats);
	}
	aml_finalize();
	return 0;
}
