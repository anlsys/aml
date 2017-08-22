#include <aml.h>
#include <assert.h>
#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

#ifndef MAX_NUMNODES
#define MAX_NUMNODES 64
#endif

int aml_init(int *argc, char **argv[])
{
	return 0;
}

int aml_finalize(void)
{
	return 0;
}
