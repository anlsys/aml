icc -std=gnu99 -DHAVE_CONFIG_H -I. -I../src -O2 -fopenmp -mkl -static-intel -I../src -g -O2 -MT matrix_multiply_omp_tile.o -MD -MP -MF .deps/matrix_multiply_omp_tile.Tpo -c -o matrix_multiply_omp_tile.o matrix_multiply_omp_tile.c

mv -f .deps/matrix_multiply_omp_tile.Tpo .deps/matrix_multiply_omp_tile.Po

/bin/sh ../libtool  --tag=CC   --mode=link icc -std=gnu99 -O2 -fopenmp -mkl -static-intel -I../src -g -O2 -fopenmp ../src/libaml.la ../jemalloc/lib/libjemalloc-aml.so  -o matrix_multiply_omp_tile matrix_multiply_omp_tile.o  -lnuma

