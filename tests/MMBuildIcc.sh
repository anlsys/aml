icc -std=gnu99 -DHAVE_CONFIG_H -I. -I../src -O3 -qopenmp -xCORE-AVX2 -mkl -static-intel -qopt-report-phase=vec  -liomp5 -lpthread -lm -ldl -I../src -O3 -MT matrix_multiply_omp_tile.o -MD -MP -MF .deps/matrix_multiply_omp_tile.Tpo -c -o matrix_multiply_omp_tile.o matrix_multiply_omp_tile.c

mv -f .deps/matrix_multiply_omp_tile.Tpo .deps/matrix_multiply_omp_tile.Po

/bin/sh ../libtool  --tag=CC   --mode=link icc -std=gnu99 -O3 -qopenmp -xCORE-AVX2 -mkl -static-intel -qopt-report-phase=vec -I../src -O3 -mkl -xCORE-AVX2 -static-intel -qopenmp -qopt-report-phase=vec ../src/libaml.la ../jemalloc/lib/libjemalloc-aml.so  -o matrix_multiply_omp_tile matrix_multiply_omp_tile.o  -lnuma -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl 
