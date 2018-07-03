#The arguments for matrix multiply are as follows:
#0) Program Name
#1) Size (in bytes) of the C matrix (and currently the A and B matrices)
#2) Number of threads to run computation
#3) Size (in bytes) of a tile.

#This run has a tile as close to the size of the L2 Cache (1MB) as possible.
./matrix_multiply_omp_tile 67094528 4 4193408 # < 4 MB
./matrix_multiply_omp_tile 67094528 4 1048352 # < 1 MB
./matrix_multiply_omp_tile 67094528 4 262088  # < 256 KB

#This set of runs will have tile sizes that are a 2^n value

./matrix_multiply_omp_tile 33554432 4 2097152 #2 MB
./matrix_multiply_omp_tile 33554432 4 524288  #512 KB
./matrix_multiply_omp_tile 33554432 4 131072  #128 KB
./matrix_multiply_omp_tile 33554432 4 32768   #32 KB




