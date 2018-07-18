#The arguments for matrix multiply are as follows:
#0) Program Name
#1) Size (in bytes) of the C matrix (and currently the A and B matrices)
#2) Number of threads to run computation
#3) Size (in bytes) of a tile.

echo "Running tests for 32 Threads" 
echo "7.03 GB"
./matrix_multiply_omp_tile 7549747200 32 819200
./matrix_multiply_omp_tile 7549747200 32 819200
./matrix_multiply_omp_tile 7549747200 32 7372800
./matrix_multiply_omp_tile 7549747200 32 7372800
echo "12.5 GB"
./matrix_multiply_omp_tile 13421772800 32 13107200
./matrix_multiply_omp_tile 13421772800 32 13107200
./matrix_multiply_omp_tile 13421772800 32 13107200
./matrix_multiply_omp_tile 13421772800 32 13107200

echo "19.53125 GB"
./matrix_multiply_omp_tile 20971520000 32 20480000
./matrix_multiply_omp_tile 20971520000 32 20480000
./matrix_multiply_omp_tile 20971520000 32 20480000
./matrix_multiply_omp_tile 20971520000 32 20480000
echo "28.125 GB"
./matrix_multiply_omp_tile 30198988800 32 7372800
./matrix_multiply_omp_tile 30198988800 32 7372800
./matrix_multiply_omp_tile 30198988800 32 29491200
./matrix_multiply_omp_tile 30198988800 32 29491200


echo "Running tests for 48 Threads" 
echo "7.03 GB"
./matrix_multiply_omp_tile 7549747200 48 3276800
./matrix_multiply_omp_tile 7549747200 48 3276800
./matrix_multiply_omp_tile 7549747200 48 3276800
./matrix_multiply_omp_tile 7549747200 48 3276800
echo "15.82 GB"
./matrix_multiply_omp_tile 16986931200 48 7372800
./matrix_multiply_omp_tile 16986931200 48 7372800
./matrix_multiply_omp_tile 16986931200 48 7372800
./matrix_multiply_omp_tile 16986931200 48 7372800
echo "28.125 GB"
./matrix_multiply_omp_tile 30198988800 48 3276800
./matrix_multiply_omp_tile 30198988800 48 3276800
./matrix_multiply_omp_tile 30198988800 48 13107200
./matrix_multiply_omp_tile 30198988800 48 13107200

echo "Running tests for 64 Threads" 
echo "3.125 GB"
./matrix_multiply_omp_tile 3355443200 64 819200
./matrix_multiply_omp_tile 3355443200 64 819200
./matrix_multiply_omp_tile 3355443200 64 819200
./matrix_multiply_omp_tile 3355443200 64 819200
echo "12.5 GB"
./matrix_multiply_omp_tile 13421772800 64 3276800
./matrix_multiply_omp_tile 13421772800 64 3276800
./matrix_multiply_omp_tile 13421772800 64 3276800
./matrix_multiply_omp_tile 13421772800 64 3276800
echo "28.125 GB"
./matrix_multiply_omp_tile 30198988800 64 7372800
./matrix_multiply_omp_tile 30198988800 64 7372800
./matrix_multiply_omp_tile 30198988800 64 7372800
./matrix_multiply_omp_tile 30198988800 64 7372800



