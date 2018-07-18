#The arguments for matrix multiply are as follows:
#0) Program Name
#1) Size (in bytes) of the C matrix (and currently the A and B matrices)
#2) Number of threads to run computation
#3) Size (in bytes) of a tile.

echo "Running tests for 16 Threads" 
echo "7.03 GB"
./matrix_multiply_omp_tile 7549747200 16 7372800
./matrix_multiply_omp_tile 7549747200 16 7372800
./matrix_multiply_omp_tile 7549747200 16 29491200
./matrix_multiply_omp_tile 7549747200 16 29491200

echo "12.5 GB"
./matrix_multiply_omp_tile 13421772800 16 819200
./matrix_multiply_omp_tile 13421772800 16 819200
./matrix_multiply_omp_tile 13421772800 16 3276800
./matrix_multiply_omp_tile 13421772800 16 3276800
./matrix_multiply_omp_tile 13421772800 16 13107200
./matrix_multiply_omp_tile 13421772800 16 13107200

echo "19.53125 GB"
./matrix_multiply_omp_tile 20971520000 16 819200
./matrix_multiply_omp_tile 20971520000 16 819200
./matrix_multiply_omp_tile 20971520000 16 5120000
./matrix_multiply_omp_tile 20971520000 16 5120000
./matrix_multiply_omp_tile 20971520000 16 20480000
./matrix_multiply_omp_tile 20971520000 16 20480000
 
echo "28.125 GB"
./matrix_multiply_omp_tile 30198988800 16 3276800
./matrix_multiply_omp_tile 30198988800 16 3276800
./matrix_multiply_omp_tile 30198988800 16 7372800
./matrix_multiply_omp_tile 30198988800 16 7372800
./matrix_multiply_omp_tile 30198988800 16 29491200
./matrix_multiply_omp_tile 30198988800 16 29491200
./matrix_multiply_omp_tile 30198988800 16 117964800
./matrix_multiply_omp_tile 30198988800 16 117964800

echo "Running tests for 8 Threads" 
echo "7.03 GB"
./matrix_multiply_omp_tile 7549747200 8 29491200 
./matrix_multiply_omp_tile 7549747200 8 29491200
./matrix_multiply_omp_tile 7549747200 8 117964800
./matrix_multiply_omp_tile 7549747200 8 117964800

echo "12.5 GB"
./matrix_multiply_omp_tile 13421772800 8 13107200
./matrix_multiply_omp_tile 13421772800 8 13107200
./matrix_multiply_omp_tile 13421772800 8 52428800
./matrix_multiply_omp_tile 13421772800 8 52428800
./matrix_multiply_omp_tile 13421772800 8 209715200
./matrix_multiply_omp_tile 13421772800 8 209715200

echo "19.53125 GB"
./matrix_multiply_omp_tile 20971520000 8 20480000 
./matrix_multiply_omp_tile 20971520000 8 20480000
./matrix_multiply_omp_tile 20971520000 8 81920000
./matrix_multiply_omp_tile 20971520000 8 81920000
./matrix_multiply_omp_tile 20971520000 8 327680000
./matrix_multiply_omp_tile 20971520000 8 327680000
 
echo "28.125 GB"
./matrix_multiply_omp_tile 30198988800 8 7372800
./matrix_multiply_omp_tile 30198988800 8 7372800
./matrix_multiply_omp_tile 30198988800 8 29491200
./matrix_multiply_omp_tile 30198988800 8 29491200
./matrix_multiply_omp_tile 30198988800 8 117964800
./matrix_multiply_omp_tile 30198988800 8 117964800
./matrix_multiply_omp_tile 30198988800 8 471859200
./matrix_multiply_omp_tile 30198988800 8 471859200

echo "Running tests for 4 Threads" 
echo "7.03 GB"
./matrix_multiply_omp_tile 7549747200 4 7372800
./matrix_multiply_omp_tile 7549747200 4 7372800
./matrix_multiply_omp_tile 7549747200 4 29491200
./matrix_multiply_omp_tile 7549747200 4 29491200

echo "12.5 GB"
./matrix_multiply_omp_tile 13421772800 4 819200 
./matrix_multiply_omp_tile 13421772800 4 819200
./matrix_multiply_omp_tile 13421772800 4 3276800
./matrix_multiply_omp_tile 13421772800 4 3276800
./matrix_multiply_omp_tile 13421772800 4 13107200
./matrix_multiply_omp_tile 13421772800 4 13107200

echo "19.53125 GB"
./matrix_multiply_omp_tile 20971520000 4 819200
./matrix_multiply_omp_tile 20971520000 4 819200
./matrix_multiply_omp_tile 20971520000 4 5120000
./matrix_multiply_omp_tile 20971520000 4 5120000
./matrix_multiply_omp_tile 20971520000 4 20480000
./matrix_multiply_omp_tile 20971520000 4 20480000
 
echo "28.125 GB"
./matrix_multiply_omp_tile 30198988800 4 3276800
./matrix_multiply_omp_tile 30198988800 4 3276800
./matrix_multiply_omp_tile 30198988800 4 7372800
./matrix_multiply_omp_tile 30198988800 4 7372800
./matrix_multiply_omp_tile 30198988800 4 29491200
./matrix_multiply_omp_tile 30198988800 4 29491200
./matrix_multiply_omp_tile 30198988800 4 117964800
./matrix_multiply_omp_tile 30198988800 4 117964800


