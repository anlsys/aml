set term pdf
set output "perf_dgemm_block.pdf"
set yrange [0:]
#set y2range [0:1024]
set xlabel "Matrix Size"
set ylabel "Performance (GFLOPS)"
set title "Task based DGEMM performance"
set key autotitle columnheader
set datafile separator ","
plot "perf_dgemm_block.csv" using 1:2 with lines, '' using 1:6 with lines #, '' using 1:5 with lines axis x1y2 

