set term pdf
set output "perf_dgemm.pdf"
set yrange [0:]
#set y2range [0:1024]
set xlabel "Matrix Size"
set ylabel "Performance (GFLOPS)"
set title "Block based DGEMM performance"
set key autotitle columnheader bottom right
set datafile separator ","
plot "perf_dgemm.csv" using 1:2 with lines, '' using 1:3 with lines, '' using 1:4 with lines
