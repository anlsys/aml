In order to reproduce the curves from the MCHPC Workshop rejected paper:
KNL is in quadrant with MCDRAM in flat mode, hyperthreading is off.

```bash
module load gcc/8.2.0 mkl/2018.3.222 ruby/2.5.1
./configure CC=gcc CFLAGS=-O3 -funroll-loops -g --enable-benchmarks
make
cd benchmarks
# Benchmark all block size using all possible memory placements
LD_PRELOAD=../jemalloc/lib/libjemalloc-aml.so ruby bench_script.rb > results_block_screening.yaml
# Generate optimal_block_size.yaml, selecting the best tile size and memory placement for each block size
ruby scripts/optimal-block-sizes.rb
# Generate matrix_block_size.yaml and the structure to use in dgemm_prefetch_blocked.c
ruby scripts/matrix-split-optimal-blocks.rb
# Benchmark all matrix size using the optimal block and memory placement
LD_PRELOAD=../jemalloc/lib/libjemalloc-aml.so ruby bench_script_blocked.rb > results_blocked.yaml
# Analyze results
ruby scripts/results-analysis.rb
# Plot figures, they are in perf_dgemm_block.pdf and perf_dgemm.pdf
gnuplot scripts/perf_dgemm_block.gnu
gnuplot scripts/perf_dgemm.gnu
```
