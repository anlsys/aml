require 'yaml'

_, h = YAML::load_stream(File::read("results_block_screening.yaml"))
block_res = {}
h.each do |k, vs|
  bs, c, ab, ts = k.match(/dgemm_noprefetch_gemm(\d\d\d)_task (\d) (\d) (\d+)/)[1..4].collect(&:to_i)
  new_perf = vs.collect { |_,_,_,p| p }.max
  old_perf = ( block_res[ts] ? block_res[ts][0] : 0 )
  if new_perf > old_perf
    block_res[ts] = [new_perf, c, ab, bs]
  end
end
block_res = block_res.to_a.sort { |(k1, _), (k2, _)| k1 <=> k2 }.to_h

h.each do |k, vs|
  bs, c, ab, ts = k.match(/dgemm_noprefetch_gemm(\d\d\d)_task (\d) (\d) (\d+)/)[1..4].collect(&:to_i)
  if [block_res[ts][3], 0, 0] == [bs, c, ab]
    new_perf = vs.collect { |_,_,_,p| p }.max
    block_res[ts].push new_perf
  end
end

File::open("perf_dgemm_block.csv", "w") { |f|
  f.puts "Matrix Size, Best Placement, MCDRAM c, MCDRAM ab, Block Size, DDR Placement"
  block_res.each { |k, v|
    f.puts "#{k}, #{v.join(", ")}"
  }
}

matrices = YAML::load_file("matrix_block_size.yaml")

_, hmat = YAML::load_stream(File::read("results_blocked.yaml"))

mat_res = {}
hmat.each do |k, vs|
  mat_size = k.match(/dgemm_prefetch_blocked (\d+)/)[1].to_i
  perf = vs.collect { |_,_,_,p| p }.max
  block_size = matrices[mat_size][0]
  block_perf_pinned = block_res[block_size][0]
  block_perf_ddr = block_res[block_size][4]
  mat_res[mat_size] = [perf, block_perf_pinned, block_perf_ddr]
end
mat_res


File::open("perf_dgemm.csv", "w") { |f|
  f.puts "Matrix Size, Prefetch, Single Block, Single Block DDR"
  mat_res.each { |k, v|
    f.puts "#{k}, #{v.join(", ")}"
  }
}
