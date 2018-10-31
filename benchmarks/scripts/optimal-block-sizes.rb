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

res = {}
block_res.each { |k, v|
  res[k] = [v[3], v[1], v[2]]
}

File::open("optimal_block_size.yaml", "w") { |f|
  f.puts YAML::dump(res)
}
