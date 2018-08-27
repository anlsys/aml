require 'yaml'

puts YAML::dump(ENV.to_h)
puts "---"
[256, 320, 384, 448, 512].each { |t|
  (8192..16384).each { |b|
    if b % t == 0
      ["0 0", "0 1", "1 0", "1 1"].each { |mem|
        cmd = "./dgemm_noprefetch_gemm#{t}_cblas_task #{mem} #{b}"
        puts "#{cmd.gsub("./","")}:"
        10.times {  puts "  - [#{`#{cmd}`.gsub("dgemm-noprefetch: ","").gsub("\n","").split(" ").join(", ")}]" }
      }
    end
  }
}
