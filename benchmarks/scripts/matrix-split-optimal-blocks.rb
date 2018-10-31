require 'yaml'

matrix_size = ARGV[0].to_i

block_size = nil
block_number = nil
tile_size = nil
tile_number = nil
# constraints:
# tile_size % (8*8) == 0
# 256 <= tile_size <= 512
# tile_size * tile_number = block_size
# block_size * block_number >= matrix_size
# 8192 <= block_size <= 16384
vector_length = 64
min_tile_size = 256
max_tile_size = 512
min_block_size = 8192
max_block_size = 16384

min_matrix_size = 8192
max_matrix_size = 16384*4

#puts "Possible tile size:"
#puts possible_tile_sizes = (256..512).step(vector_length).collect.to_a

optimal_block_sizes = YAML::load_file("optimal_block_size.yaml")

#puts "Optimal block sizes:"
#optimal_block_sizes.each { |k, v| puts "#{k} (#{v[0]})" }

optimal_matrix_splits = {}

# Matrix use the biggest possible block size
optimal_block_sizes.keys.each { |b|
  (1..Float::INFINITY).lazy.each { |i|
    candidate = b * i
    if candidate.between?(min_matrix_size, max_matrix_size)
      optimal_matrix_splits[candidate] = b
    end
    break if candidate >= max_matrix_size
  }
}
optimal_matrix_splits = optimal_matrix_splits.to_a.sort! { |(k1,_),(k2,_)| k1 <=> k2 }.to_h

matrix_block_size = {}
optimal_matrix_splits.each { |k,v|
  matrix_block_size[k] = [v, k/v, optimal_block_sizes[v][0], v/optimal_block_sizes[v][0], optimal_block_sizes[v][1].to_s, optimal_block_sizes[v][2].to_s ]
}

#puts "Optimal matrix splits:"
puts "matrix_split_t matrix_splits[#{optimal_matrix_splits.length}] = {"
puts matrix_block_size.collect { |k, v| "\t{#{k}, #{v.collect(&:inspect).join(", ")}}" }.join(",\n")
puts "};"

File::open("matrix_block_size.yaml", "w") { |f|
  f.puts "---"
  matrix_block_size.each { |k, v|
    f.puts "#{k}: [ #{v.join(", ")} ]"
  }
}
