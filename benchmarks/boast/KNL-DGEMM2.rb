require 'BOAST'
include BOAST

set_lang(C)
set_array_start(0)

type_map = { 4 => NArray::SFLOAT, 8 => NArray::FLOAT}

def dgemm_inner(vector_length:, super_block_size_i:, super_block_size_j:, super_block_length:, block_size_i:, block_size_j:, block_length:, tile_size_i:, tile_size_j:, unroll:, local: true)
  raise "Incompatible tile_size_i (#{tile_size_i}) and block_size_i (#{block_size_i}) and vector_length (#{vector_length})!" if block_size_i % (tile_size_i*vector_length) != 0
  raise "Incompatible tile_size_j (#{tile_size_j}) and block_size_j (#{block_size_j})!" if block_size_j % tile_size_j != 0
  raise "Incompatible block_size_i (#{block_size_i}) and vector_length (#{vector_length})!" if block_size_i % vector_length != 0
  nvec = block_size_i/vector_length
  nvec_ldb = super_block_size_i/vector_length
  nvec_ldc = super_block_size_i/vector_length
  a = Real :a, dim: [Dim(super_block_length), Dim(block_size_j)], dir: :in, restrict: true, align: 8*vector_length
  b = Real :b, vector_length: vector_length, dim: [Dim(nvec_ldb), Dim(block_length)], dir: :in, restrict: true
  c = Real :c, vector_length: vector_length, dim: [Dim(nvec_ldc), Dim(block_size_j)], dir: :inout, restrict: true
  p = Procedure( :"gemm_inn_#{block_size_i}_#{block_size_j}_#{block_length}", [a, b, c], local: local ) {
    i = Int :i
    j = Int :j
    k = Int :k
    tmp_c = tile_size_i.times.collect { |k|
      tile_size_j.times.collect { |l|
        Real(:"tmp_#{k}_#{l}", vector_length: vector_length)
      }
    }
    tmp_a = tile_size_j.times.collect { |l|
      Real :"tmpa_#{l}", vector_length: vector_length
    }
    tmp_b = tile_size_i.times.collect { |l|
      Real :"tmpb_#{l}", vector_length: vector_length
    }
    
    decl i, j, k, *(tmp_c.flatten), *tmp_a, *tmp_b

    p_inner_reverse = lambda { |k|
      loaded = {}
      #get_output.puts "_mm_prefetch(&#{b[i, k+1]},_MM_HINT_T0);"
      tile_size_j.times { |m|
        pr tmp_a[m].set(a[k, j+m])
        tile_size_i.times { |l|
          unless loaded[l]
            loaded[l] = true
            pr tmp_b[l] === b[i+l, k]
          end
          pr tmp_c[l][m] === FMA(tmp_a[m], tmp_b[l], tmp_c[l][m])
        }
      }
    }

    p_inner = lambda { |k|
      loaded = {}
      #get_output.puts "_mm_prefetch(&#{b[i, k+1]},_MM_HINT_T0);"
      tile_size_i.times { |l|
        pr tmp_b[l] === b[i+l, k]
        tile_size_j.times { |m|
          unless loaded[m]
            loaded[m] = true
            pr tmp_a[m].set(a[k, j+m])
          end
          pr tmp_c[l][m] === FMA(tmp_a[m], tmp_b[l], tmp_c[l][m])
        }
      }
    }

    p_inn = lambda {
      pr For( k, 0, block_length - 1, step: unroll ) {
        unroll.times { |l|
          p_inner.call(k+l)
        }
      }
    }

    pr For(j, 0, block_size_j - 1, step: tile_size_j) {
      pr For(i, 0, nvec - 1, step: tile_size_i) {
        tile_size_j.times { |m|
          tile_size_i.times { |l|
            pr tmp_c[l][m] === c[i+l,j+m]
          }
        }
        p_inn.call
        tile_size_j.times { |m|
          tile_size_i.times { |l|
            pr c[i+l,j+m] === tmp_c[l][m]
          }
        }
      }
    }
  }
  return p
end

def get_borders(vector_length:, super_block_size_i:, super_block_size_j:, super_block_length:, block_size_i:, block_size_j:, block_length:, tile_size_i:, tile_size_j:, unroll:)

  remainder_i = super_block_size_i % block_size_i
  remainder_j = super_block_size_j % block_size_j
  remainder_length = super_block_length % block_length

  ts_remainder_i = tile_size_i
  ts_remainder_j = tile_size_j
  unroll_remainder = unroll

  loop do
    break if remainder_i % (ts_remainder_i*vector_length) == 0
    ts_remainder_i -= 1
  end

  loop do
    break if remainder_j % ts_remainder_j == 0
    ts_remainder_j -= 1
  end

  loop do
    break if remainder_length % unroll_remainder == 0
    unroll_remainder -= 1
  end

  return remainder_i, remainder_j, remainder_length, ts_remainder_i, ts_remainder_j, unroll_remainder
end

def get_sub_ops(**opts)

  remainder_i, remainder_j, remainder_length, ts_remainder_i, ts_remainder_j, unroll_remainder = get_borders(**opts)

  ps = {}
  [false, true].product([false, true],[false, true]).each { |a|
    sub_opts = opts.dup
    if a[0]
      sub_opts[:block_size_i] = remainder_i
      sub_opts[:tile_size_i] = ts_remainder_i
    end
    if a[1]
      sub_opts[:block_size_j] = remainder_j
      sub_opts[:tile_size_j] = ts_remainder_j
    end
    if a[2]
      sub_opts[:block_length] = remainder_length
      sub_opts[:unroll] = unroll_remainder
    end
    ps[a] = dgemm_inner(**sub_opts)
  }

  ps
end


def dgemm(**opts)
  vector_length = opts.fetch(:vector_length)
  super_block_size_i = opts.fetch(:super_block_size_i)
  super_block_size_j = opts.fetch(:super_block_size_j)
  super_block_length = opts.fetch(:super_block_length)
  block_size_i = opts.fetch(:block_size_i)
  block_size_j = opts.fetch(:block_size_j)
  block_length = opts.fetch(:block_length)
  tile_size_i = opts.fetch(:tile_size_i)
  tile_size_j = opts.fetch(:tile_size_j)
  unroll = opts.fetch(:unroll)
  raise "Incompatible super_block_size_i (#{super_block_size_i}) and vector_length (#{vector_length})!" if super_block_size_i % vector_length != 0

  if super_block_size_i == block_size_i && super_block_size_j == block_size_j && super_block_length == block_length
    p = dgemm_inner(local: false, **opts)
    return p.ckernel(:includes => "immintrin.h")
  end

  ps = get_sub_ops(**opts)

  nvec = block_size_i/vector_length
  nvec_ldb = super_block_size_i/vector_length
  nvec_ldc = super_block_size_i/vector_length
  
  remainder_i, remainder_j, remainder_length, ts_remainder_i, ts_remainder_j, unroll_remainder = get_borders(**opts)

  iter_space_i = (0...super_block_size_i).step(block_size_i).to_a.collect { |v| [v/block_size_i, false] }
  iter_space_i.last[1] = true if remainder_i != 0
  iter_space_j = (0...super_block_size_j).step(block_size_j).to_a.collect { |v| [v/block_size_j, false] }
  iter_space_j.last[1] = true if remainder_j != 0
  iter_space_k = (0...super_block_length).step(block_length).to_a.collect { |v| [v/block_length, false] }
  iter_space_k.last[1] = true if remainder_length != 0

  a = Real :a, dim: [Dim(super_block_length), Dim(super_block_size_j)], dir: :in, restrict: true, align: 8*vector_length
  b = Real :b, vector_length: vector_length, dim: [Dim(nvec_ldb), Dim(super_block_length)], dir: :in, restrict: true
  c = Real :c, vector_length: vector_length, dim: [Dim(nvec_ldc), Dim(super_block_size_j)], dir: :inout, restrict: true
  used_procs = []
  p2 = Procedure( :"gemm_#{super_block_size_i}_#{super_block_size_j}_#{super_block_length}", [a, b, c]) {
        iter_space_k.each { |k, kb|
      iter_space_j.each { |j, jb|
    iter_space_i.each { |i, ib|
          pr ps[[ib,jb,kb]].call(a[k*block_length, j*block_size_j].address , b[nvec*i, k*block_length].address,  c[nvec*i, j*block_size_j].address)
        }
      }
    }
  }

  iter_space_i.each { |i, ib|
    iter_space_j.each { |j, jb|
      iter_space_k.each { |k, kb|
        used_procs.push ps[[ib,jb,kb]]
      }
    }
  }

  k = CKernel::new(:includes => "immintrin.h") {
    used_procs.uniq.each { |p|
      pr p
    }
    pr p2
  }
  k.procedure = p2
  k
end

opts = {
  vector_length: 4,
  super_block_size_i: 128,
  super_block_size_j: 128,
  super_block_length: 128,
  block_size_i: 64,#64
  block_size_j: 48,#48
  block_length: 128,#256
  tile_size_i: 4,
  tile_size_j: 3,
  unroll: 1
}

a = NMatrix::new( type_map[get_default_real_size], opts[:super_block_length], opts[:super_block_size_j] ).randomn!
b = NMatrix::new( type_map[get_default_real_size], opts[:super_block_size_i], opts[:super_block_length] ).randomn!

c_ref = a * b

a_in = ANArray::new( type_map[get_default_real_size], get_default_real_size*8, opts[:super_block_length], opts[:super_block_size_j] )
b_in = ANArray::new( type_map[get_default_real_size], get_default_real_size*8, opts[:super_block_size_i], opts[:super_block_length] )
c_in = ANArray::new( type_map[get_default_real_size], get_default_real_size*8, opts[:super_block_size_i], opts[:super_block_size_j] )

a_in[true, true] = a[true, true]
b_in[true, true] = b[true, true]

puts k = dgemm(**opts)
k.build(CFLAGS: "-O3 -funroll-loops")
k.run(a_in, b_in, c_in)
opts[:super_block_size_i].times { |i|
  opts[:super_block_size_j].times { |j|
    #puts "#{i} #{j} #{c_in[i, j]} #{c_ref[i, j]}"
    c_in[i, j] -= c_ref[i, j]
  }
}
if c_in.abs.max > 1e-7
  puts "Computation error!"
end
repeat = 10
res = 1000.times.collect { k.run(a_in, b_in, c_in, repeat: repeat) }
best = res.min { |r1, r2|
  r1[:duration] <=> r2[:duration]
}
p best
puts "#{opts[:super_block_length]*opts[:super_block_size_i]*opts[:super_block_size_j]*2*repeat/(best[:duration] * 1e9)} GFlops"
Dir.mkdir("gemm") unless Dir.exist?("gemm")
Dir.mkdir("gemm/sources") unless Dir.exist?("gemm/sources")
k.dump_binary("gemm/#{k.procedure.name}.o")
k.dump_source("gemm/sources/#{k.procedure.name}.c")
