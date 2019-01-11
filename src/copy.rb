stdin1, stdout0 = IO.pipe
stdin2, stdout1 = IO.pipe

pid1 = Process.fork {
  stdout0.close
  stdin2.close
  require 'cast'

  parser = C::Parser::new
  parser.type_names << '__builtin_va_list'
  cpp = C::Preprocessor::new
  cpp.macros['__attribute__(a)'] = ''
  cpp.macros['__restrict'] = 'restrict'
  cpp.macros['__extension__'] = ''
  cpp.macros['__asm__(a)'] = ''
  cpp.include_path << './'



  preprocessed_sources = cpp.preprocess(<<EOF).gsub(/^#.*?$/, '')
#include <stddef.h>
#include <stdint.h>
#include <aml-layout.h>
#include <aml-layout-dense.h>
#include <string.h>
#include <alloca.h>
EOF

  parser.parse(preprocessed_sources)

  ast = parser.parse(stdin1.read)
  stdin1.close

  ast.postorder { |n|
    n.stmt = n.stmt.stmts.first if n.For? && n.stmt.Block? && n.stmt.stmts.size == 1
    n.then = n.then.stmts.first if n.If? && n.then.Block? && n.then.stmts.size == 1
    n.else = n.else.stmts.first if n.If? && n.else && n.else.Block? && n.else.stmts.size == 1
  }

  stdout1.puts <<EOF
#include <aml.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>

EOF
  stdout1.puts ast
  stdout1.close
}

pid2 = Process.fork {
  stdin1.close
  stdout0.close
  stdout1.close
  require 'open3'
  Open3.popen3('indent -nbad -bap -nbc -bbo -hnl -br -brs -c33 -cd33 -ncdb -ce -ci4 -cli0 -d0 -di1 -nfc1 -i8 -ip0 -l80 -lp -npcs -nprs -npsl -sai -saf -saw -ncs -nsc -sob -nfca -cp33 -ss -ts8 -il1') do |i, o, t|
    i.write stdin2.read
    stdin2.close
    i.close
    puts o.read
  end
}

stdin1.close
stdout1.close
stdin2.close

require 'BOAST'
include BOAST

set_array_start(0)
set_lang(C)
set_default_int_size(nil)
set_output(stdout0)

register_funccall( :alloca )
register_funccall( :memcpy )
register_funccall( :assert )
register_funccall( :sizeof )

def name_prefix
  "aml_copy_"
end

def name(suffix = nil, stride: false, shuffle: false)
  name = name_prefix
  name << "sh" if shuffle
  name << "nd"
  name << "str" if stride
  name << "_#{suffix}" if suffix
  name
end

def transpose_name(reverse: false, stride: false, cumulative: false)
  name = name_prefix
  name << "r" if reverse
  name << "tnd"
  name << "str" if stride
  name << "_c" if cumulative
  name
end

def aml_compute_cumulative_pitch
  d = Sizet :d
  cumul_dst_pitch = Sizet :cumul_dst_pitch, dim: Dim(d), dir: :out
  cumul_src_pitch = Sizet :cumul_src_pitch, dim: Dim(d), dir: :out
  dst_pitch = Sizet :dst_pitch, dim: Dim(d), dir: :in
  src_pitch = Sizet :src_pitch, dim: Dim(d), dir: :in
  elem_size = Sizet :elem_size
  i = Sizet :i
  p = Procedure( :aml_compute_cumulative_pitch,
                 [ d,
                   cumul_dst_pitch, cumul_src_pitch,
                   dst_pitch, src_pitch,
                   elem_size ],
                 local: true,
                 inline: true ) {
    pr cumul_dst_pitch[0] === elem_size;
    pr cumul_src_pitch[0] === elem_size;
    
    pr For(i, 0, d - 1, operator: '<', declit: true) {
      pr cumul_dst_pitch[i + 1] === dst_pitch[i] * cumul_dst_pitch[i]
      pr cumul_src_pitch[i + 1] === src_pitch[i] * cumul_src_pitch[i]
    } 
  }
end

def aml_copy_nd_helper(stride: false, shuffle: false)
  d = Sizet :d
  target_dims = Sizet :target_dims, dim: Dim(), dir: :in
  dst = Pointer :dst, dir: :out
  cumul_dst_pitch = Sizet :cumul_dst_pitch, dim: Dim(), dir: :in
  dst_stride = Sizet :dst_stride, dim: Dim(), dir: :in
  src = Pointer :src, dir: :in
  cumul_src_pitch = Sizet :cumul_src_pitch, dim: Dim(), dir: :in
  src_stride = Sizet :src_stride, dim: Dim(), dir: :in
  elem_number = Sizet :elem_number, dim: Dim(), dir: :in
  elem_size = Sizet :elem_size
  i = Sizet :i

  args = []
  args += [ d ]
  args += [ target_dims ] if shuffle
  args += [ dst, cumul_dst_pitch ]
  args += [ dst_stride ] if stride
  args += [ src, cumul_src_pitch ]
  args += [ src_stride ] if stride
  args += [ elem_number, elem_size ]

  effective_dst_pitch = lambda { |d| cumul_dst_pitch[d] }
  effective_src_pitch = lambda { |d| cumul_src_pitch[d] }
  if stride
    tmp_dst = effective_dst_pitch
    effective_dst_pitch = lambda { |d| dst_stride[d] * tmp_dst[d] }
    tmp_src = effective_src_pitch
    effective_src_pitch = lambda { |d| src_stride[d] * tmp_src[d] }
  end

  src_index = lambda { |d| d }
  dst_index = lambda { |d| d }
  elem_index = lambda { |d| d }
  if shuffle
    elem_index = lambda { |d| target_dims[d] }
    src_index = lambda { |d| target_dims[d] }
  end

  name = name(:helper, stride: stride, shuffle: shuffle)

  p = Procedure( name,
                 args,
                 local: true,
                 inline: true ) {
    pr If( d == 1 => lambda {
      pr If( And(effective_dst_pitch[dst_index[0]] == elem_size,
                 effective_src_pitch[src_index[0]] == elem_size) => lambda {
        pr memcpy(dst, src, elem_number[elem_index[0]] * elem_size) 
      }, else: lambda {
        pr For( i, 0, elem_number[elem_index[0]], operator: '<', declit: true ) {
          pr memcpy( (dst.cast(Intptrt) + i * effective_dst_pitch[dst_index[0]]).cast(dst),
                     (src.cast(Intptrt) + i * effective_src_pitch[src_index[0]]).cast(src),
                     elem_size)
        }
      })
    }, else: lambda {
      pr For( i, 0, elem_number[elem_index[d - 1]], operator: '<', declit: true ) {
        args[0] = d - 1
        pr p.call(*args)
        pr dst === (dst.cast(Intptrt) + effective_dst_pitch[dst_index[d - 1]]).cast(dst)
        pr src === (src.cast(Intptrt) + effective_src_pitch[src_index[d - 1]]).cast(src)
      }
    })
 
  }
end

def aml_copy_nd_c(stride: false, shuffle: false)
  d = Sizet :d
  target_dims = Sizet :target_dims, dim: Dim(), dir: :in
  dst = Pointer :dst, dir: :out
  cumul_dst_pitch = Sizet :cumul_dst_pitch, dim: Dim(d), dir: :in
  dst_stride = Sizet :dst_stride, dim: Dim(d), dir: :in
  src = Pointer :src, dir: :in
  cumul_src_pitch = Sizet :cumul_src_pitch, dim: Dim(d), dir: :in
  src_stride = Sizet :src_stride, dim: Dim(d), dir: :in
  elem_number = Sizet :elem_number, dim: Dim(d), dir: :in
  elem_size = Sizet :elem_size
  i = Sizet :i
  present_dims = Sizet :present_dims

  args = []
  args += [ d ]
  args += [ target_dims ] if shuffle
  args += [ dst, cumul_dst_pitch ]
  args += [ dst_stride ] if stride
  args += [ src, cumul_src_pitch]
  args += [ src_stride ] if stride
  args += [ elem_number, elem_size]

  effective_dst_pitch = lambda { |d| cumul_dst_pitch[d] }
  effective_src_pitch = lambda { |d| cumul_src_pitch[d] }
  if stride
    tmp_dst = effective_dst_pitch
    effective_dst_pitch = lambda { |d| dst_stride[d] * tmp_dst[d] }
    tmp_src = effective_src_pitch
    effective_src_pitch = lambda { |d| src_stride[d] * tmp_src[d] }
  end

  elem_index = lambda { |d| d }
  if shuffle
    elem_index = lambda { |d| target_dims[d] }
  end

  name = name(:c, stride: stride, shuffle: shuffle)

  p = Procedure( name,
                 args,
                 return_type: Int ) {
    pr assert(d > 0)
    if shuffle
      decl present_dims
      pr present_dims === 0
      pr For(i, 0, d, operator: '<', declit: true ) {
        pr assert(target_dims[i] < d)
        get_output.puts "#{present_dims} |= 1 << #{target_dims[i]};"
      }
      pr For(i, 0, d, operator: '<', declit: true ) {
        pr assert("#{present_dims} & (1 << #{i})")
      }
    end
    pr For(i, 0, d - 1, operator: '<', declit: true ) {
      pr assert(cumul_dst_pitch[i + 1] >= effective_dst_pitch[i] * elem_number[elem_index[i]]);
      pr assert(cumul_src_pitch[i + 1] >= effective_src_pitch[i] * elem_number[i]);
    }
    pr aml_copy_nd_helper(stride: stride, shuffle: shuffle).call( *args )
    pr Return(0)
  }
end

def aml_copy_nd(stride: false, shuffle: false)
  d = Sizet :d
  target_dims = Sizet :target_dims, dim: Dim(d), dir: :in
  dst = Pointer :dst, dir: :out
  dst_pitch = Sizet :dst_pitch, dim: Dim(d), dir: :in
  dst_stride = Sizet :dst_stride, dim: Dim(d), dir: :in
  src = Pointer :src, dir: :in
  src_pitch = Sizet :src_pitch, dim: Dim(d), dir: :in
  src_stride = Sizet :src_stride, dim: Dim(d), dir: :in
  elem_number = Sizet :elem_number, dim: Dim(d), dir: :in
  elem_size = Sizet :elem_size
  cumul_dst_pitch = Pointer :cumul_dst_pitch, type: Sizet
  cumul_src_pitch = Pointer :cumul_src_pitch, type: Sizet

  args = []
  args += [ d ]
  args += [ target_dims ] if shuffle
  args += [ dst, dst_pitch ]
  args += [ dst_stride ] if stride
  args += [ src, src_pitch]
  args += [ src_stride ] if stride
  args += [ elem_number, elem_size]

  name = name(stride: stride, shuffle: shuffle)

  p = Procedure( name,
                 args,
                 return_type: Int ) {
    pr assert(d > 0);
    decl cumul_dst_pitch, cumul_src_pitch
    pr cumul_dst_pitch === alloca(d * sizeof("size_t")).cast(cumul_dst_pitch)
    pr cumul_src_pitch === alloca(d * sizeof("size_t")).cast(cumul_src_pitch)
    pr $aml_compute_cumulative_pitch.call(d, cumul_dst_pitch, cumul_src_pitch,
                                          dst_pitch, src_pitch, elem_size);
    args = []
    args += [ d ]
    args += [ target_dims ] if shuffle
    args += [ dst, cumul_dst_pitch ]
    args += [ dst_stride ] if stride
    args += [ src, cumul_src_pitch]
    args += [ src_stride ] if stride
    args +=  [ elem_number, elem_size]

    pr aml_copy_nd_c(stride: stride, shuffle: shuffle).call( *args )
    pr Return(0)
  }
end

def aml_copy_tnd(reverse: false, stride: false, cumulative: false)
  d = Sizet :d
  dst = Pointer :dst, dir: :out
  dst_pitch = Sizet :dst_pitch, dim: Dim(d), dir: :in
  cumul_dst_pitch = Sizet :cumul_dst_pitch, dim: Dim(d), dir: :in
  dst_stride = Sizet :dst_stride, dim: Dim(d), dir: :in
  src = Pointer :src, dir: :in
  src_pitch = Sizet :src_pitch, dim: Dim(d), dir: :in
  src_stride = Sizet :src_stride, dim: Dim(d), dir: :in
  cumul_src_pitch = Sizet :cumul_src_pitch, dim: Dim(d), dir: :in
  elem_number = Sizet :elem_number, dim: Dim(d), dir: :in
  elem_size = Sizet :elem_size

  args = []
  args += [ d, dst ]
  args += cumulative ? [ cumul_dst_pitch ] : [ dst_pitch ] 
  args += [ dst_stride ] if stride
  args += [ src ]
  args += cumulative ? [ cumul_src_pitch ] : [ src_pitch ]
  args += [ src_stride ] if stride
  args += [ elem_number, elem_size]

  target_dims = Sizet :target_dims, dim: Dim(d)
  i = Sizet :i

  name = transpose_name(reverse: reverse, stride: stride, cumulative: cumulative)

  p = Procedure( name,
                 args,
                 return_type: Int ) {
    pr assert(d > 0);
    decl target_dims
    pr target_dims === alloca(d * sizeof("size_t")).cast(target_dims)
    if reverse
      pr target_dims[0] === d - 1
      pr For(i, 1, d, operator: '<', declit: true) {
        pr target_dims[i] === i - 1
      }
    else
      pr target_dims[d - 1] === 0
      pr For(i, 0, d - 1, operator: '<', declit: true) {
        pr target_dims[i] === i + 1
      }
    end

    args.insert(1, target_dims)

    if cumulative
      pr aml_copy_nd_c(stride: stride, shuffle: true).call(*args)
    else
      pr aml_copy_nd(stride: stride, shuffle: true).call(*args)
    end
    pr Return(0)
  }
end

def aml_copy_layout_generic_helper(shuffle: false)
  d = Sizet :d
  dst = Pointer :dst, type: CStruct::new(type_name: :aml_layout, members: {}), dir: :inout
  src = Pointer :src, type: CStruct::new(type_name: :aml_layout, members: {}), dir: :in
  elem_number = Sizet :elem_number, dim: Dim(), dir: :in
  elem_size = Sizet :elem_size
  coords = Sizet :coords, dim: Dim(), dir: :inout
  coords_out = Sizet :coords_out, dim: Dim(), dir: :inout
  target_dims = Sizet :target_dims, dim: Dim(), dir: :in

  i = Sizet :i

  name = name_prefix + "layout_"
  name << "transform_" if shuffle
  name << "generic_helper"

  args = [d, dst, src, elem_number, elem_size, coords]
  args << coords_out << target_dims if shuffle

  src_index = lambda { |d| d }
  dst_index = lambda { |d| d }
  elem_index = lambda { |d| d }
  if shuffle
    elem_index = lambda { |d| target_dims[d] }
    src_index = lambda { |d| target_dims[d] }
  end

  coord_src = coords
  coord_dst = coords
  if shuffle
    coord_dst = coords_out
  end

  p = Procedure( name, args, local: true, inline: true ) {
    pr If( d == 1 => lambda {
      pr For( i, 0, elem_number[elem_index[0]], operator: '<', declit: true ) {
        pr coord_dst[dst_index[0]] === i
        pr coord_src[src_index[0]] === i
        pr memcpy( FuncCall(:aml_layout_aderef, dst, coord_dst), FuncCall(:aml_layout_aderef, src, coord_src), elem_size )
      }
    }, else: lambda {
      pr For( i, 0, elem_number[elem_index[d - 1]], operator: '<', declit: true ) {
        args[0] = d - 1
        pr coord_dst[dst_index[d - 1]] === i
        pr coord_src[src_index[d - 1]] === i
        pr p.call(*args)
      }
    })
  }
end

def aml_copy_layout(native: true, shuffle: false)
  dst = Pointer :dst, type: CStruct::new(type_name: :aml_layout, members: {}), dir: :inout
  src = Pointer :src, type: CStruct::new(type_name: :aml_layout, members: {}), dir: :in
  target_dims = Sizet :target_dims, dim: Dim(), dir: :in

  ddst = Pointer :ddst, type: CStruct::new(type_name: :aml_layout_data_native, members: {})
  dsrc = Pointer :dsrc, type: CStruct::new(type_name: :aml_layout_data_native, members: {})
  d = Sizet :d
  elem_size = Sizet :elem_size
  i = Sizet :i

  src_index = lambda { |d| d }
  dst_index = lambda { |d| d }
  if shuffle
    src_index = lambda { |d| target_dims[d] }
  end

  name = name_prefix + "layout_"
  name << "transform_" if shuffle
  name << (native ? "native" : "generic")

  args = [dst, src]
  args << target_dims if shuffle

  p = Procedure( name, args, return_type: Int ) {
    decl d, elem_size

    if native
      decl ddst, dsrc

      pr ddst === "(struct aml_layout_data_native *)#{dst}->data"
      pr dsrc === "(struct aml_layout_data_native *)#{src}->data"
      pr d === "#{dsrc}->ndims"
      pr assert(d > 0);

      pr elem_size === "#{dsrc}->cpitch[0]"
      pr assert(d == "#{ddst}->ndims")
      pr assert(elem_size == "#{ddst}->cpitch[0]")
      pr For(i, 0, d, operator: '<', declit: true) {
        pr assert( "#{dsrc}->dims[#{src_index[i]}] == #{ddst}->dims[#{dst_index[i]}]" )
      }

      args = []
      args += [ d ]
      args += [ target_dims ] if shuffle
      args += [ "#{ddst}->ptr", "#{ddst}->cpitch", "#{ddst}->stride",
                "#{dsrc}->ptr", "#{dsrc}->cpitch", "#{dsrc}->stride",
                "#{dsrc}->dims", elem_size ]
      pr Return(aml_copy_nd_c(stride: true, shuffle: shuffle).call(*args))
    else
      coords = Sizet :coords, dim: Dim()
      coords_out = Sizet :coords_out, dim: Dim()
      elem_number = Sizet :elem_number, dim: Dim()
      decl coords
      decl coords_out if shuffle
      decl elem_number

      pr assert( FuncCall( :aml_layout_ndims, dst ) == FuncCall( :aml_layout_ndims, src ) )
      pr d === FuncCall( :aml_layout_ndims, dst )
      pr assert( FuncCall( :aml_layout_element_size, dst ) == FuncCall( :aml_layout_element_size, src ) )
      pr elem_size === FuncCall( :aml_layout_element_size, dst )
      pr coords === alloca(d * sizeof("size_t")).cast(coords)
      pr coords_out === alloca(d * sizeof("size_t")).cast(coords_out) if shuffle
      pr elem_number === alloca(d * sizeof("size_t")).cast(elem_number)
      pr FuncCall( :aml_layout_adims, src, elem_number )

      new_args = [d, dst, src, elem_number, elem_size, coords]
      new_args << coords_out << target_dims if shuffle

      pr aml_copy_layout_generic_helper(shuffle: shuffle).call(*new_args)
      pr Return(0)
    end
  }
end

def aml_copy_layout_transpose(native: true, reverse: false)
  dst = Pointer :dst, type: CStruct::new(type_name: :aml_layout, members: {}), dir: :inout
  src = Pointer :src, type: CStruct::new(type_name: :aml_layout, members: {}), dir: :in

  dsrc = Pointer :dsrc, type: CStruct::new(type_name: :aml_layout_data_native, members: {})
  target_dims = Sizet :target_dims, dim: Dim()
  d = Sizet :d
  i = Sizet :i

  name = name_prefix + "layout_"
  name << "reverse_" if reverse
  name << "transpose_"
  name << (native ? "native" : "generic")
  p = Procedure( name, [ dst, src ], return_type: Int ) {
    decl d
    decl target_dims
    decl dsrc

    pr dsrc === "(struct aml_layout_data_native *)#{src}->data"
    pr d === "#{dsrc}->ndims"
    pr target_dims === alloca(d * sizeof("size_t")).cast(target_dims)
    if reverse
      pr target_dims[0] === d - 1
      pr For(i, 1, d, operator: '<', declit: true) {
        pr target_dims[i] === i - 1
      }
    else
      pr target_dims[d - 1] === 0
      pr For(i, 0, d - 1, operator: '<', declit: true) {
        pr target_dims[i] === i + 1
      }
    end
    pr Return( aml_copy_layout(native: native, shuffle: true).call( dst, src, target_dims) )
  }
end

pr $aml_compute_cumulative_pitch = aml_compute_cumulative_pitch

generation_space = BruteForceOptimizer::new(
  OptimizationSpace::new(
    shuffle: [false, true],
    stride: [false, true]
  )
)

transpose_generation_space = BruteForceOptimizer::new(
  OptimizationSpace::new(
    stride: [false, true],
    reverse: [false, true],
    cumulative: [false, true]
  )
)

generation_space.each { |params|
  pr aml_copy_nd_helper(**params)
  pr aml_copy_nd_c(**params)
  pr aml_copy_nd(**params)
}

transpose_generation_space.each { |params|
  pr aml_copy_tnd(**params)
}

pr aml_copy_layout
pr aml_copy_layout(shuffle: true)
pr aml_copy_layout_transpose
pr aml_copy_layout_transpose(reverse: true)

pr aml_copy_layout_generic_helper(shuffle: false)
pr aml_copy_layout_generic_helper(shuffle: true)
pr aml_copy_layout(native: false)
pr aml_copy_layout(native: false, shuffle: true)
pr aml_copy_layout_transpose(native: false)
pr aml_copy_layout_transpose(native: false, reverse: true)

stdout0.close

Process.wait(pid1)
Process.wait(pid2)

