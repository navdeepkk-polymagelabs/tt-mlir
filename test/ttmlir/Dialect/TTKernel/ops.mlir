// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @test_tilize_uninit
func.func @test_tilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>} {
  %tilized_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>
  %untilized_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<memref<128x128xf32, #l1_>>
  ttkernel.tilize_uninit(%untilized_cb, %tilized_cb) : (!ttkernel.cb<memref<128x128xf32, #l1_>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>) -> ()
  // CHECK: ttkernel.tilize_uninit(%{{.*}}, %{{.*}}) : (!ttkernel.cb<memref<128x128xf32, #l1>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>) -> ()
  return
}

// CHECK-LABEL: func.func @test_untilize_uninit
func.func @test_untilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>} {
  %tilized_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>
  ttkernel.untilize_uninit(%tilized_cb) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>) -> ()
  // CHECK: ttkernel.untilize_uninit(%{{.*}}) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>) -> ()
  return
}

// CHECK-LABEL: func.func @test_add_binary_tile_init
func.func @test_add_binary_tile_init() -> () {
  ttkernel.add_binary_tile_init() : () -> ()
  // CHECK: ttkernel.add_binary_tile_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_add_binary_tile
func.func @test_add_binary_tile() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.add_binary_tile(%c0, %c0) : (index, index) -> ()
  // CHECK: ttkernel.add_binary_tile(%{{.*}}, %{{.*}}) : (index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_mul_binary_tile_init
func.func @test_mul_binary_tile_init() -> () {
  ttkernel.mul_binary_tile_init() : () -> ()
  // CHECK: ttkernel.mul_binary_tile_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_mul_binary_tile
func.func @test_mul_binary_tile() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.mul_binary_tile(%c0, %c0) : (index, index) -> ()
  // CHECK: ttkernel.mul_binary_tile(%{{.*}}, %{{.*}}) : (index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_sub_binary_tile_init
func.func @test_sub_binary_tile_init() -> () {
  ttkernel.sub_binary_tile_init() : () -> ()
  // CHECK: ttkernel.sub_binary_tile_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_sub_binary_tile
func.func @test_sub_binary_tile() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.sub_binary_tile(%c0, %c0) : (index, index) -> ()
  // CHECK: ttkernel.sub_binary_tile(%{{.*}}, %{{.*}}) : (index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_copy_dest_values_init
func.func @test_copy_dest_values_init() -> () {
  // CHECK: ttkernel.copy_dest_values_init() : () -> ()
  ttkernel.copy_dest_values_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_copy_dest_values
func.func @test_copy_dest_values() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.copy_dest_values(%c0, %c0) : (index, index) -> ()
  // CHECK: ttkernel.copy_dest_values(%{{.*}}, %{{.*}}) : (index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_transpose_wh_dest_init_short
func.func @test_transpose_wh_dest_init_short() -> () {
  // CHECK: ttkernel.transpose_wh_dest_init_short(true) : () -> ()
  ttkernel.transpose_wh_dest_init_short(true) : () -> ()
  return
}

// CHECK-LABEL: func.func @test_transpose_wh_dest
func.func @test_transpose_wh_dest() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.transpose_wh_dest(%c0, true) : (index) -> ()
  // CHECK: ttkernel.transpose_wh_dest(%{{.*}}, true) : (index) -> ()
  return
}

// CHECK-LABEL: func.func @test_get_absolute_logical_x
func.func @test_get_absolute_logical_x() -> (i8) {
  // CHECK: %[[X:.*]] = ttkernel.get_absolute_logical_x : () -> i8
  // CHECK: return %[[X]] : i8
  %0 = ttkernel.get_absolute_logical_x : () -> i8
  return %0 : i8
}

// CHECK-LABEL: func.func @test_get_absolute_logical_y
func.func @test_get_absolute_logical_y() -> (i8) {
  // CHECK: %[[Y:.*]] = ttkernel.get_absolute_logical_y : () -> i8
  // CHECK: return %[[Y]] : i8
  %0 = ttkernel.get_absolute_logical_y : () -> i8
  return %0 : i8
}

// CHECK-LABEL: func.func @test_unary_bcast_init
func.func @test_unary_bcast_init() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>} {
  %input = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>
  %output = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>
  ttkernel.unary_bcast_init(%input, %output, <bcast_dim_col>) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>) -> ()
  ttkernel.unary_bcast_init(%input, %output, <bcast_dim_row>) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>) -> ()
  ttkernel.unary_bcast_init(%input, %output, <bcast_dim_scalar>) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>) -> ()
  // CHECK: %[[IN:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>
  // CHECK: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>
  // CHECK: ttkernel.unary_bcast_init(%[[IN]], %[[OUT]], <bcast_dim_col>) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>) -> ()
  // CHECK: ttkernel.unary_bcast_init(%[[IN]], %[[OUT]], <bcast_dim_row>) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>) -> ()
  // CHECK: ttkernel.unary_bcast_init(%[[IN]], %[[OUT]], <bcast_dim_scalar>) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>) -> ()
  return
}

// CHECK-LABEL: func.func @test_unary_bcast
func.func @test_unary_bcast() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>} {
  %input = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
  %c0 = arith.constant 0 : index
  ttkernel.unary_bcast(%input, %c0, %c0, <bcast_dim_col>) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, index, index) -> ()
  ttkernel.unary_bcast(%input, %c0, %c0, <bcast_dim_row>) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, index, index) -> ()
  ttkernel.unary_bcast(%input, %c0, %c0, <bcast_dim_scalar>) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>, index, index) -> ()
  // CHECK:  %[[IN:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  // CHECK:  %[[INX:.*]] = arith.constant 0 : index
  // CHECK:  ttkernel.unary_bcast(%[[IN]], %[[INX]], %[[INX]], <bcast_dim_col>) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index) -> ()
  // CHECK:  ttkernel.unary_bcast(%[[IN]], %[[INX]], %[[INX]], <bcast_dim_row>) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index) -> ()
  // CHECK:  ttkernel.unary_bcast(%[[IN]], %[[INX]], %[[INX]], <bcast_dim_scalar>) : (!ttkernel.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, index, index) -> ()
  return
}
