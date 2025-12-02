// RUN: ttmlir-opt --strip-non-stablehlo-attributes %s | FileCheck %s

module @SyncTensorsGraph.105 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, stablehlo.test_attr = "keep_me"} {
  // CHECK: module
  // CHECK-SAME: stablehlo.test_attr = "keep_me"
  // CHECK-NOT: mhlo.cross_program_prefetches
  // CHECK-NOT: ttcore.meshes
  func.func @main(%arg0: tensor<10xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "l__self___fc3_bias", stablehlo.arg_attr = "keep_me_too"}) -> (tensor<10xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>, stablehlo.res_attr = "keep_me_three"}) {
    // CHECK: func.func @main(%arg0: tensor<10xbf16> {stablehlo.arg_attr = "keep_me_too"}) -> (tensor<10xbf16> {stablehlo.res_attr = "keep_me_three"})
    return %arg0 : tensor<10xbf16>
  }
}
