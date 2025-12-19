add_mlir_extension_library(TTMLIRTransforms
        CollapseParallelLoops.cpp
        ConstEvalHoist.cpp
        EmbedCudaTargetAttributes.cpp
        ExtractGPUModules.cpp
        ReenableLostDPS.cpp
        RemoveReturnValues.cpp
        StripNonStableHLOAttributes.cpp
        SerializeToCustomCall.cpp
        TTBridge.cpp
        TTIRToStableHLOCustomCall.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
