#ifndef TT_BRIDGE_H
#define TT_BRIDGE_H
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt_bridge {

static constexpr llvm::StringRef kTargetOp = "tt.target_op";
static constexpr llvm::StringRef kAttrPayload = "tt.attr_payload";
static constexpr llvm::StringRef kMetadata = "tt.metadata";
static constexpr llvm::StringRef kNumInputs = "num_inputs";
static constexpr llvm::StringRef kNumResults = "num_results";

static constexpr llvm::StringRef kBridgeTarget = "tt_custom_dispatch";
static constexpr llvm::StringRef kBoxedMarker = "tt.boxed";

Operation *serializeToCustomCall(Operation *op, PatternRewriter &rewriter);
Operation *deserializeFromAnyCustomCall(Operation *op,
                                        PatternRewriter &rewriter);

} // namespace mlir::tt_bridge

#endif
