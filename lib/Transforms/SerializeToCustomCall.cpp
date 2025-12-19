#include "ttmlir/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Transforms/TTBridge.h"

using namespace mlir;
namespace mlir::tt::transforms {
#define GEN_PASS_DEF_SERIALIZETOCUSTOMCALL
#include "ttmlir/Transforms/Passes.h.inc"

struct TtirSerializationPattern : public RewritePattern {
  TtirSerializationPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() !=
        ttir::TTIRDialect::getDialectNamespace()) {
      return failure();
    }

    Operation *bridgeOp = mlir::tt_bridge::serializeToCustomCall(op, rewriter);

    if (!bridgeOp) {
      return failure();
    }
    rewriter.replaceOp(op, bridgeOp->getResults());
    return success();
  }
};

class SerializeToCustomCall
    : public impl::SerializeToCustomCallBase<SerializeToCustomCall> {
public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TtirSerializationPattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::tt::transforms
