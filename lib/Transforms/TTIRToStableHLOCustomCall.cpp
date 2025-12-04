#include "ttmlir/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "llvm/Support/JSON.h"

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_TTIRTOSTABLEHLOCUSTOMCALL
#include "ttmlir/Transforms/Passes.h.inc"

static llvm::json::Value convertAttrToJson(Attribute attr) {
  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    return strAttr.getValue();
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getType().isIndex()) {
      return intAttr.getInt();
    }
    if (intAttr.getType().isSignedInteger()) {
      return intAttr.getSInt();
    }
    if (intAttr.getType().isUnsignedInteger()) {
      return intAttr.getUInt();
    }
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValueAsDouble();
  }
  if (auto boolAttr = dyn_cast<BoolAttr>(attr)) {
    return boolAttr.getValue();
  }
  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    llvm::json::Array arr;
    for (auto elt : arrayAttr) {
      arr.push_back(convertAttrToJson(elt));
    }
    return std::move(arr);
  }
  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    llvm::json::Object obj;
    for (auto namedAttr : dictAttr) {
      obj[namedAttr.getName().str()] = convertAttrToJson(namedAttr.getValue());
    }
    return std::move(obj);
  }
  std::string s;
  llvm::raw_string_ostream os(s);
  attr.print(os);
  return s;
}

class TTIRToStableHLOCustomCall
    : public impl::TTIRToStableHLOCustomCallBase<TTIRToStableHLOCustomCall> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    module.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() != "ttir") {
        return;
      }

      builder.setInsertionPoint(op);

      // Convert attributes to JSON string.
      llvm::json::Object backendConfigJson;
      for (const auto &attr : op->getAttrs()) {
        backendConfigJson[attr.getName().str()] =
            convertAttrToJson(attr.getValue());
      }
      std::string backendConfigStr;
      llvm::raw_string_ostream os(backendConfigStr);
      os << llvm::json::Value(std::move(backendConfigJson));
      StringAttr backendConfig = builder.getStringAttr(backendConfigStr);

      auto versionAttr = mlir::stablehlo::CustomCallApiVersionAttr::get(
          op->getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_UNSPECIFIED);
      auto customCall = builder.create<stablehlo::CustomCallOp>(
          op->getLoc(), op->getResultTypes(), op->getOperands(),
          builder.getStringAttr(op->getName().getStringRef()),
          builder.getBoolAttr(false), backendConfig, versionAttr,
          builder.getArrayAttr({}), ArrayAttr(), ArrayAttr(),
          builder.getArrayAttr({}));

      op->replaceAllUsesWith(customCall);
      op->erase();
    });
  }
};

} // namespace mlir::tt::transforms
