#include "ttmlir/Transforms/TTBridge.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt_bridge {
static Attribute boxAttribute(Attribute attr, MLIRContext *ctx) {
  if (!attr) {
    return nullptr;
  }
  if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
    NamedAttrList cleanList;
    for (auto namedAttr : dict) {
      auto boxed = boxAttribute(namedAttr.getValue(), ctx);
      if (boxed) {
        cleanList.append(namedAttr.getName(), boxed);
      }
    }
    return cleanList.getDictionary(ctx);
  }
  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    SmallVector<Attribute> cleanElements;
    for (auto elt : array) {
      cleanElements.push_back(boxAttribute(elt, ctx));
    }
    return ArrayAttr::get(ctx, cleanElements);
  }

  bool isBuiltin = isa<IntegerAttr, FloatAttr, StringAttr, TypeAttr, UnitAttr,
                       DenseIntOrFPElementsAttr>(attr);
  if (!isBuiltin) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    attr.print(os);
    NamedAttrList boxed;
    boxed.append(kBoxedMarker, UnitAttr::get(ctx));
    boxed.append("value", StringAttr::get(ctx, os.str()));
    return boxed.getDictionary(ctx);
  }
  return attr;
}

static Attribute unboxAttribute(Attribute attr, MLIRContext *ctx) {
  if (!attr) {
    return nullptr;
  }
  auto dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    if (auto array = dyn_cast<ArrayAttr>(attr)) {
      SmallVector<Attribute> cleanElements;
      for (auto elt : array) {
        cleanElements.push_back(unboxAttribute(elt, ctx));
      }
      return ArrayAttr::get(ctx, cleanElements);
    }
    return attr;
  }
  if (dict.get(kBoxedMarker)) {
    if (auto valueAttr = dict.getAs<StringAttr>("value")) {
      Attribute parsed = mlir::parseAttribute(valueAttr.getValue(), ctx);
      if (parsed) {
        return parsed;
      }
    }
  }
  NamedAttrList unboxedList;
  for (auto namedAttr : dict) {
    auto val = unboxAttribute(namedAttr.getValue(), ctx);
    if (val) {
      unboxedList.append(namedAttr.getName(), val);
    }
  }
  return unboxedList.getDictionary(ctx);
}

static std::string convertTTIROpToTTNNOp(llvm::StringRef targetOp) {
  if (targetOp == "ttir.mesh_shard") {
    return "ttnn.mesh_shard";
  }
  if (targetOp == "ttir.all_reduce") {
    return "ttnn.all_reduce";
  }
  if (targetOp == "ttir.all_gather") {
    return "ttnn.all_gather";
  }
  if (targetOp == "ttir.reduce_scatter") {
    return "ttnn.reduce_scatter";
  }
  return targetOp.str();
}

Operation *serializeToCustomCall(Operation *op, PatternRewriter &rewriter) {
  if (op->getNumRegions() > 0) {
    return nullptr;
  }

  MLIRContext *ctx = rewriter.getContext();
  DictionaryAttr payload =
      cast<DictionaryAttr>(boxAttribute(op->getAttrDictionary(), ctx));

  // 2. Capture Metadata (Operand Counts)
  NamedAttrList metaList;
  metaList.append(kNumInputs, rewriter.getI64IntegerAttr(op->getNumOperands()));
  metaList.append(kNumResults, rewriter.getI64IntegerAttr(op->getNumResults()));

  // 3. Assemble the top-level config
  NamedAttrList config;
  config.append(kTargetOp, rewriter.getStringAttr(convertTTIROpToTTNNOp(
                               op->getName().getStringRef())));
  config.append(kMetadata, metaList.getDictionary(ctx));
  config.append(kAttrPayload, payload);

  // StableHLO 0.10.1 workaround: serialize the whole thing to one string
  std::string configStr;
  llvm::raw_string_ostream os(configStr);
  config.getDictionary(ctx).print(os);

  // return rewriter.create<stablehlo::CustomCallOp>(
  //     op->getLoc(),
  //     op->getResultTypes(),
  //     op->getOperands(),
  //     rewriter.getStringAttr(kBridgeTarget),
  //     /*has_side_effect=*/false,
  //     /*backend_config=*/rewriter.getStringAttr(os.str()),
  //     /*api_version=*/rewriter.getI32IntegerAttr(1),
  //     /*called_computations=*/rewriter.getArrayAttr({}),
  //     nullptr
  //);
  // Collapse to string for StableHLO 0.10.1
  auto versionAttr = mlir::stablehlo::CustomCallApiVersionAttr::get(
      op->getContext(),
      mlir::stablehlo::CustomCallApiVersion::API_VERSION_UNSPECIFIED);
  SmallVector<int64_t> layout({0, 1});
  auto attribute = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({static_cast<int64_t>(layout.size())},
                                  IndexType::get(ctx)),
      layout);
  auto operandsLayoutAttr = rewriter.getArrayAttr(
      SmallVector<Attribute>(op->getNumOperands(), attribute));
  auto resultsLayoutAttr = rewriter.getArrayAttr(
      SmallVector<Attribute>(op->getNumOperands(), attribute));
  return rewriter.create<stablehlo::CustomCallOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands(),
      /*call_target_name=*/rewriter.getStringAttr(kBridgeTarget),
      /*has_side_effects=*/rewriter.getBoolAttr(true),
      /*backend_config=*/rewriter.getStringAttr(os.str()),
      /*api_version=*/versionAttr,
      /*called_computations=*/rewriter.getArrayAttr({}),
      /*operand_layouts=*/operandsLayoutAttr,
      /*result_layouts=*/resultsLayoutAttr,
      /*output_operand_aliases=*/rewriter.getArrayAttr({}));
}

Operation *deserializeFromAnyCustomCall(Operation *op,
                                        PatternRewriter &rewriter) {
  auto configAttr = op->getAttr("backend_config");
  if (!configAttr) {
    return nullptr;
  }

  DictionaryAttr configDict;
  if (auto strAttr = dyn_cast<StringAttr>(configAttr)) {
    auto parsed = mlir::parseAttribute(strAttr.getValue(), op->getContext());
    configDict = dyn_cast_or_null<DictionaryAttr>(parsed);
  } else {
    configDict = dyn_cast_or_null<DictionaryAttr>(configAttr);
  }

  if (!configDict) {
    return nullptr;
  }

  auto targetName = configDict.getAs<StringAttr>(kTargetOp);
  auto payload = configDict.getAs<DictionaryAttr>(kAttrPayload);
  if (!targetName || !payload) {
    return nullptr;
  }

  Attribute unboxedAttrs = unboxAttribute(payload, op->getContext());
  if (!unboxedAttrs) {
    return nullptr;
  }

  OperationState state(op->getLoc(), targetName.getValue());
  state.addOperands(op->getOperands());
  state.addTypes(op->getResultTypes());
  state.addAttributes(cast<DictionaryAttr>(unboxedAttrs).getValue());

  return rewriter.create(state);
}

} // namespace mlir::tt_bridge
