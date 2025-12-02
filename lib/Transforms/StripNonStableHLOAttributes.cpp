// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#if TTMLIR_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_STRIPNONSTABLEHLOATTRIBUTES
#include "ttmlir/Transforms/Passes.h.inc"

namespace {
class StripNonStableHLOAttributes
    : public impl::StripNonStableHLOAttributesBase<
          StripNonStableHLOAttributes> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto keepAttr = [](NamedAttribute attr) {
      StringRef name = attr.getName().getValue();
#if TTMLIR_ENABLE_STABLEHLO
      StringRef dialectNamespace =
          mlir::stablehlo::StablehloDialect::getDialectNamespace();
      if (auto *dialect = attr.getNameDialect()) {
        return dialect->getNamespace() == dialectNamespace;
      }
#endif
      return name.starts_with("shlo.") || name.starts_with("stablehlo.");
    };

    // Strip module attributes.
    SmallVector<NamedAttribute> newModuleAttrs;
    for (auto attr : module->getAttrs()) {
      if (keepAttr(attr)) {
        newModuleAttrs.push_back(attr);
      }
    }

    module->setAttrs(DictionaryAttr::get(module.getContext(), newModuleAttrs));

    // Strip function argument and result attributes.
    module.walk([&](func::FuncOp func) {
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        SmallVector<NamedAttribute> newArgAttrs;
        if (auto argAttrs = func.getArgAttrDict(i)) {
          for (auto attr : argAttrs) {
            if (keepAttr(attr)) {
              newArgAttrs.push_back(attr);
            }
          }
        }

        func.setArgAttrs(i,
                         DictionaryAttr::get(func.getContext(), newArgAttrs));
      }
      for (unsigned i = 0; i < func.getNumResults(); ++i) {
        SmallVector<NamedAttribute> newResAttrs;
        if (auto resAttrs = func.getResultAttrDict(i)) {
          for (auto attr : resAttrs) {
            if (keepAttr(attr)) {
              newResAttrs.push_back(attr);
            }
          }
        }

        func.setResultAttrs(
            i, DictionaryAttr::get(func.getContext(), newResAttrs));
      }
    });
  }
};
} // namespace
} // namespace mlir::tt::transforms
