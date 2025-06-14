#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <iostream>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEWMMALAYOUTCONVERSIONS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static void loadAsMma(ttg::LocalLoadOp loadOp) {
  IRRewriter b(loadOp);
  //ttg::BlockedEncodingAttr::get(ctx, shape, newSizePerThread, order,
  //                                     numWarps, threadsPerWarp, numCTAs);
  // Check the dst of cvt has dotOperand layout
  RankedTensorType rtType = dyn_cast<RankedTensorType>(loadOp.getType());
  if (!rtType)
    return;
  Attribute encoding = rtType.getEncoding();
  if (!encoding)
    return;
  if (!isa<ttg::DotOperandEncodingAttr>(encoding))
    return;

  // Iterate over all uses and check if they are in a loop
  llvm::outs() << "getUses()\n";
  loadOp->dumpPretty();
  bool doRematerialization = false;
  for (auto &use : loadOp.getResult().getUses()) {
    llvm::outs() << "check\n";
    Operation* operation = use.getOwner();
    scf::ForOp parentForOp = operation->getParentOfType<scf::ForOp>();
    if (!parentForOp || parentForOp->isAncestor(loadOp)) {
      continue;
    }
    doRematerialization = true;
  }

  if (doRematerialization) {
    // Change target layout of the local_load to wmma and convert to dot_op right before the dot
    b.setInsertionPoint(loadOp);
    auto origTensorType = cast<RankedTensorType>(loadOp.getResult().getType());
    auto dotOpLayout = cast<mlir::triton::gpu::DotOperandEncodingAttr>(origTensorType.getEncoding());
    auto newTensorType = cast<RankedTensorType>(loadOp.getResult().getType()).cloneWithEncoding(dotOpLayout.getParent());
    auto mmaValue = b.create<ttg::LocalLoadOp>(loadOp.getLoc(), newTensorType, loadOp.getOperand(0));
    for (auto &use : llvm::make_early_inc_range(loadOp.getResult().getUses())) {
      Operation* operation = use.getOwner();
      // Convert back right before the use
      b.setInsertionPoint(use.getOwner());
      auto dotOpValue = b.create<ttg::ConvertLayoutOp>(use.getOwner()->getLoc(), origTensorType, mmaValue);
      b.modifyOpInPlace(operation, [&]() { use.set(dotOpValue); });
    }
    loadOp->erase();
  }
}

} // anonymous namespace

struct TritonAMDGPUOptimizeWMMALayoutConversionsPass 
    : public impl::TritonAMDGPUOptimizeWMMALayoutConversionsBase<TritonAMDGPUOptimizeWMMALayoutConversionsPass> {

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    SmallVector<ttg::LocalLoadOp> loadOps;
    moduleOp.walk([&](ttg::LocalLoadOp loadOp) { loadOps.push_back(loadOp); });

    for (auto loadOp : loadOps)
      loadAsMma(loadOp);
  }
};

} // namespace mlir
