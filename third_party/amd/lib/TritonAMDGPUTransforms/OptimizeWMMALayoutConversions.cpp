#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEWMMALAYOUTCONVERSIONS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static void loadAsMma(ttg::LocalLoadOp loadOp) {
  OpBuilder b(loadOp);
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
  for (auto &use : loadOp.getResult().getUses()) {
    scf::ForOp parentForOp = use.getOwner()->getParentOfType<scf::ForOp>();
    if (!parentForOp || parentForOp->isAncestor(loadOp)) {
      continue;
    }
    // Change target layout of the local_load to wmma and convert to dot_op right before the dot
    b.setInsertionPoint(loadOp);
    auto origTensorType = cast<RankedTensorType>(loadOp.getResult().getType());
    auto dotOpLayout = cast<mlir::triton::gpu::DotOperandEncodingAttr>(origTensorType.getEncoding());
    auto newTensorType = cast<RankedTensorType>(loadOp.getResult().getType()).cloneWithEncoding(dotOpLayout.getParent());
    auto mmaValue = b.create<ttg::LocalLoadOp>(use.getOwner()->getLoc(), newTensorType, loadOp.getOperand(0));

    // Convert back right before the use
    b.setInsertionPoint(use.getOwner());
    auto dotOpValue = b.create<ttg::ConvertLayoutOp>(use.getOwner()->getLoc(), origTensorType, mmaValue);
    use.set(dotOpValue);
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
