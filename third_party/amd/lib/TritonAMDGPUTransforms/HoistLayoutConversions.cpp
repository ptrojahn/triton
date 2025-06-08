#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUHOISTLAYOUTCONVERSIONS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Hoist convert_layout out of the loop if the src is defined out of the loop.
// This is a heuristic driven by optimizing fused attention kernels, in which
// we want to load Q tensor and keep it in register, instead of loading it
// (neither from global or shared memory) at every iteration of the loop.
static void hoistCvtDotOpOutOfLoop(ttg::ConvertLayoutOp cvtOp) {
  OpBuilder b(cvtOp);
  //ttg::BlockedEncodingAttr::get(ctx, shape, newSizePerThread, order,
  //                                     numWarps, threadsPerWarp, numCTAs);
  // Check the dst of cvt has dotOperand layout
  RankedTensorType rtType = dyn_cast<RankedTensorType>(cvtOp.getType());
  if (!rtType)
    return;
  Attribute encoding = rtType.getEncoding();
  if (!encoding)
    return;
  if (!isa<ttg::DotOperandEncodingAttr>(encoding))
    return;
  // Check the src of cvt is defined out of the loop
  auto srcDefOp = cvtOp.getSrc().getDefiningOp();
  if (srcDefOp) {
    scf::ForOp parentForOp = cvtOp->getParentOfType<scf::ForOp>();
    if (parentForOp && !parentForOp->isAncestor(srcDefOp)) {
      cvtOp->moveAfter(srcDefOp);

      // Add an additional transformation right before the use to create the lane duplication
      auto dotOpLayoutDst = cast<mlir::triton::gpu::DotOperandEncodingAttr>(
          cast<RankedTensorType>(cvtOp.getResult().getType()).getEncoding());
      auto wmmaLayoutDst = cast<mlir::triton::gpu::AMDWmmaEncodingAttr>(dotOpLayoutDst.getParent());
      ttg::AMDWmmaEncodingAttr intermediateEncoding =
        ttg::AMDWmmaEncodingAttr::get(wmmaLayoutDst.getContext(), 1, false, wmmaLayoutDst.getWarpsPerCTA(), wmmaLayoutDst.getCTALayout());

      // Convert to intermediate format first
      auto dotOpType = cast<RankedTensorType>(cvtOp.getResult().getType());
      auto newType = dotOpType.cloneWithEncoding(intermediateEncoding);
      b.setInsertionPoint(cvtOp);
      auto cvtOpNew = b.create<ttg::ConvertLayoutOp>(cvtOp.getLoc(), newType, srcDefOp->getResult(0));

      // Add convert to the dot type for all uses
      for (auto &use : cvtOp.getResult().getUses()) {
        b.setInsertionPoint(use.getOwner());
        auto dotOpValue = b.create<ttg::ConvertLayoutOp>(use.getOwner()->getLoc(), dotOpType, cvtOpNew.getResult());
        use.set(dotOpValue);
      }
    }
  }
}

} // anonymous namespace

struct TritonAMDGPUHoistLayoutConversionsPass
    : public impl::TritonAMDGPUHoistLayoutConversionsBase<
          TritonAMDGPUHoistLayoutConversionsPass> {

  void runOnOperation() override {
    tt::FuncOp funcOp = getOperation();

    SmallVector<ttg::ConvertLayoutOp> cvtOps;
    funcOp.walk([&](ttg::ConvertLayoutOp cvtOp) { cvtOps.push_back(cvtOp); });

    for (auto cvtOp : cvtOps)
      hoistCvtDotOpOutOfLoop(cvtOp);
  }
};

} // namespace mlir
