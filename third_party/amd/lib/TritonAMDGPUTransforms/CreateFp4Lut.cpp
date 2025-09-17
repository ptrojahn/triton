#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define DEBUG_TYPE "tritonamdgpu-create-fp4-lut"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCREATEFP4LUT
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUCreateFp4LutPass
    : public impl::TritonAMDGPUCreateFp4LutBase<TritonAMDGPUCreateFp4LutPass> {
public:
  void runOnOperation() override {
    tt::FuncOp func = getOperation();
    mlir::MLIRContext *ctx = func.getContext();
    // Find all fp4 -> bf16 conversions
    llvm::SmallVector<mlir::Operation*> converts;
    func.walk([&](ttg::Fp4ToFpOp fpOp) -> void {
      auto dataType = dyn_cast<RankedTensorType>(fpOp.getType());
      if (dataType && dataType.getElementType().isBF16()) {
        converts.push_back(fpOp);
        DBGS() << "Found Fp4ToFpOp: " << fpOp << "\n";
      }
    });

    if (converts.size() == 0)
      return;

    // Create the shared memory lookup table
    DominanceInfo domInfo(func);
    Operation *domOp = findNearestCommonDominator(converts, domInfo);
    DBGS() << "Found insertion for LUT: " << *domOp << "\n";
    
    OpBuilder builder(ctx);
    Location loc = domOp->getLoc();
    builder.setInsertionPoint(domOp);
    llvm::SmallVector<int64_t> lutShape = {256};
    ttg::BlockedEncodingAttr blockedEncoding =
        ttg::getDefaultBlockedEncoding(ctx, lutShape, ttg::lookupNumWarps(domOp),
                                   ttg::lookupThreadsPerWarp(builder), /*this->numCTAs*/1);
    RankedTensorType lutTy = RankedTensorType::get(lutShape, builder.getI32Type(), blockedEncoding);
    mlir::Attribute zeroAttr = builder.getZeroAttr(lutTy.getElementType());
    auto lutVals = builder.create<arith::ConstantOp>(loc, DenseElementsAttr::get(lutTy, zeroAttr));
    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(ctx, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto ldsEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
        ctx, 1, 1, 1, {0}, ctaLayout);
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto lutLdsType = triton::gpu::MemDescType::get(
        lutShape, builder.getI32Type(), ldsEncoding,
        sharedMemorySpace, /*mutable_memory=*/true);

    auto lut = builder.create<triton::gpu::LocalAllocOp>(loc, lutLdsType, lutVals);

    // Replace fp4_to_fp ops with lookup
    // We load two elements (8 bits) at a time, so the result value has 32 bits and needs to be split
    for (mlir::Operation* conv : converts) {
      builder.setInsertionPointAfter(conv);
      auto indexType = dyn_cast<RankedTensorType>(conv->getOperand(0).getType());
      //auto resTy = indexType.cloneWith(newShape, builder.getBF16Type());
      Attribute inEnc = indexType.getEncoding();
      Attribute outEnc;
      int axis = 0;
      SmallVector<int64_t> shape(indexType.getShape());
      shape[axis] *= 2;
      auto result = inEnc.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>()
          ->inferFp4ToFpOpEncoding(shape, axis, inEnc, outEnc,
                                   /*fwdInference=*/true, conv->getLoc());
      assert(succeeded(result));
      auto resTy = RankedTensorType::get(shape, builder.getBF16Type(), outEnc);
      auto lookupRes = builder.create<ttag::TableLookupOp>(conv->getLoc(), resTy, lut, conv->getOperand(0));
      conv->replaceAllUsesWith(lookupRes);
      conv->erase();
    }
  }
};

}
