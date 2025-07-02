#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

static void loadTranspose(tt::LoadOp loadOp) {
    IRRewriter b(loadOp);
    mlir::OpResult result = loadOp->getResult(0);
    //assert(result.hasOneUse());
    llvm::outs() << loadOp << "\n";
    mlir::Value loadedTensor = loadOp->getResult(0);
    if (!isa<RankedTensorType>(loadedTensor.getType()))
        return;
    auto loadedTensorType = cast<RankedTensorType>(loadedTensor.getType());
    auto loadedEncoding = cast<mlir::triton::gpu::BlockedEncodingAttr>(loadedTensorType.getEncoding());
    for (auto &use : result.getUses()) {
        // Check if this tensor is transposed after loading with a convert_layout
        if (isa<ttg::ConvertLayoutOp>(use.getOwner())) {
            ttg::ConvertLayoutOp conversion = cast<ttg::ConvertLayoutOp>(use.getOwner());
            auto conversionType = cast<RankedTensorType>(conversion->getResult(0).getType());
            mlir::triton::gpu::DotOperandEncodingAttr resultLayout = cast<mlir::triton::gpu::DotOperandEncodingAttr>(conversionType.getEncoding());

            if (resultLayout.getOpIdx() == 1 && loadedEncoding.getOrder()[0] == 1) {
                // Found a suboptimal load -> convert_layout chain we can replace with transposed load
                std::vector<unsigned> warpsPerCTA = {2, 2};
                auto wmmaEncodingTransposed = mlir::triton::gpu::AMDWmmaEncodingAttr::get(loadedEncoding.getContext(), 2, true, warpsPerCTA, loadedEncoding.getCTALayout());
                auto wmmaEncodingUntransposed = mlir::triton::gpu::AMDWmmaEncodingAttr::get(loadedEncoding.getContext(), 2, false, warpsPerCTA, loadedEncoding.getCTALayout());
                auto newLoadedTensorType = loadedTensorType.cloneWithEncoding(wmmaEncodingTransposed);
                auto newAddrType = cast<RankedTensorType>(loadOp.getPtr().getType()).cloneWithEncoding(wmmaEncodingUntransposed);
                // We first need to convert the addresses to the wmma format
                b.setInsertionPoint(loadOp);
                auto newPtr = b.create<ttg::ConvertLayoutOp>(loadOp->getLoc(), newAddrType, loadOp.getPtr());
                b.replaceOpWithNewOp<mlir::triton::amdgpu::LoadWarpTransposeOp>(loadOp, newLoadedTensorType, newPtr);
            }
        }
    }

    /*RankedTensorType resultType = cast<RankedTensorType>(op.getResult().getType());
    if (mlir::triton::gpu::DotOperandEncodingAttr resultLayout = cast<mlir::triton::gpu::DotOperandEncodingAttr>(resultType.getEncoding())) {
        mlir::Value localAlloc = op.getOperand(0);
        mlir::triton::LoadOp loadOp = cast<mlir::triton::LoadOp>(localAlloc.getDefiningOp()->getOperand(0).getDefiningOp());
        loadOp->dumpPretty();
        mlir::Value blockedTensor = loadOp->getResult(0);
        auto blockedTensorType = cast<RankedTensorType>(blockedTensor.getType());
        auto blockedEncoding = cast<mlir::triton::gpu::BlockedEncodingAttr>(blockedTensorType.getEncoding());
        auto order = blockedEncoding.getOrder();
        if (resultLayout.getOpIdx() == 1 && order[0] == 1) {
            llvm::outs() << "Found dot_op<opIdx = 1> with row major blocked layout! " << op << "\n";
            llvm::outs() << "loadOp: " << loadOp << "Ptr: " << loadOp.getPtr() << "\n";
            std::vector<unsigned> warpsPerCTA = {2, 2};
            auto wmmaEncodingTransposed = mlir::triton::gpu::AMDWmmaEncodingAttr::get(blockedEncoding.getContext(), 2, true, blockedEncoding.getWarpsPerCTA()warpsPerCTA, blockedEncoding.getCTALayout());
            auto newBlockedTensorType = blockedTensorType.cloneWithEncoding(wmmaEncodingTransposed);
            b.setInsertionPoint(loadOp);
            loadOp.getPtr().dump();
            b.insert
            b.replaceOpWithNewOp<mlir::triton::amdgpu::LoadWarpTransposeOp>(loadOp, newBlockedTensorType, loadOp.getPtr());
        }
    }*/
}

} // anonymous namespace

struct TritonAMDGPUConvertToTransposeLoadsPass 
    : public TritonAMDGPUConvertToTransposeLoadsBase<TritonAMDGPUConvertToTransposeLoadsPass> {

    TritonAMDGPUConvertToTransposeLoadsPass() = default;
    TritonAMDGPUConvertToTransposeLoadsPass(StringRef archGen) {
        //this->archGenerationName = archGen.data();
    };

    void runOnOperation() override {
        mlir::ModuleOp moduleOp = getOperation();

        SmallVector<tt::LoadOp> loadOps;
        moduleOp.walk([&](tt::LoadOp loadOp) { loadOps.push_back(loadOp); });

        for (auto loadOp : loadOps)
            loadTranspose(loadOp);
    }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUConvertToTransposeLoadsPass() {
  return std::make_unique<TritonAMDGPUConvertToTransposeLoadsPass>();
}