#include "PatternTritonGPUOpToLLVM.h"

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::AMD::upcast8xMxfp4;

namespace {

SmallVector<Value> upcastMxfp4(RewriterBase &rewriter,
                               amdgpu::UpcastMXFPOp upcastOp, bool toFp16,
                               ArrayRef<Value> values) {
  assert(values.size() % 4 == 0);
  Location loc = upcastOp.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> results;
  results.reserve(values.size() * 2);
  Type elemType = toFp16 ? f16_ty : bf16_ty;
  for (int i = 0; i < values.size(); i += 4) {
    Value v0 = values[i];
    Value v1 = values[i + 1];
    Value v2 = values[i + 2];
    Value v3 = values[i + 3];
    Value packedVec = b.undef(vec_ty(i8_ty, 4));
    packedVec = b.insert_element(packedVec, v0, b.i32_val(0));
    packedVec = b.insert_element(packedVec, v1, b.i32_val(1));
    packedVec = b.insert_element(packedVec, v2, b.i32_val(2));
    packedVec = b.insert_element(packedVec, v3, b.i32_val(3));
    SmallVector<Value, 4> v4i32 =
        upcast8xMxfp4(rewriter, upcastOp, toFp16, packedVec);
    for (int j = 0; j < 4; j++) {
      Value elements = b.bitcast(v4i32[j], vec_ty(elemType, 2));
      results.push_back(b.extract_element(elements, b.i32_val(0)));
      results.push_back(b.extract_element(elements, b.i32_val(1)));
    }
  }
  return results;
}

Value mxfpScaleFp16(RewriterBase &rewriter, Location loc, Value v, Value scale,
                    bool fastMath) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scaleF32 =
      b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  Value scaleF16 =
      LLVM::AMD::cvtFp32ToFp16RTNE_oneValue(loc, rewriter, scaleF32);
  Value mulF16 = b.fmul(v, scaleF16);
  if (fastMath)
    return mulF16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = b.icmp_eq(scale, b.i8_val(0xff));
  Value nanF16 = b.bitcast(b.i16_val(0x7c01), f16_ty);
  return b.select(scaleIsNan, nanF16, b.bitcast(mulF16, f16_ty));
};

// Scales the given bf16 v using the given scale factor without relying on bf16
// multiplication.
//
// In gfx9 architectures, we don't have bf16 VALU ops. So instead this function
// handles v * scale multiplication using fp32 VALU ops. LLVM backend can do it
// for us, just with unnecessary overheads.
Value mxfpScaleBf16ViaF32(RewriterBase &rewriter, Location loc, Value v,
                          Value scale, bool fastMath) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value c16 = b.i32_val(16);
  Value vF32 =
      b.bitcast(b.shl(b.zext(i32_ty, b.bitcast(v, i16_ty)), c16), f32_ty);
  Value scaleF32 =
      b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  Value mulF32 = b.fmul(vF32, scaleF32);
  Value mulI16 = b.trunc(i16_ty, b.lshr(b.bitcast(mulF32, i32_ty), c16));
  Value mulBf16 = b.bitcast(mulI16, bf16_ty);
  if (fastMath)
    return mulBf16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = b.icmp_eq(scale, b.i8_val(0xff));
  Value nanBf16 = b.bitcast(b.i16_val(0x7fff), bf16_ty);
  return b.select(scaleIsNan, nanBf16, mulBf16);
};

class UpcastMXFPOpPattern
    : public ConvertOpToLLVMPattern<amdgpu::UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(amdgpu::UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fpType = op.getFpType();
    bool isPacked = fpType == ScaleDotElemType::E2M1;
    if (!(isPacked || fpType == ScaleDotElemType::E4M3 ||
          fpType == ScaleDotElemType::E5M2))
      return rewriter.notifyMatchFailure(op, "NYI: non-mxfp4/mxfp8 cases");

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto scaleVals = unpackLLElements(loc, adaptor.getScale(), rewriter);
    LDBG("x: " << xVals.size() << " x " << xVals.front().getType());
    LDBG("scale: " << scaleVals.size() << " x " << scaleVals.front().getType());

    // When we lower scaled dot op, we made sure to distribute K only on one
    // warp. MXFP spec mandates 1 scale value for every 32 onsecutive values
    // along the K dimension. So in total each thread should read 32x main
    // element values.
    if (xVals.size() != scaleVals.size() * (isPacked ? 16 : 32))
      return rewriter.notifyMatchFailure(op, "unsupported problem size");

    auto dotEncoding =
        cast<DotOperandEncodingAttr>(op.getSrc().getType().getEncoding());
    auto mfmaEncoding = dyn_cast<AMDMfmaEncodingAttr>(dotEncoding.getParent());
    if (!mfmaEncoding)
      return rewriter.notifyMatchFailure(op, "NYI: non-mfma dot operand");
    LDBG("mfma: " << mfmaEncoding);

    int mDim = mfmaEncoding.getMDim();
    if (mDim != 32 && mDim != 16)
      return rewriter.notifyMatchFailure(op, "NYI: non-mfma32/16 intrinsics");

    int numThreads = lookupThreadsPerWarp(rewriter);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    bool useFp16 = op.getType().getElementType().isF16();
    if (isPacked) {
      xVals = upcastMxfp4(rewriter, op, useFp16, xVals);
    }

    // Given that MFMA layout for the A tensor arranges thread in a column-major
    // manner, for the current tid, it's at row (tid % mDim). When we set up
    // blocked layout for the A scale tensor, we made sure that it has a
    // threadsPerWarp = [M=mDim, K=64/mDim]. So the threads holding scale values
    // for the current thread starts at ((tid % mDim) * (64 / mDim)).
    Value offset =
        b.mul(b.urem(laneId, b.i32_val(mDim)), b.i32_val(numThreads / mDim));

    if (mDim == 32) {
      // One mfma32 intrinsic processes a 32x8 A tensor slice. Due to how we
      // tile, the same warp owns the whole K dim. Inside a warp, each thread
      // only holds 4 consecutive elements along K--a 1x4 vector. We need to
      // tile the warp 4 times to cover 32 values along K. So for a thread, the
      // first 4 1x4 vectors it holds shares the first scale value at row (tid %
      // mDim). the second 4 1x4 vectors shares the second scale value at row
      // (tid % mDim); and so forth.
      std::array<Value, 2> scaleThreads = {offset, b.add(offset, b.i32_val(1))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        std::array<Value, 2> si = {
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
        };

        for (int j = 0; j < 32; ++j) {
          int index = 32 * i + j;
          xVals[index] =
              useFp16 ? mxfpScaleFp16(rewriter, loc, xVals[index], si[j / 16],
                                      op.getFastMath())
                      : mxfpScaleBf16ViaF32(rewriter, loc, xVals[index],
                                            si[j / 16], op.getFastMath());
        }
      }
    } else {
      assert(mDim == 16);
      // One mfma16 intrinsic processes a 16x16 A tensor slice. Similarly, we
      // need to tile the warp 2 times to cover 32 values. So for a thread, the
      // first 2 1x4 vectors shares the first scale value at row (tid % mDim).
      std::array<Value, 4> scaleThreads = {offset, b.add(offset, b.i32_val(1)),
                                           b.add(offset, b.i32_val(2)),
                                           b.add(offset, b.i32_val(3))};

      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        auto si = std::array<Value, 4>{
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[1]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[2]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, scaleThreads[3]),
        };

        for (int j = 0; j < 32; ++j) {
          int index = 32 * i + j;
          xVals[index] = useFp16
                             ? mxfpScaleFp16(rewriter, loc, xVals[index],
                                             si[j / 8], op.getFastMath())
                             : mxfpScaleBf16ViaF32(rewriter, loc, xVals[index],
                                                   si[j / 8], op.getFastMath());
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
