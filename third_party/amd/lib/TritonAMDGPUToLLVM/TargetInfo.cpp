#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include <iostream>

using mlir::triton::AMD::DppCtrl;
namespace mlir::triton::AMD {

namespace {
template <typename T>
LLVM::LLVMFuncOp getOrInsertFunction(T &moduleOp, const Location loc,
                                     RewriterBase &rewriter, StringRef name,
                                     LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

// Extend all values to 64-bit per printf call requirements.
Value printfPromoteValue(RewriterBase &rewriter, Value value) {
  auto *context = rewriter.getContext();
  auto loc = UnknownLoc::get(context);
  auto type = value.getType();

  if (isa<LLVM::LLVMPointerType>(type)) {
    // The llvm.ptrtoint op requires signless integer types.
    return ptrtoint(i64_ty, value);
  }

  assert(type.getIntOrFloatBitWidth() <= 64);

  if (auto floatType = dyn_cast<FloatType>(type)) {
    Value newValue = value;
    if (!floatType.isF64())
      newValue = fpext(f64_ty, newValue);
    return bitcast(newValue, i64_ty);
  }

  assert(type.isIntOrIndex());
  if (type.getIntOrFloatBitWidth() < 64) {
    if (type.isUnsignedInteger())
      return zext(ui64_ty, value);
    if (type.isSignedInteger())
      return sext(i64_ty, value);
    // Signless integers are printed using unsigned integer formats.
    return zext(i64_ty, value);
  }

  return value;
}
} // namespace

int TargetInfo::getSharedMemorySize() const { return 64 * 1024; }

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // On AMD hardware we don't have CTA clusters like NVIDIA. So this will always
  // be zero. Whoever calling into this should make sure the whole program does
  // not try to utilize CTA clusters.
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.ballot",
                                         type, cmp)
      ->getResult(0);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  mlir::LLVM::AMD::llStore(rewriter, loc, ptr, val, pred);
}

bool TargetInfo::canUseStMatrix(RankedTensorType tensorTy,
                                ArrayRef<unsigned> repShape,
                                ArrayRef<unsigned> paddedRepShape,
                                ArrayRef<unsigned> order,
                                int swizzleByteSize) const {
  // AMD does not support stmatrix
  return false;
}

void TargetInfo::storeMatrixShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value val) const {
  llvm::report_fatal_error("AMDGPU does not support stmatrix");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  Value falseVal = rewriter.create<LLVM::ConstantOp>(
      loc, elemTy, rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::AMD::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleXor(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::AMD::shuffleUp(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::AMD::llGetPid(loc, rewriter, moduleOp, axis);
}

// Cast and sext values into specific-length int to meet the requirements of
// instructions like UpdateDpp or readlane if necessary.
static inline Type castToAndSExtInt(RewriterBase &rewriter, Location loc,
                                    Value &val, Type fromType,
                                    unsigned toBits) {
  unsigned originalBits = fromType.getIntOrFloatBitWidth();
  Type toType = fromType;

  if (!fromType.isIntOrIndex()) {
    val = bitcast(val, int_ty(originalBits));
    toType = int_ty(originalBits);
  }

  if (originalBits < toBits) {
    val = sext(int_ty(toBits), val);
    toType = int_ty(toBits);
  }

  return toType;
}

// Trunc the value to specific length and then cast it to given type if
// necessary. This function is typically used in conjunction with
// castToAndSExtInt.
static inline Value truncAndCastFromInt(RewriterBase &rewriter, Location loc,
                                        Value val, Type valType,
                                        unsigned fromBits) {
  unsigned originalBits = valType.getIntOrFloatBitWidth();
  Value toVal = val;

  if (originalBits < fromBits) {
    toVal = trunc(int_ty(originalBits), toVal);
  }

  if (!valType.isIntOrIndex()) {
    toVal = bitcast(toVal, valType);
  }

  return toVal;
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  //std::cout << "numLaneToReduce: " << numLaneToReduce << std::endl;
  if (numLaneToReduce != 32)
    return false;

  /*if (auto family = getISAFamily();
      family != ISAFamily::CDNA3 && family != ISAFamily::CDNA2) {
    return false;
  }*/

  Operation *reduxOp = op.getSingleCombiner();
  if (!reduxOp)
    return false;

  auto createDppReduxOpWithBoundCtrl = [&](Type valType, Value &src,
                                           uint32_t dppCtrl, int rowMask,
                                           int bankMask) -> Value {
    // DPP has limited support for data types, so here we need to
    // cast non-integer types or integer types shorter than 32 bits
    // to int32, except for fp32.
    Type actualType = valType;
    if (!valType.isF32()) {
      actualType = castToAndSExtInt(rewriter, loc, src, valType, 32);
    }

    Value dppResult =
        rewriter
            .create<ROCDL::DPPUpdateOp>(loc, actualType, src, src,
                                        rewriter.getI32IntegerAttr(dppCtrl),
                                        rewriter.getI32IntegerAttr(rowMask),
                                        rewriter.getI32IntegerAttr(bankMask),
                                        rewriter.getBoolAttr(true))
            .getRes();

    if (!valType.isF32()) {
      src = truncAndCastFromInt(rewriter, loc, src, valType, 32);
      dppResult = truncAndCastFromInt(rewriter, loc, dppResult, valType, 32);
    }

    IRMapping mapping;
    mapping.map(reduxOp->getOperand(0), src);
    mapping.map(reduxOp->getOperand(1), dppResult);
    return rewriter.clone(*reduxOp, mapping)->getResult(0);
  };

  for (int i = 0; i < acc.size(); i++) {
    Value buf;
    auto valType = acc[i].getType();

    const int allRows = 0xf;
    const int allBanks = 0xf;

    // Addition over clusters of four lanes
    // A, B, C, D -> A+B, B+A, C+D, D+C
    buf = createDppReduxOpWithBoundCtrl(valType, acc[i], static_cast<uint32_t>(DppCtrl::QuadPerm1032),
                                        allRows, allBanks);

    // E, E, F, F -> E+F, E+F, F+E, F+E
    buf = createDppReduxOpWithBoundCtrl(valType, buf, static_cast<uint32_t>(DppCtrl::QuadPerm2301),
                                        allRows, allBanks);

    // Add adjacent clusters of 4 lanes
    buf = createDppReduxOpWithBoundCtrl(valType, buf, static_cast<uint32_t>(DppCtrl::RowHalfMirror),
                                        allRows, allBanks);

    // Add adjacent clusters of 8 lanes
    buf = createDppReduxOpWithBoundCtrl(valType, buf, static_cast<uint32_t>(DppCtrl::RowMirror),
                                        allRows, allBanks);

    // Similarly, we need to cast data types for permlanex16 instruction.
    Type actualType = castToAndSExtInt(rewriter, loc, buf, valType, 32);

    // Add adjacent clusters of 16 lanes by reading the last lane of the adjacent row(16 lanes) for all lanes in the current row
    // @param origValue : The original value we are going to update.
    // @param updateValue : The value to update with.
    // @param selectBitsLow : Select bits low.
    // @param selectBitsHigh : Select bits high.
    // @param fetchInactive : FI mode, whether to fetch inactive lane.
    // @param boundCtrl : Whether bound_ctrl is used or not.
    Value result = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.permlanex16", actualType, ValueRange{i32_val(0), buf, i32_val(-1), i32_val(-1), true_val(), false_val()})->getResult(0);

    result = truncAndCastFromInt(rewriter, loc, result, valType, 32);
    buf = truncAndCastFromInt(rewriter, loc, buf, valType, 32);
    // Do the final addition after the 16 element mirrow 
    IRMapping mapping;
    mapping.map(reduxOp->getOperand(0), buf);
    mapping.map(reduxOp->getOperand(1), result);
    result = rewriter.clone(*reduxOp, mapping)->getResult(0);

    // Get reduction result from lane 31
    result = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.readlane", valType, ValueRange{result, i32_val(31)})->getResult(0);

    result = truncAndCastFromInt(rewriter, loc, result, valType, 32);

    acc[i] = result;
  }

  return true;
}

void TargetInfo::printfImpl(Value formatStrStart, int formatStrByteCount,
                            ValueRange args, RewriterBase &rewriter,
                            bool useStdErr) const {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  mlir::Location loc = UnknownLoc::get(ctx);

  // See
  // https://github.com/ROCm/ROCm-Device-Libs/blob/rocm-6.0.x/ockl/src/services.cl#L263-L361
  // for details about the following HIP device print functions.
  LLVM::LLVMFuncOp printBeginFn = getOrInsertFunction(
      moduleOp, loc, rewriter,
      useStdErr ? "__ockl_fprintf_stderr_begin" : "__ockl_printf_begin",
      LLVM::LLVMFunctionType::get(i64_ty,
                                  useStdErr ? ArrayRef<Type>() : i64_ty));
  LLVM::LLVMFuncOp printStrFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
      LLVM::LLVMFunctionType::get(
          i64_ty, {i64_ty, ptr_ty(ctx), /*length=*/i64_ty, /*isLast=*/i32_ty}));
  LLVM::LLVMFuncOp printArgsFn;
  if (!args.empty()) {
    printArgsFn = getOrInsertFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_args",
        LLVM::LLVMFunctionType::get(
            i64_ty, {i64_ty, /*numArgs=*/i32_ty, i64_ty, i64_ty, i64_ty, i64_ty,
                     i64_ty, i64_ty, i64_ty, /*isLast=*/i32_ty}));
  }

  // Emit the intrinsic function call to begin the printf.
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, i64_ty, 0);
  Value message =
      call(printBeginFn, useStdErr ? ValueRange() : zeroI64).getResult();

  // Emit the intrinsic function call to handle the printf format string.
  Value oneI32 = i32_val(1);
  Value zeroI32 = i32_val(0);
  Value formatStrLen =
      rewriter.create<LLVM::ConstantOp>(loc, i64_ty, formatStrByteCount);
  SmallVector<Value, 4> arguments = {message, formatStrStart, formatStrLen,
                                     args.empty() ? oneI32 : zeroI32};
  message = call(printStrFn, arguments).getResult();

  // Emit the intrinsic function call to handle arguments iteratively.
  // We can only handle at most 7 values each time.
  constexpr size_t kArgsPerGroup = 7;
  for (size_t group = 0; group < args.size(); group += kArgsPerGroup) {
    size_t bound = std::min(group + kArgsPerGroup, args.size());
    size_t numArgs = bound - group;

    SmallVector<Value, 2 + kArgsPerGroup + 1> arguments;
    arguments.push_back(message);
    arguments.push_back(i32_val(numArgs));
    for (size_t i = group; i < bound; ++i) {
      arguments.push_back(printfPromoteValue(rewriter, args[i]));
    }
    // Pad out to 7 arguments since the function always needs 7 args.
    for (size_t extra = numArgs; extra < kArgsPerGroup; ++extra) {
      arguments.push_back(zeroI64);
    }

    Value isLast = (bound == args.size()) ? oneI32 : zeroI32;
    arguments.push_back(isLast);
    message = call(printArgsFn, arguments).getResult();
  }
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__ockl_mul_hi_u32" : "__ockl_mul_hi_u64";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args) const {
  return printfImpl(formatStrStart, formatStrByteCount, args, rewriter,
                    /*useStdError=*/false);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                        ValueRange args) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  // Compose and print an assert message.
  llvm::SmallString<256> msgBuffer;
  llvm::Twine("device assertion failed: '" + message + "', in " + func +
              " at " + file + ":" + llvm::Twine(line) + "\n\0")
      .toStringRef(msgBuffer);
  Value msgValue =
      LLVM::addStringToModule(loc, rewriter, "printfFormat_", msgBuffer);
  printfImpl(msgValue, msgBuffer.size_in_bytes(), /*args=*/ValueRange(),
             rewriter, /*useStdError=*/true);

  // Set block barrrier before aborting kernel, give a chance for all
  // the threads in a block to check/print the assert failure.
  barrier();
  // Perform the trap to abort the kernel.
  rewriter.create<LLVM::Trap>(loc);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

bool TargetInfo::supportVectorizedAtomics() const {
  // Note: not currently tested or used, but AMD generally supports vectorized
  // atomics.
  return true;
}

} // namespace mlir::triton::AMD
