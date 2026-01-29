"""Int4 Matrix Multiplication Test - packed int4 format.

Uses V_WMMA_I32_16X16X32_IU4 instruction on RDNA4 which operates on packed int4
values stored in int8 containers. Each int8 contains two int4 values.
"""

import triton
import triton.language as tl
import torch


def pack_int4(values: torch.Tensor, axis: int) -> torch.Tensor:
    """Pack two int4 values into one int8 along specified axis.
    
    Low nibble from first value, high nibble from second value.
    
    Args:
        values: Tensor with int4 values stored as int8
        axis: Axis to pack along (0 for K in b, 1 for K in a)
    """
    assert values.shape[axis] % 2 == 0
    if axis == 0:  # Pack along first axis (K for b)
        # (K, N) -> (K//2, 2, N)
        grouped = values.reshape(values.shape[0] // 2, 2, *values.shape[1:])
        low = grouped[:, 0, :].to(torch.uint8) & 0xF
        high = (grouped[:, 1, :].to(torch.uint8) & 0xF) << 4
        return (low | high).view(torch.int8)
    else:  # axis == 1 or -1: Pack along last axis (K for a)
        # (M, K) -> (M, K//2, 2)
        grouped = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
        low = grouped[..., 0].to(torch.uint8) & 0xF
        high = (grouped[..., 1].to(torch.uint8) & 0xF) << 4
        return (low | high).view(torch.int8)


@triton.jit
def int4_packed_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Simple packed Int4 matmul using V_WMMA_I32_16X16X32_IU4.
    
    Single threadblock, no loop. Loads packed int8 and uses dot.
    
    For V_WMMA_I32_16X16X32_IU4:
    - M=16, N=16, K=32 (in int4 elements)
    - K=16 in packed int8 elements (32 int4 / 2)
    """
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K // 2)  # Packed: K/2 bytes
    
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c = tl.dot(a, b, out_dtype=tl.int32)
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def test_int4_packed_matmul():
    """Test 16x16 int4 matmul with packed format using V_WMMA_I32_16X16X32_IU4."""
    M, N, K = 16, 16, 32  # K=32 int4 elements = 16 packed int8 bytes
    
    torch.manual_seed(42)
    a_vals = torch.randint(-8, 8, (M, K), dtype=torch.int8, device='cuda')
    b_vals = torch.randint(-8, 8, (K, N), dtype=torch.int8, device='cuda')
    
    a_packed = pack_int4(a_vals, axis=1)  # Pack along K axis (second dim): [M, K/2] = [16, 16]
    b_packed = pack_int4(b_vals, axis=0)  # Pack along K axis (first dim): [K/2, N] = [16, 16]
    c = torch.zeros((M, N), dtype=torch.int32, device='cuda')
    
    grid = (1, 1)
    int4_packed_matmul_kernel[grid](
        a_packed, b_packed, c,
        a_packed.stride(0), a_packed.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=32,
        num_warps=1,
    )
    torch.cuda.synchronize()
    
    c_ref = torch.matmul(a_vals.cpu().to(torch.int32), b_vals.cpu().to(torch.int32)).cuda()
    
    print(f"A:\n{a_vals[:4, :4]}")
    print(f"B:\n{b_vals[:4, :4]}")
    print(f"Output:\n{c[:4, :4]}")
    print(f"Reference:\n{c_ref[:4, :4]}")
    print("✓ PASSED" if torch.equal(c, c_ref) else f"✗ FAILED (max diff: {(c-c_ref).abs().max()})")


if __name__ == "__main__":
    test_int4_packed_matmul() if torch.cuda.is_available() else print("No GPU")
