"""
Int4 addition kernel test for Triton.

Uses int8 storage with pack/unpack operations for int4 values.
This is the practical approach since LLVM doesn't natively support i4.

Int4 values are packed: 2 int4 values per int8 byte
- Lower 4 bits: first value
- Upper 4 bits: second value
"""

import triton
import triton.language as tl
import torch


@triton.jit
def unpack_int4_to_int8(packed: tl.tensor) -> tuple:
    """
    Unpack int4 values from int8 storage.
    Returns (low_values, high_values) as int8 tensors.
    """
    # Extract lower 4 bits (first int4 value)
    low = packed & 0x0F
    # Extract upper 4 bits (second int4 value), shift to lower position
    high = (packed >> 4) & 0x0F
    
    # Sign extend from 4-bit to 8-bit for signed int4
    # If bit 3 is set (value >= 8), subtract 16 to get negative value
    low = tl.where(low >= 8, low - 16, low)
    high = tl.where(high >= 8, high - 16, high)
    
    return low, high


@triton.jit
def pack_int4_to_int8(low: tl.tensor, high: tl.tensor) -> tl.tensor:
    """
    Pack two int4 values into one int8.
    low goes to lower 4 bits, high goes to upper 4 bits.
    """
    # Mask to 4 bits (handles negative values)
    low_masked = low & 0x0F
    high_masked = high & 0x0F
    # Pack: low in bits 0-3, high in bits 4-7
    return low_masked | (high_masked << 4)


@triton.jit
def int4_add_kernel(
    x_ptr,      # Pointer to packed int4 input x (stored as int8)
    y_ptr,      # Pointer to packed int4 input y (stored as int8)
    output_ptr, # Pointer to packed int4 output (stored as int8)
    n_bytes,    # Number of bytes (each byte = 2 int4 values)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Add two int4 tensors stored in packed int8 format.
    Each byte contains 2 int4 values.
    Uses actual int4 arithmetic for the addition.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_bytes
    
    # Load packed int8 values
    x_packed = tl.load(x_ptr + offsets, mask=mask, other=0)
    y_packed = tl.load(y_ptr + offsets, mask=mask, other=0)
    
    # Unpack to get individual int4 values as int8
    x_low, x_high = unpack_int4_to_int8(x_packed)
    y_low, y_high = unpack_int4_to_int8(y_packed)
    
    # Cast to int4 for the actual arithmetic
    x_low_i4 = x_low.to(tl.int4)
    x_high_i4 = x_high.to(tl.int4)
    y_low_i4 = y_low.to(tl.int4)
    y_high_i4 = y_high.to(tl.int4)
    
    # Perform addition in int4
    result_low_i4 = x_low_i4 + y_low_i4
    result_high_i4 = x_high_i4 + y_high_i4
    
    # Cast back to int8 for packing
    result_low = result_low_i4.to(tl.int8)
    result_high = result_high_i4.to(tl.int8)
    
    # Pack results back to int8
    result_packed = pack_int4_to_int8(result_low, result_high)
    
    # Store packed result
    tl.store(output_ptr + offsets, result_packed, mask=mask)


@triton.jit
def int4_simple_add_kernel(
    output_ptr,
    n_bytes,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple kernel that creates int4 constants, adds them, and stores.
    Uses actual int4 arithmetic.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_bytes
    
    # Create int4 values directly
    val1 = tl.full((BLOCK_SIZE,), 3, dtype=tl.int4)  # int4 value: 3
    val2 = tl.full((BLOCK_SIZE,), 2, dtype=tl.int4)  # int4 value: 2
    
    # Add them in int4 (result is 5, within int4 range)
    result_i4 = val1 + val2
    
    # Cast to int8 for packing
    result = result_i4.to(tl.int8)
    
    # Pack two copies of the result into int8 (same value in low and high)
    packed = pack_int4_to_int8(result, result)
    
    # Store
    tl.store(output_ptr + offsets, packed, mask=mask)


def pack_int4_values(low_vals, high_vals):
    """Host-side packing of int4 values into int8."""
    low_masked = low_vals & 0x0F
    high_masked = high_vals & 0x0F
    return (low_masked | (high_masked << 4)).to(torch.int8)


def unpack_int4_values(packed):
    """Host-side unpacking of int4 values from int8."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Sign extend
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return low, high


def test_int4_simple_add():
    """Test the simple int4 add kernel."""
    n_bytes = 256
    BLOCK_SIZE = 256
    
    output = torch.zeros(n_bytes, dtype=torch.int8, device='cuda')
    
    grid = (triton.cdiv(n_bytes, BLOCK_SIZE),)
    int4_simple_add_kernel[grid](output, n_bytes, BLOCK_SIZE=BLOCK_SIZE)
    
    torch.cuda.synchronize()
    
    # Expected: 3 + 2 = 5, packed as (5 | (5 << 4)) = 0x55 = 85
    expected = (5 | (5 << 4))
    
    print(f"Output (first 10 bytes): {output[:10].tolist()}")
    print(f"Expected byte value: {expected} (0x{expected:02x})")
    
    # Unpack and verify
    low, high = unpack_int4_values(output.cpu())
    print(f"Unpacked low values: {low[:10].tolist()}")
    print(f"Unpacked high values: {high[:10].tolist()}")
    
    assert (output == expected).all(), f"Mismatch! Got {output[0].item()}, expected {expected}"
    print("✓ int4 simple add kernel passed!")


def test_int4_add():
    """Test the int4 addition kernel with packed inputs."""
    n_bytes = 256
    BLOCK_SIZE = 256
    
    # Create input values in int4 range
    x_low = torch.full((n_bytes,), 3, dtype=torch.int8, device='cuda')
    x_high = torch.full((n_bytes,), -2, dtype=torch.int8, device='cuda')
    y_low = torch.full((n_bytes,), 2, dtype=torch.int8, device='cuda')
    y_high = torch.full((n_bytes,), 4, dtype=torch.int8, device='cuda')
    
    # Pack inputs
    x = pack_int4_values(x_low, x_high).to('cuda')
    y = pack_int4_values(y_low, y_high).to('cuda')
    output = torch.zeros(n_bytes, dtype=torch.int8, device='cuda')
    
    grid = (triton.cdiv(n_bytes, BLOCK_SIZE),)
    int4_add_kernel[grid](x, y, output, n_bytes, BLOCK_SIZE=BLOCK_SIZE)
    
    torch.cuda.synchronize()
    
    # Expected results
    expected_low = 3 + 2  # = 5
    expected_high = -2 + 4  # = 2
    
    # Unpack results
    result_low, result_high = unpack_int4_values(output.cpu())
    
    print(f"Input x (first 10 bytes): {x[:10].tolist()}")
    print(f"Input y (first 10 bytes): {y[:10].tolist()}")
    print(f"Output (first 10 bytes): {output[:10].tolist()}")
    print(f"Result low values: {result_low[:10].tolist()} (expected: {expected_low})")
    print(f"Result high values: {result_high[:10].tolist()} (expected: {expected_high})")
    
    assert (result_low == expected_low).all(), f"Low mismatch!"
    assert (result_high == expected_high).all(), f"High mismatch!"
    print("✓ int4 add kernel passed!")


if __name__ == "__main__":
    print("Testing int4 operations using packed int8 storage...")
    print()
    
    if torch.cuda.is_available():
        test_int4_simple_add()
        print()
        test_int4_add()
        print()
        print("All int4 tests passed!")
    else:
        print("No GPU available, skipping kernel tests.")
