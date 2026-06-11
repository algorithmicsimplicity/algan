import taichi as ti
import numpy as np
import torch

ti.init(arch=ti.gpu)# if ti.is_arch_available(ti.gpu) else ti.cpu)

VECTOR_DIM = 6


# --- Kernel for the combination step ---
@ti.kernel
def add_offset_kernel(data: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      offset: ti.types.ndarray(dtype=ti.f32, ndim=1),
                      ):
    """
    Adds a constant vector 'offset' to every row of the 'data' tensor.
    """
    for i in range(data.shape[0]):
        for j in range(VECTOR_DIM):
            data[i, j] = blend(offset[j], data[i, j])

@ti.func
def blend(a, b):
    return blend_pytorch(a, b)

def blend_pytorch(a, b):
    return a + b

# --- Power-of-Two Scan Kernels (our existing, optimized building block) ---
@ti.func
def binary_op_inplace(data: ti.template(), idx1: int, idx2: int):
    for i in range(VECTOR_DIM):
        data[idx2, i] = blend(data[idx1, i], data[idx2, i])


@ti.kernel
def scan_power_of_two(data: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    n = data.shape[0]
    d = 0
    while (1 << d) < n:
        stride = 1 << (d + 1)
        num_iterations = n // stride
        for i in range(num_iterations):
            k = i * stride
            idx1 = k + (1 << d) - 1
            idx2 = k + (1 << (d + 1)) - 1
            binary_op_inplace(data, idx1, idx2)
        d += 1

    n = data.shape[0]
    for i in range(VECTOR_DIM):
        data[n - 1, i] = 0.0
    d = 0
    power_of_2 = 1
    while power_of_2 * 2 < n:
        power_of_2 *= 2
        d += 1
    while d >= 0:
        stride = 1 << (d + 1)
        num_iterations = n // stride
        for i in range(num_iterations):
            k = i * stride
            idx1 = k + (1 << d) - 1
            idx2 = k + (1 << (d + 1)) - 1
            temp_vec = ti.Vector([0.0] * VECTOR_DIM)
            for j in range(VECTOR_DIM): temp_vec[j] = data[idx1, j]
            for j in range(VECTOR_DIM): data[idx1, j] = data[idx2, j]
            for j in range(VECTOR_DIM): data[idx2, j] = temp_vec[j] + data[idx2, j]
        d -= 1

# --- Main Recursive Function ---
def recursive_prefix_scan(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the parallel exclusive prefix scan using a memory-efficient
    divide-and-conquer strategy.

    Args:
        input_tensor: A PyTorch tensor of shape [N, VECTOR_DIM].

    Returns:
        A new PyTorch tensor containing the scan result.
    """
    N, D = input_tensor.shape
    if N == 0:
        return torch.empty_like(input_tensor)

    # Base Case: If N is a power of two, we can scan it directly.
    is_power_of_two = (N > 0) and ((N & (N - 1)) == 0)
    if is_power_of_two:
        result = input_tensor.clone()
        scan_power_of_two(result)
        return result

    # Recursive Step: N is not a power of two.
    # 1. Find the largest power of two smaller than N.
    k = 1 << (N.bit_length() - 1)

    # 2. Split the tensor.
    head_input = input_tensor[:k]
    tail_input = input_tensor[k:]

    # 3. Recursively scan both parts.
    head_result = recursive_prefix_scan(head_input)
    tail_result = recursive_prefix_scan(tail_input)

    # 4. Calculate the total sum of the head.
    # sum = last element of exclusive scan + last element of original input.
    sum_of_head = blend_pytorch(head_result[-1], head_input[-1])

    # 5. Correct the tail result by adding the sum of the head.
    add_offset_kernel(tail_result, sum_of_head)

    # 6. Concatenate the final results.
    return torch.cat((head_result, tail_result), dim=0)


if __name__ == '__main__':
    N = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_tensor = (torch.arange(N * VECTOR_DIM, dtype=torch.float32)
                    .reshape(N, VECTOR_DIM).to(device) + 1)

    print(f"--- Recursive, Memory-Efficient Scan on {device.upper()} ---")
    print("Original Tensor (last 3 rows):\n", input_tensor[-3:, :])

    result_tensor = recursive_prefix_scan(input_tensor.clone())

    print("\nResult Tensor (last 3 rows):\n", result_tensor[-3:, :])

    # --- Verification ---
    sequential_result = torch.zeros_like(input_tensor)
    if N > 1:
        for i in range(1, N):
            sequential_result[i] = sequential_result[i - 1] + input_tensor[i - 1]

    print("\nSequential Scan for Verification (last 3 rows):\n", sequential_result[-3:, :])

    assert torch.allclose(result_tensor, sequential_result, atol=1e-5)
    print("\nResults match! The recursive strategy was successful.")