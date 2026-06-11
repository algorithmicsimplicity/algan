import taichi as ti
import numpy as np
import torch

ti.init(arch=ti.gpu)# if ti.is_arch_available(ti.gpu) else ti.cpu)

VECTOR_DIM = 6


# --- Kernel for the combination step ---
@ti.kernel
def add_offset_kernel(data: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      offset: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      ):
    """
    Adds a constant vector 'offset' to every row of the 'data' tensor.
    """
    for i in range(data.shape[0]):
        blended = segmented_blend_over_op_taichi(ti.Vector([offset[i, 0],
                                           offset[i, 1],
                                           offset[i, 2],
                                           offset[i, 3],
                                           offset[i, 4],
                                           offset[i, 5]]), ti.Vector([data[i, 0],
                                           data[i, 1],
                                           data[i, 2],
                                           data[i, 3],
                                           data[i, 4],
                                           data[i, 5]]))
        for j in range(VECTOR_DIM):
            data[i, j] = blended[j]

@ti.func
def blend_over_op_taichi(A, B):
    """out = ti.Vector([A[0] + B[0],
                     A[1] + B[1],
                     A[2] + B[2],
                     A[3] + B[3],
                     A[4] + B[4],
                     A[5]])
    return out"""
    """Blends fragment A over fragment B."""
    na = 1-B[4]
    out = ti.Vector([A[0] * na + B[0],
                     A[1] * na + B[1],
                     A[2] * na + B[2],
                     A[3] * na + B[3],
                     A[4] * na + B[4],
                     A[5]])
    return out


@ti.func
def segmented_blend_over_op_taichi(A, B):
    """
    Associative operator for segmented blending.
    'A' is the accumulated result from the left, 'B' is the next element.
    """
    is_head_B = B[5]

    out = B
    if ti.abs(is_head_B) <= 0.5:
        # B is in the same segment as A, so we blend them.
        # The new combined fragment belongs to A's segment, so we keep is_head_A.
        out = blend_over_op_taichi(A, B)
    return out


def blend_over_op(A, B):
    """Blends fragment A over fragment B."""
    color_a, alpha_a = A
    color_b, alpha_b = B
    alpha_out = alpha_a + alpha_b * (1 - alpha_a)
    color_out = color_a + color_b * (1 - alpha_a)
    return (color_out, alpha_out)


def segmented_blend_over_op(A, B):
    """
    Associative operator for segmented blending.
    'A' is the accumulated result from the left, 'B' is the next element.
    """
    frag_A, is_head_A = A
    frag_B, is_head_B = B

    if is_head_B:
        # B is the start of a new segment, so we discard A's work.
        return B
    else:
        # B is in the same segment as A, so we blend them.
        # The new combined fragment belongs to A's segment, so we keep is_head_A.
        blended_frag = blend_over_op(frag_A, frag_B)
        return (blended_frag, is_head_A)



# --- Power-of-Two Scan Kernels (our existing, optimized building block) ---
@ti.func
def binary_op_inplace(data: ti.template(), idx1: int, idx2: int):
    blended = segmented_blend_over_op_taichi(ti.Vector([data[idx1, 0],
                                                        data[idx1, 1],
                                                        data[idx1, 2],
                                                        data[idx1, 3],
                                                        data[idx1, 4],
                                                        data[idx1, 5],]),
                                             ti.Vector([data[idx2, 0],
                                                        data[idx2, 1],
                                                        data[idx2, 2],
                                                        data[idx2, 3],
                                                        data[idx2, 4],
                                                        data[idx2, 5],]))
    for i in range(VECTOR_DIM):
        data[idx2, i] = blended[i]


@ti.kernel
def downsweep_step(data: ti.types.ndarray(dtype=ti.f32, ndim=2), num_iterations: int, stride: int, d: int):
    for i in range(num_iterations):
        k = i * stride
        idx1 = k + (1 << d) - 1
        idx2 = k + (1 << (d + 1)) - 1
        binary_op_inplace(data, idx1, idx2)

@ti.kernel
def upsweep_step(data: ti.types.ndarray(dtype=ti.f32, ndim=2), num_iterations: int, stride: int, d: int):
    for i in range(num_iterations):
        k = i * stride
        idx1 = k + (1 << d) - 1
        idx2 = k + (1 << (d + 1)) - 1
        temp_vec = ti.Vector([0.0] * VECTOR_DIM)
        temp_vec2 = ti.Vector([0.0] * VECTOR_DIM)
        for j in range(VECTOR_DIM):
            temp_vec[j] = data[idx1, j]
            temp_vec2[j] = data[idx2, j]
            data[idx1, j] = data[idx2, j]
        blended = segmented_blend_over_op_taichi(temp_vec2, temp_vec)
        for j in range(VECTOR_DIM):
            data[idx2, j] = blended[j]

def blend_packed_power_of_two_fragment_list(data: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    #data_orig = data.clone()
    n = data.shape[0]
    d = 0
    while (1 << d) < n:
        stride = 1 << (d + 1)
        num_iterations = n // stride
        downsweep_step(data, num_iterations, stride, d)
        d += 1

    n = data.shape[0]
    data[n - 1, :] = 0.0 #TODO is this necessary?
    #data[n-1, -1] = 1.0
    d = 0
    power_of_2 = 1
    while power_of_2 * 2 < n:
        power_of_2 *= 2
        d += 1
    while d >= 0:
        stride = 1 << (d + 1)
        num_iterations = n // stride
        upsweep_step(data, num_iterations, stride, d)
        d -= 1
    #add_offset_kernel(data, data_orig)

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
    sum_of_head = segmented_blend_over_op(head_result[-1], head_input[-1])

    # 5. Correct the tail result by adding the sum of the head.
    add_offset_kernel(tail_result, sum_of_head)

    # 6. Concatenate the final results.
    return torch.cat((head_result, tail_result), dim=0)


if __name__ == '__main__':
    data = torch.tensor((3,1,7,0, 4,1,6,3)).float().view(-1,1).expand(-1,6).contiguous()
    data[:,-1] = 0
    data[0, -1] = 1
    data[4,-1] = 1
    data[6, -1] = 1
    blend_packed_power_of_two_fragment_list(data)
    print(data)