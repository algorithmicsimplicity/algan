import torch
import math  # For math.pi

# Epsilon for floating point comparisons
F32_EPSILON_IMAG_ZERO = 1e-4
F64_EPSILON_IMAG_ZERO = 1e-12

# Adjusted F32_COEFF_EPSILON to be > 1.1921e-07
F32_COEFF_EPSILON = 1e-5  # Was 1e-7. Max observed A for TC2 was ~1.19e-7.
F64_COEFF_EPSILON = 1e-12

F32_CLAMP_EPSILON = 1e-5
F64_CLAMP_EPSILON = 1e-15
F32_QUAD_DELTA_EPSILON = 1e-5
F64_QUAD_DELTA_EPSILON = 1e-12


def scuffed_abs(z):
    return (z.real**2 + z.imag**2)**0.5


def _solve_batched_cubic_pytorch(A_real, B_real, C_real, D_real, device, complex_dtype, real_dtype):
    is_tc2_debug = False
    # Simplified TC2 debug check slightly
    if A_real.numel() == 1 and \
            torch.isclose(A_real, torch.tensor([0.0], dtype=real_dtype, device=device), atol=2e-7) and \
            torch.isclose(B_real, torch.tensor([-4.0], dtype=real_dtype, device=device), atol=1e-5):
        is_tc2_debug = True
        # print("\n[DEBUG TC2] Solver activated for TC2-like coeffs")
        # print(f"[DEBUG TC2] A_real: {A_real.item()}, B_real: {B_real.item()}, C_real: {C_real.item()}, D_real: {D_real.item()}")

    if real_dtype == torch.float32:
        COEFF_EPS = F32_COEFF_EPSILON
        CLAMP_EPS = F32_CLAMP_EPSILON
        QUAD_DELTA_EPS = F32_QUAD_DELTA_EPSILON
    else:
        COEFF_EPS = F64_COEFF_EPSILON
        CLAMP_EPS = F64_CLAMP_EPSILON
        QUAD_DELTA_EPS = F64_QUAD_DELTA_EPSILON

    A = A_real.to(dtype=real_dtype, device=device)
    B = B_real.to(dtype=real_dtype, device=device)
    C = C_real.to(dtype=real_dtype, device=device)
    D = D_real.to(dtype=real_dtype, device=device)

    num_eqs = A.shape[0]
    roots = torch.full((num_eqs, 3), torch.nan + torch.nan * 1j, dtype=complex_dtype, device=device)

    # Path for A is effectively zero
    a_is_zero_mask = A.abs() < COEFF_EPS

    # Path for A is non-zero (cubic)
    a_is_not_zero_mask = ~a_is_zero_mask  # Equivalent to A.abs() >= COEFF_EPS

    # --- Handle A is effectively zero (Linear or Quadratic) ---
    if torch.any(a_is_zero_mask):
        A_sub, B_sub, C_sub, D_sub = A[a_is_zero_mask], B[a_is_zero_mask], C[a_is_zero_mask], D[a_is_zero_mask]
        # Create a sub-roots tensor for this branch
        sub_roots = torch.full((A_sub.shape[0], 3), torch.nan + torch.nan * 1j, dtype=complex_dtype, device=device)

        # Linear: B is also zero
        lin_mask_sub = B_sub.abs() < COEFF_EPS
        if torch.any(lin_mask_sub):
            # C_lin must be non-zero for a solution
            C_lin, D_lin = C_sub[lin_mask_sub], D_sub[lin_mask_sub]
            solve_lin_mask = C_lin.abs() >= COEFF_EPS

            if is_tc2_debug and torch.any(
                a_is_zero_mask[is_tc2_debug_idx(A_real)] & lin_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]): print(
                "[DEBUG TC2] In Linear Mask")

            if torch.any(solve_lin_mask):
                sub_roots[lin_mask_sub.nonzero(as_tuple=True)[0][solve_lin_mask], 0] = (
                            -D_lin[solve_lin_mask] / C_lin[solve_lin_mask]).to(complex_dtype)

        # Quadratic: B is non-zero
        quad_mask_sub = B_sub.abs() >= COEFF_EPS
        if torch.any(quad_mask_sub):
            if is_tc2_debug and torch.any(
                a_is_zero_mask[is_tc2_debug_idx(A_real)] & quad_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]): print(
                "[DEBUG TC2] In Quadratic Mask")

            B_q, C_q, D_q = B_sub[quad_mask_sub], C_sub[quad_mask_sub], D_sub[quad_mask_sub]
            if is_tc2_debug and torch.any(
                    a_is_zero_mask[is_tc2_debug_idx(A_real)] & quad_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]):
                print(f"[DEBUG TC2] B_q: {B_q.item()}, C_q: {C_q.item()}, D_q: {D_q.item()}")

            delta_q_real = C_q ** 2 - 4 * B_q * D_q
            if is_tc2_debug and torch.any(
                a_is_zero_mask[is_tc2_debug_idx(A_real)] & quad_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]): print(
                f"[DEBUG TC2] delta_q_real (before zeroing): {delta_q_real.item()}")

            delta_q_real_zeroed_check = delta_q_real.abs() < QUAD_DELTA_EPS
            if is_tc2_debug and torch.any(
                a_is_zero_mask[is_tc2_debug_idx(A_real)] & quad_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]): print(
                f"[DEBUG TC2] delta_q_real.abs() < QUAD_DELTA_EPS ({QUAD_DELTA_EPS}): {delta_q_real_zeroed_check.item()}")

            delta_q_real = torch.where(delta_q_real_zeroed_check, torch.zeros_like(delta_q_real), delta_q_real)
            if is_tc2_debug and torch.any(
                a_is_zero_mask[is_tc2_debug_idx(A_real)] & quad_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]): print(
                f"[DEBUG TC2] delta_q_real (after zeroing): {delta_q_real.item()}")

            sqrt_delta_q_cplx = torch.sqrt(delta_q_real.to(complex_dtype))
            if is_tc2_debug and torch.any(
                a_is_zero_mask[is_tc2_debug_idx(A_real)] & quad_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]): print(
                f"[DEBUG TC2] sqrt_delta_q_cplx: {sqrt_delta_q_cplx.item()}")

            C_q_cplx, B_q_cplx = C_q.to(complex_dtype), B_q.to(complex_dtype)
            denom = 2 * B_q_cplx
            denom_safe = denom + (denom.abs() < COEFF_EPS) * COEFF_EPS * (
                        1 + 1j)  # Avoid div by zero if B_q_cplx is zero

            roots_q1 = (-C_q_cplx + sqrt_delta_q_cplx) / denom_safe
            roots_q2 = (-C_q_cplx - sqrt_delta_q_cplx) / denom_safe
            if is_tc2_debug and torch.any(
                a_is_zero_mask[is_tc2_debug_idx(A_real)] & quad_mask_sub[is_tc2_debug_idx(B[a_is_zero_mask])]): print(
                f"[DEBUG TC2] roots_q1: {roots_q1.item()}, roots_q2: {roots_q2.item()}")

            # Assign to the correct rows in sub_roots
            quad_indices_in_sub = quad_mask_sub.nonzero(as_tuple=True)[0]
            sub_roots[quad_indices_in_sub, 0] = roots_q1
            sub_roots[quad_indices_in_sub, 1] = roots_q2

        roots[a_is_zero_mask] = sub_roots  # Place results back into main roots tensor

    # --- Handle A is non-zero (Cubic) ---
    if torch.any(a_is_not_zero_mask):
        if is_tc2_debug and torch.any(a_is_not_zero_mask[is_tc2_debug_idx(A_real)]): print(
            "[DEBUG TC2] In Cubic Mask (ERROR if TC2)")

        # Select only cubic cases for processing
        a_r, b_r, c_r, d_r = A[a_is_not_zero_mask], B[a_is_not_zero_mask], C[a_is_not_zero_mask], D[a_is_not_zero_mask]

        # Create a sub-roots tensor for this branch
        sub_roots_cubic = torch.full((a_r.shape[0], 3), torch.nan + torch.nan * 1j, dtype=complex_dtype, device=device)

        p_r = (3 * a_r * c_r - b_r ** 2) / (3 * a_r ** 2)
        q_cubic_r = (2 * b_r ** 3 - 9 * a_r * b_r * c_r + 27 * a_r ** 2 * d_r) / (27 * a_r ** 3)
        offset_r = -b_r / (3 * a_r)

        delta_cubic_r = (q_cubic_r / 2) ** 2 + (p_r / 3) ** 3

        # original_cubic_indices = torch.where(a_is_not_zero_mask)[0] # Not needed if we assign back with a_is_not_zero_mask

        three_real_mask_sub = delta_cubic_r < -COEFF_EPS  # Mask relative to the 'cubic' subset (a_r, b_r, ...)
        if torch.any(three_real_mask_sub):
            p_3r, q_3r, offset_3r = p_r[three_real_mask_sub], q_cubic_r[three_real_mask_sub], offset_r[
                three_real_mask_sub]

            acos_arg_val = (3 * q_3r / (2 * p_3r)) * torch.sqrt(-3 / p_3r)
            acos_arg_clamped = torch.clamp(acos_arg_val, -1.0 + CLAMP_EPS, 1.0 - CLAMP_EPS)
            phi = torch.acos(acos_arg_clamped)
            term_coeff = 2 * torch.sqrt(-p_3r / 3)

            x1_r = term_coeff * torch.cos(phi / 3)
            x2_r = term_coeff * torch.cos((phi + 2 * math.pi) / 3)
            x3_r = term_coeff * torch.cos((phi - 2 * math.pi) / 3)

            # Assign to correct rows in sub_roots_cubic
            idx_3r_in_sub = three_real_mask_sub.nonzero(as_tuple=True)[0]
            sub_roots_cubic[idx_3r_in_sub, 0] = (x1_r + offset_3r).to(complex_dtype)
            sub_roots_cubic[idx_3r_in_sub, 1] = (x2_r + offset_3r).to(complex_dtype)
            sub_roots_cubic[idx_3r_in_sub, 2] = (x3_r + offset_3r).to(complex_dtype)

        one_real_multi_mask_sub = delta_cubic_r >= -COEFF_EPS  # Mask relative to 'cubic' subset
        if torch.any(one_real_multi_mask_sub):
            p_s, q_s, delta_s, offset_s = p_r[one_real_multi_mask_sub], q_cubic_r[one_real_multi_mask_sub], \
            delta_cubic_r[one_real_multi_mask_sub], offset_r[one_real_multi_mask_sub]

            p_c, q_c, delta_c, offset_c = p_s.to(complex_dtype), q_s.to(complex_dtype), delta_s.to(
                complex_dtype), offset_s.to(complex_dtype)
            sqrt_delta_val = torch.sqrt(delta_c)
            S_cubed = -q_c / 2 + sqrt_delta_val
            S_val = torch.pow(S_cubed, 1 / 3.0)
            T_val_temp_cubed = -q_c / 2 - sqrt_delta_val
            T_val_principal = torch.pow(T_val_temp_cubed, 1 / 3.0)

            S_val_is_zero = S_val.abs() < COEFF_EPS
            p_c_is_zero = p_c.abs() < COEFF_EPS
            denom_s_val = S_val + (S_val_is_zero * COEFF_EPS * (1 + 1j))
            T_val_corrected = torch.where(S_val_is_zero | p_c_is_zero, T_val_principal, -p_c / (3 * denom_s_val))

            omega = torch.tensor(-0.5 + (math.sqrt(3) / 2) * 1j, dtype=complex_dtype, device=device)
            omega_sq = omega.conj()

            x1_c = S_val + T_val_corrected
            x2_c = S_val * omega + T_val_corrected * omega_sq
            x3_c = S_val * omega_sq + T_val_corrected * omega

            # Assign to correct rows in sub_roots_cubic
            idx_1r_in_sub = one_real_multi_mask_sub.nonzero(as_tuple=True)[0]
            sub_roots_cubic[idx_1r_in_sub, 0] = x1_c + offset_c
            sub_roots_cubic[idx_1r_in_sub, 1] = x2_c + offset_c
            sub_roots_cubic[idx_1r_in_sub, 2] = x3_c + offset_c

        roots[a_is_not_zero_mask] = sub_roots_cubic  # Place results back into main roots tensor

    return roots


# Helper for TC2 debug indexing if needed (not strictly necessary with current simplified check)
def is_tc2_debug_idx(tensor_being_indexed):
    # This function is just to make the debug print conditions less verbose if we were indexing into sub-tensors
    # For now, the A_real.numel()==1 check in is_tc2_debug means we can use .item()
    return slice(None)  # effectively no change for single item tensors


# --- count_line_bezier_intersections and main remain the same as previous correct version ---
def count_line_bezier_intersections(lines_b, lines_w, bezier_cps):
    device = lines_b.device
    real_dtype = lines_b.dtype
    if real_dtype == torch.float64:
        complex_dtype = torch.complex128
        EPSILON_IMAG = F64_EPSILON_IMAG_ZERO
        COEFF_EPS_MAIN = F64_COEFF_EPSILON  # Epsilon for this function's logic
    else:
        complex_dtype = torch.complex64
        EPSILON_IMAG = F32_EPSILON_IMAG_ZERO
        COEFF_EPS_MAIN = F32_COEFF_EPSILON

    b_exp = lines_b.unsqueeze(-2)
    w_exp = lines_w.unsqueeze(-2)
    P_exp = bezier_cps.unsqueeze(-3)

    P_minus_b = P_exp - b_exp
    d = torch.sum(P_minus_b * w_exp, dim=-1)

    d0, d1, d2, d3 = d[..., 0], d[..., 1], d[..., 2], d[..., 3]

    A = d3 - 3 * d2 + 3 * d1 - d0
    B = 3 * d2 - 6 * d1 + 3 * d0
    C = 3 * d1 - 3 * d0
    D = d0

    # For TC2 debug:
    if lines_b.shape[0] == 1 and tc2_lines_b is not None and torch.allclose(lines_b,
                                                                            tc2_lines_b):  # Check if tc2_lines_b is defined
        print(f"\n[DEBUG TC2 Main] A: {A.item():.9e}, B: {B.item():.9e}, C: {C.item():.9e}, D: {D.item():.9e}")

    A_flat, B_flat, C_flat, D_flat = A.reshape(-1), B.reshape(-1), C.reshape(-1), D.reshape(-1)

    is_on_line_flat = (A_flat.abs() < COEFF_EPS_MAIN) & \
                      (B_flat.abs() < COEFF_EPS_MAIN) & \
                      (C_flat.abs() < COEFF_EPS_MAIN) & \
                      (D_flat.abs() < COEFF_EPS_MAIN)

    has_intersection_flat = torch.zeros_like(A_flat, dtype=torch.bool)
    has_intersection_flat[is_on_line_flat] = True

    solvable_mask_flat = ~is_on_line_flat
    num_solvable = torch.sum(solvable_mask_flat)

    A_s, B_s, C_s, D_s = (A_flat[solvable_mask_flat], B_flat[solvable_mask_flat],
                          C_flat[solvable_mask_flat], D_flat[solvable_mask_flat])

    roots_s_cplx = _solve_batched_cubic_pytorch(A_s, B_s, C_s, D_s, device, complex_dtype, real_dtype)

    real_roots_mask = roots_s_cplx.imag.abs() < EPSILON_IMAG
    t_values = roots_s_cplx.real
    BOUNDS_EPS = COEFF_EPS_MAIN
    #valid_t_range_mask = (t_values >= -BOUNDS_EPS) & (t_values <= 1.0 + BOUNDS_EPS)

    t_values[~(real_roots_mask)] = 1e12
    return t_values

    actual_intersections_mask_s = real_roots_mask & valid_t_range_mask
    return actual_intersections_mask_s.sum(-1, keepdim=True)

    has_valid_root_s = torch.any(actual_intersections_mask_s, dim=-1)

    has_intersection_flat[solvable_mask_flat] = has_valid_root_s

    #has_intersection = has_intersection_flat.reshape(N_lines, N_beziers)
    #intersections_per_line = torch.sum(has_intersection.int(), dim=1)

    return has_intersection_flat#intersections_per_line


tc2_lines_b = None  # Define globally for debug prints in main
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    if dtype == torch.float64:
        current_eps_imag = F64_EPSILON_IMAG_ZERO
        current_coeff_eps = F64_COEFF_EPSILON
        current_clamp_eps = F64_CLAMP_EPSILON
    else:
        current_eps_imag = F32_EPSILON_IMAG_ZERO
        current_coeff_eps = F32_COEFF_EPSILON  # This is the critical one we just changed
        current_clamp_eps = F32_CLAMP_EPSILON

    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Epsilon for imag==0 check: {current_eps_imag}")
    print(f"Epsilon for coeff==0 check (MAIN): {current_coeff_eps}")  # This is F32_COEFF_EPSILON
    print(f"Epsilon for clamp: {current_clamp_eps}")

    print("\n--- Test Case 1 ---")
    lines_b1 = torch.tensor([[0.5, 0.5]], device=device, dtype=dtype)
    lines_w1 = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
    bezier_cps1 = torch.tensor([[[0.0, 0.0], [0.25, 1.0], [0.75, 1.0], [1.0, 0.0]]], device=device, dtype=dtype)
    counts1 = count_line_bezier_intersections(lines_b1, lines_w1, bezier_cps1)
    print("Expected: [1], Got:", counts1)

    print("\n--- Test Case 2 ---")
    tc2_lines_b = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)  # Assign to global for debug
    tc2_lines_w = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
    tc2_bezier_cps = torch.tensor([[[0.0, 0.0], [0.5, 4. / 3.], [0.5, 4. / 3.], [1.0, 0.0]]], device=device,
                                  dtype=dtype)
    counts2 = count_line_bezier_intersections(tc2_lines_b, tc2_lines_w, tc2_bezier_cps)
    print("Expected: [1] (tangent), Got:", counts2)
    tc2_lines_b = None  # Reset

    # ... rest of tests
    print("\n--- Test Case 3 ---")
    lines_b3 = torch.tensor([[0.0, 2.0]], device=device, dtype=dtype)
    lines_w3 = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
    original_bezier_cps2 = torch.tensor([[[0.0, 0.0], [0.5, 1.0], [0.5, 1.0], [1.0, 0.0]]], device=device, dtype=dtype)
    counts3 = count_line_bezier_intersections(lines_b3, lines_w3, original_bezier_cps2)
    print("Expected: [0], Got:", counts3)

    print("\n--- Test Case 4 ---")
    lines_b4 = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
    lines_w4 = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
    bezier_cps4 = torch.tensor([[[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [1.0, 0.0]]], device=device, dtype=dtype)
    counts4 = count_line_bezier_intersections(lines_b4, lines_w4, bezier_cps4)
    print("Expected: [1], Got:", counts4)

    print("\n--- Test Case 5 ---")
    lines_b5 = torch.tensor([[0.5, 0.5], [0.0, 2.0]], device=device, dtype=dtype)
    lines_w5 = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device=device, dtype=dtype)
    bezier_cps5 = torch.stack([bezier_cps1.squeeze(0), bezier_cps4.squeeze(0)], dim=0)
    counts5 = count_line_bezier_intersections(lines_b5, lines_w5, bezier_cps5)
    print("Expected: [1, 0], Got:", counts5)

    print("\n--- Test Case 6 (Quadratic A=0) ---")
    P0_q, P1_q, P2_q = torch.tensor([0.0, 0.0], dtype=dtype), torch.tensor([0.5, 1.0], dtype=dtype), torch.tensor(
        [1.0, 0.0], dtype=dtype)
    CP0_qc, CP1_qc = P0_q, (P0_q + 2 * P1_q) / 3
    CP2_qc, CP3_qc = (2 * P1_q + P2_q) / 3, P2_q
    bezier_cps_quad = torch.stack([CP0_qc, CP1_qc, CP2_qc, CP3_qc], dim=0).unsqueeze(0).to(device=device)

    lines_b6 = torch.tensor([[0.0, 0.5]], device=device, dtype=dtype)
    lines_w6 = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
    counts6 = count_line_bezier_intersections(lines_b6, lines_w6, bezier_cps_quad)
    print("Expected: [1] (quadratic intersects y=0.5), Got:", counts6)

    print("\n--- Test Case 7 (Linear A=B=0) ---")
    P0_L, P1_L = torch.tensor([0.0, 0.0], dtype=dtype), torch.tensor([1.0, 1.0], dtype=dtype)
    CP0_lc, CP1_lc = P0_L, (2 * P0_L + P1_L) / 3
    CP2_lc, CP3_lc = (P0_L + 2 * P1_L) / 3, P1_L
    bezier_cps_linear = torch.stack([CP0_lc, CP1_lc, CP2_lc, CP3_lc], dim=0).unsqueeze(0).to(device=device)

    lines_b7 = torch.tensor([[0.0, 0.5]], device=device, dtype=dtype)
    lines_w7 = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
    counts7 = count_line_bezier_intersections(lines_b7, lines_w7, bezier_cps_linear)
    print("Expected: [1] (linear y=x intersects y=0.5), Got:", counts7)

    lines_b8 = torch.tensor([[0.0, 2.0]], device=device, dtype=dtype)
    lines_w8 = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype)
    counts8 = count_line_bezier_intersections(lines_b8, lines_w8, bezier_cps_linear)
    print("Expected: [0] (linear y=x does not intersect y=2.0 in range), Got:", counts8)

    print("\n--- Test Case 8 (Empty inputs) ---")
    empty_lines_b = torch.empty((0, 2), device=device, dtype=dtype)
    empty_lines_w = torch.empty((0, 2), device=device, dtype=dtype)
    empty_beziers = torch.empty((0, 4, 2), device=device, dtype=dtype)

    counts_empty_lines = count_line_bezier_intersections(empty_lines_b, empty_lines_w, bezier_cps1)
    print("Expected empty lines: [], Got:", counts_empty_lines)
    counts_empty_beziers = count_line_bezier_intersections(lines_b1, lines_w1, empty_beziers)
    print("Expected empty beziers: [0], Got:", counts_empty_beziers)