"""A distant camera must not turn projection roundoff into PN curvature."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from algan.rendering.logical_pn import dice_pattern, evaluate_logical_pn
from algan.rendering.raytracing.primitives import (
    LogicalPNTrianglePrimitive,
    _pn_criterion_inputs,
)
from algan.rendering.taichi_runtime import (
    ensure_taichi_for_render,
    taichi_arch_is_cpu,
)
from algan.settings import SETTINGS
from algan.settings._startup import render_device


def _probe():
    # These methods only need the tolerance, not the material/arena machinery.
    primitive = object.__new__(LogicalPNTrianglePrimitive)
    primitive.render_tolerance_pixels = 0.5
    return primitive


def _capture(device):
    path = Path(__file__).parents[1] / "assets" / "pn_distant_camera.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    tensors = {
        name: torch.tensor(values[name], dtype=torch.float32, device=device)
        for name in ("control_points", "edge_controls", "cam_o", "sp", "sb", "slack")
    }
    cam = tuple(tensors[name] for name in ("cam_o", "sp", "sb"))
    sign = torch.sign(((cam[1] - cam[0]) * cam[2][:, 2]).sum(-1))
    return tensors, cam, sign, values["screen_height"]


def _reference_error(exact, approximated, cam, sign, height, slack=None):
    """Independent float64 ray/plane intersections, including the screen guard."""
    exact, approximated = exact.cpu().double(), approximated.cpu().double()
    origin, screen, basis = (value.cpu().double() for value in cam)
    shape = (-1,) + (1,) * (exact.ndim - 2) + (3,)
    origin, screen = origin.reshape(shape), screen.reshape(shape)
    sx, sy, normal = (basis[:, axis].reshape(shape) for axis in range(3))
    distance = ((screen - origin) * normal).sum(-1)

    def project(points):
        ray = points - origin
        depth = (ray * normal).sum(-1)
        intersection = origin + (distance / depth).unsqueeze(-1) * ray
        relative = intersection - screen
        pixels = torch.stack(((relative * sx).sum(-1), (relative * sy).sum(-1)), -1)
        return pixels * (height / 2), depth

    a, da = project(approximated)
    e, de = project(exact)
    guard = height * 1.5
    error = (e.clamp(-guard, guard) - a.clamp(-guard, guard)).norm(dim=-1)
    if slack is not None:
        allowance = slack.cpu().double().reshape(-1, *((1,) * (error.ndim - 1)))
        error = (error - allowance * (distance / de).abs() * height / 2).clamp_min(0)
    sign = sign.cpu().double().reshape(-1, *((1,) * (error.ndim - 1)))
    usable = (
        torch.isfinite(error)
        & (da * sign > 1e-7)
        & (de * sign > 1e-7)
        & ((a.abs() <= guard).all(-1) | (e.abs() <= guard).all(-1))
    )
    return torch.where(usable, error, 0)


@pytest.mark.parametrize("kernel", [False, True], ids=["torch", "kernel"])
def test_distant_camera_patches_converge_without_hitting_the_safety_cap(kernel):
    SETTINGS.raytracing.experimental.pn_criterion_kernel = kernel
    device = render_device() if kernel else torch.device("cpu")
    if kernel:
        ensure_taichi_for_render()
        # The same arrangements ``pn_criterion_kernel_active`` accepts: the
        # arch is the CPU, or projection runs on a CUDA render device. An MPS
        # render device is neither, so the renderer never runs the kernels.
        if not (taichi_arch_is_cpu() or device.type == "cuda"):
            pytest.skip(f"the PN criterion kernels do not run on {device}")
    data, cam, sign, height = _capture(device)
    inputs = _pn_criterion_inputs(
        data["control_points"], data["edge_controls"], *cam, sign, data["slack"]
    )
    assert (inputs is not None) == kernel, "must exercise the requested criterion"
    primitive = _probe()
    # A broken search need only reach four, not allocate all 4**8 trial cells.
    primitive.max_subdivision_level = 4
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        levels, _, _, _ = primitive._required_subdivision_levels(
            data["control_points"],
            data["edge_controls"],
            *cam,
            height,
            False,
            data["slack"],
        )
    assert not [w for w in caught if "safety cap" in str(w.message)]
    assert levels.tolist() == [[2, 2, 0]]


def test_small_projected_errors_match_double_precision_for_a_distant_camera():
    data, cam, sign, height = _capture(torch.device("cpu"))
    primitive = _probe()
    pattern = dice_pattern(3, 3, 0, device=torch.device("cpu"), dtype=torch.float32)
    # Test actual curved-patch samples at a refinement which must pass. Keep
    # the same float32 world points for both projection calculations.
    weights = torch.tensor(primitive._flatness_sample_weights)
    vertices = evaluate_logical_pn(data["control_points"], pattern.vertex_uv)
    corner_uv = pattern.vertex_uv[pattern.triangle_indices]
    exact = evaluate_logical_pn(
        data["control_points"], torch.einsum("sk,mka->msa", weights, corner_uv)
    )
    approximated = torch.einsum(
        "sk,tpmkc->tpmsc", weights, vertices[:, :, pattern.triangle_indices]
    )
    expected = _reference_error(exact, approximated, cam, sign, height, data["slack"])
    actual = primitive._guarded_pixel_error(
        exact, approximated, cam, sign, height, data["slack"]
    )
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)
    assert (
        float(actual.max())
        < primitive._pixel_threshold(height) / primitive._flatness_safety_factor
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_projected_error_preserves_guard_clipping_and_front_depth(sign):
    # Screen coordinates need not use an orthogonal or unit-length x/y basis.
    basis = torch.tensor([[[1.2, 0.1, 0], [0.2, 0.8, 0], [0, 0, 1.0]]])
    origin = torch.zeros(1, 3)
    screen = torch.tensor([[0, 0, sign]])
    cam = (origin, screen, basis)
    pairs = torch.tensor(
        [
            [[0, 0, 1], [0.1, 0.2, 1.2]],  # both in frame, changing depth
            [[0.2, 0.3, 1], [0.2, 0.3, 1]],  # no displacement
            [[0, 0, 1], [8, 8, 1]],  # exit the guard on both axes
            [[-8, 0, 1], [0, 0, 1]],  # enter from beyond the other edge
            [[8, 0, 1], [9, 0, 1]],  # both outside: ignore
            [[-8, 0, 1], [8, 0, 1]],  # neither endpoint inside: ignore
            [[0, 0, -1], [0.1, 0.1, 1]],  # one behind the eye
            [[0, 0, 0], [0.1, 0.1, 1]],  # one on the eye plane
            [[0, 0, 1], [0.1, 0.1, 0]],  # the other on the eye plane
            [[0, 0, 1e-8], [0.1, 0.1, 1]],  # below the minimum depth
            [[0, 0, 10000], [0.1, 0.1, 1e-4]],  # exact depth must not cancel to zero
        ],
        dtype=torch.float32,
    )
    pairs[..., 2] *= sign
    approximated, exact = (
        pairs[:, 0].reshape(1, 1, -1, 3),
        pairs[:, 1].reshape(1, 1, -1, 3),
    )
    front_sign = torch.tensor([sign])
    expected = _reference_error(exact, approximated, cam, front_sign, 396)
    actual = _probe()._guarded_pixel_error(exact, approximated, cam, front_sign, 396)
    torch.testing.assert_close(actual.double(), expected, atol=2e-4, rtol=2e-6)
