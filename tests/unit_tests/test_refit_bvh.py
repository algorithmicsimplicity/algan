"""Focused tests for the shared-topology refit BVH builder."""

import warnings

import pytest
import torch

from algan.rendering.raytracing.refit_bvh import build_refit_bvh


def test_refit_bvh_does_not_expose_index_reduce_beta_warning():
    """The internal PyTorch beta warning must not reach Algan users."""
    lo = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]]])
    hi = lo + 0.5

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"index_reduce\(\) is in beta and the API may change at any time\.",
            category=UserWarning,
        )
        build_refit_bvh(lo, hi, num_frames=1)


def _block_bits(tree):
    """Bit pattern of a tree's blocks (f16 blocks carry link words as NaN
    payloads, which ``torch.equal`` would never call equal).
    """
    blocks = tree.blocks
    return blocks.view(torch.int16 if blocks.dtype == torch.float16 else torch.int32)


def _static_bounds(n=6):
    torch.manual_seed(0)
    lo = torch.rand(1, n, 3)
    return lo, lo + 0.5


def test_refit_bvh_reduces_per_frame_opacity_over_a_static_tree():
    """Static bounds with a still-per-frame opacity mask must build.

    The merge collapses each temporally-constant table on its own, so a batch
    whose geometry holds still while a mob fades reaches the builder as
    ``Tc == 1, To == T``. One tree covers every frame, so the flag holds only
    where it holds on all of them -- the reduction build_stbvh applies to its
    static instances.
    """
    lo, hi = _static_bounds()
    n = lo.shape[1]
    opaque = torch.zeros((4, n), dtype=torch.bool)
    opaque[:2] = True  # opaque early, translucent later

    tree = build_refit_bvh(lo, hi, num_frames=4, opaque=opaque)
    conservative = build_refit_bvh(
        lo, hi, num_frames=4, opaque=opaque.all(0, keepdim=True)
    )
    assert torch.equal(_block_bits(tree), _block_bits(conservative))

    # A primitive opaque on every frame keeps its flag.
    always = build_refit_bvh(
        lo, hi, num_frames=4, opaque=torch.ones((4, n), dtype=torch.bool)
    )
    flagged = build_refit_bvh(
        lo, hi, num_frames=4, opaque=torch.ones((1, n), dtype=torch.bool)
    )
    assert torch.equal(_block_bits(always), _block_bits(flagged))
    assert not torch.equal(_block_bits(always), _block_bits(tree))


def test_refit_bvh_rejects_an_opacity_mask_of_a_foreign_frame_count():
    lo, hi = _static_bounds()
    lo = lo.expand(4, -1, -1).contiguous()
    hi = hi.expand(4, -1, -1).contiguous()
    with pytest.raises(ValueError, match="opacity mask has 3 frames"):
        build_refit_bvh(
            lo, hi, num_frames=4, opaque=torch.zeros((3, lo.shape[1]), dtype=torch.bool)
        )


def _local_refit_device():
    from algan.rendering.taichi_runtime import (
        init_taichi,
        taichi_arch_is_cuda,
        taichi_launch_is_local,
    )

    init_taichi()
    device = torch.device("cuda" if taichi_arch_is_cuda() else "cpu")
    if not taichi_launch_is_local(device):
        pytest.skip("packed refit requires a local CPU or CUDA tensor/backend pair")
    return device


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize(("frames", "primitives"), [(1, 1), (1, 41), (4, 257)])
def test_packed_refit_matches_torch_bits(half, frames, primitives, monkeypatch):
    """Pack the same topology with moving visibility, flags and duplicate leaves."""
    from algan import SETTINGS
    from algan.rendering.raytracing import refit_bvh

    device = _local_refit_device()
    generator = torch.Generator().manual_seed(832)
    lo = torch.randn(frames, primitives, 3, generator=generator).to(device)
    hi = lo + torch.rand(frames, primitives, 3, generator=generator).to(device)
    if primitives > 1:
        lo[0, ::3], hi[0, ::3] = 1e17, -1e17
    opaque = (torch.rand(frames, primitives, generator=generator) > 0.5).to(device)
    casts = (torch.rand(primitives, generator=generator) > 0.5).to(device)
    leaf_prim = torch.arange(primitives, device=device) // 2
    # Exercise only the builder: no traversal compiled with the startup-fixed
    # layout reads these trees. The new pack kernel takes layout as a template.
    monkeypatch.setattr(refit_bvh, "bvh_block_f16", half)
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=False):
        reference = build_refit_bvh(
            lo, hi, opaque=opaque, casts=casts, leaf_prim=leaf_prim
        )
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=True):
        candidate = build_refit_bvh(
            lo, hi, opaque=opaque, casts=casts, leaf_prim=leaf_prim
        )
    assert candidate.first_leaf == reference.first_leaf
    assert torch.equal(_block_bits(candidate), _block_bits(reference))


def test_packed_refit_half_rounding_boundaries(monkeypatch):
    """Outward rounding must preserve tiny and clamped boxes on either side of zero."""
    from algan import SETTINGS
    from algan.rendering.raytracing import refit_bvh

    device = _local_refit_device()
    values = torch.tensor(
        [
            -1e17,
            -65504.0,
            -1.001,
            -(2**-14),
            -(2**-20),
            -(2**-25),
            -(2**-30),
            -0.0,
            0.0,
            2**-30,
            2**-25,
            2**-20,
            2**-14,
            1.001,
            65504.0,
            1e17,
        ],
        device=device,
    )
    lo = values.reshape(1, -1, 1).expand(-1, -1, 3).contiguous()
    hi = lo.clone()
    monkeypatch.setattr(refit_bvh, "bvh_block_f16", True)
    original = refit_bvh.build_refit_bvh
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=False):
        reference = original(lo, hi)
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=True):
        candidate = original(lo, hi)
    actual, expected = _block_bits(candidate), _block_bits(reference)
    different = actual != expected
    assert not bool(different.any()), (
        different.nonzero().tolist(),
        actual[different].tolist(),
        expected[different].tolist(),
    )


@pytest.mark.parametrize("hidden", [False, True])
def test_packed_refit_static_opacity_and_empty_tree(hidden, monkeypatch):
    from algan import SETTINGS
    from algan.rendering.raytracing import refit_bvh_taichi

    device = _local_refit_device()
    lo, hi = (bound.to(device) for bound in _static_bounds())
    if hidden:
        lo.fill_(1e17)
        hi.fill_(-1e17)
    opacity = torch.ones((4, lo.shape[1]), dtype=torch.bool, device=device)
    opacity[1, ::2] = False
    original = refit_bvh_taichi.refit_pack_level
    launches = []

    def tracked(*args):
        launches.append(1)
        return original(*args)

    monkeypatch.setattr(refit_bvh_taichi, "refit_pack_level", tracked)
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=False):
        reference = build_refit_bvh(lo, hi, num_frames=4, opaque=opacity)
    assert not launches
    with SETTINGS.raytracing.experimental.override(refit_pack_kernel=True):
        candidate = build_refit_bvh(lo, hi, num_frames=4, opaque=opacity)
    assert launches, "the packed kernel must actually run in the parity test"
    assert torch.equal(_block_bits(candidate), _block_bits(reference))
