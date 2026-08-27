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
