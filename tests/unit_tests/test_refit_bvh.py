"""Focused tests for the shared-topology refit BVH builder."""

import warnings

import torch

from algan.rendering.raytracing.refit_bvh import build_refit_bvh


def test_refit_bvh_does_not_expose_index_reduce_beta_warning():
    """The internal PyTorch beta warning must not reach Algan users."""
    lo = torch.tensor(
        [[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]]]
    )
    hi = lo + 0.5

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"index_reduce\(\) is in beta and the API may change at any time\.",
            category=UserWarning,
        )
        build_refit_bvh(lo, hi, num_frames=1)
