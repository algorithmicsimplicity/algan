"""f16 sibling bounds must contain their children at every coordinate scale.

``stbvh._half_bits_directed`` promises the decoded f16 is ``>= x`` for an upper
bound and ``<= x`` for a lower one -- that guarantee is the whole reason the
compressed blocks can never falsely cull a hit. It used to hold only inside the
finite f16 range: bounds were clamped to +-65504 *before* the directed
rounding, so a scene whose geometry reached past that got sibling boxes that
decoded strictly inside their children and dropped real intersections with no
error anywhere. These are the out-of-range cases.
"""

import torch

from algan.rendering.raytracing.stbvh import _F16_MAX, _half_bits_directed


def _decode(x, up):
    return _half_bits_directed(x, up=up).view(torch.float16).float()


#: Beyond the f16 range, at it, and safely inside it.
_MAGNITUDES = torch.tensor(
    [1e9, 1e6, 1e5, 70000.0, 65505.0, _F16_MAX, 1000.0, 0.5, 0.0]
)


def test_upper_bounds_never_decode_below_the_value_they_bound():
    x = torch.cat((_MAGNITUDES, -_MAGNITUDES))
    assert torch.all(_decode(x, up=True) >= x)


def test_lower_bounds_never_decode_above_the_value_they_bound():
    x = torch.cat((_MAGNITUDES, -_MAGNITUDES))
    assert torch.all(_decode(x, up=False) <= x)


def test_out_of_range_saturates_to_infinity_not_to_the_finite_maximum():
    # The distinguishing case: clamping produced 65504 here, which is the bug.
    big = torch.tensor([1e6])
    assert torch.isinf(_decode(big, up=True)).all()
    assert torch.isinf(_decode(-big, up=False)).all()


def test_the_representable_side_still_clamps_rather_than_saturating():
    # An upper bound below -65504 is safely represented by -65504 (which is
    # already >= it), and must NOT become -inf. This is what keeps the packers'
    # empty-slot sentinels -- lo=+1e17, hi=-1e17, encoded as a lo > hi box that
    # no ray selects -- from turning into infinite boxes every ray enters.
    lo = _decode(torch.tensor([1e17]), up=False)
    hi = _decode(torch.tensor([-1e17]), up=True)
    assert lo.item() == _F16_MAX
    assert hi.item() == -_F16_MAX
    assert lo.item() > hi.item()


def test_in_range_values_are_unchanged_by_the_saturation_path():
    x = torch.tensor([0.0, 1e-7, -1e-7, 0.5, -0.5, 1000.0, -1000.0, _F16_MAX])
    up = _decode(x, up=True)
    down = _decode(x, up=False)
    assert torch.all(torch.isfinite(up))
    assert torch.all(torch.isfinite(down))
    assert torch.all(up >= x)
    assert torch.all(down <= x)
