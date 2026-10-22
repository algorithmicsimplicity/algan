"""The authoring validators' fast paths answer exactly as the full checks do.

``reject_non_finite`` and ``reject_outside_unit_interval`` run on every
animatable write, so they take shortcuts for the values that are almost always
valid: a Python number is judged without building a tensor, a tensor's
finiteness is read from one reduction, and the unit interval from one
``aminmax``. Each shortcut may only ever say "valid" when the elementwise test
would; everything else falls through to that test. These tests hold the
shortcuts to the reference behaviour -- same verdict, same message -- on the
inputs where a shortcut could plausibly get it wrong.
"""

from __future__ import annotations

import pytest
import torch

from algan import BLUE, Square
from algan.errors import AlganConfigurationError
from algan.utils.tensor_utils import (
    cast_to_tensor,
    reject_non_finite,
    reject_outside_unit_interval,
)


def _reference_non_finite(name, value, given=None):
    """The validator as it was before the fast paths, verbatim."""
    if torch.is_tensor(value) and value.is_floating_point():
        finite = torch.isfinite(value)
        if not bool(finite.all()):
            what = "NaN" if bool(torch.isnan(value).any()) else "infinite"
            shown = value if given is None else given
            raise AlganConfigurationError(
                f"{name} must be finite; got a {what} value: {shown!r}. A "
                f"non-finite value propagates through every later frame and "
                f"renders as a blank or missing shape rather than raising."
            )
    return value


def _reference_unit_interval(name, value, given=None):
    _reference_non_finite(name, value, given=given)
    if torch.is_tensor(value):
        outside = bool(((value < 0) | (value > 1)).any())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        outside = not 0.0 <= float(value) <= 1.0
    else:
        outside = False
    if outside:
        shown = value if given is None else given
        raise AlganConfigurationError(
            f"{name} must be between 0 and 1 inclusive; got {shown!r}. It is a "
            f"fraction, so a value outside that range has no rendered meaning; "
            f"it is rejected at the line that wrote it rather than clamped "
            f"silently at materialization."
        )
    return value


def _outcome(check, *args, **kwargs):
    try:
        check(*args, **kwargs)
    except AlganConfigurationError as error:
        return str(error)
    return None


NAN, INF = float("nan"), float("inf")

# Python values a user writes, each cast the way the authoring funnel casts it.
SOURCES = [
    0,
    1,
    0.5,
    -2.5,
    3.0e38,  # finite in float32
    1.0e39,  # finite in Python, infinite once cast to float32
    -1.0e39,
    10**39,  # an int beyond float32's range
    NAN,
    INF,
    -INF,
    True,
    (0.0, 1.0, 2.0),
    (0.0, NAN, 2.0),
]

TENSORS = [
    torch.tensor([[[0.5]]]),
    torch.tensor([[[3.0e38, 3.0e38]]]),  # finite elements whose sum overflows
    torch.tensor([[[-3.0e38, 3.0e38, 3.0e38]]]),
    torch.tensor([[[1.0, NAN]]]),
    torch.tensor([[[INF, -INF]]]),  # sums to NaN, reported as infinite
    torch.tensor([[[INF, 1.0]]]),
    torch.empty(1, 0, 3),
    torch.tensor([[[1.0e-45]]]),  # a denormal
    torch.tensor([[[-1.0e-45]]]),  # just below the interval
    torch.tensor([[[1.0000001]]]),  # just above it
    torch.tensor([[[0.0, 1.0]]]),
    torch.tensor([[[-0.0]]]),
    torch.tensor([[[2]]]),  # integer tensors are not checked for finiteness
    torch.tensor([[[0.5, NAN]]], dtype=torch.float64),
    torch.tensor([[[0.5]]], dtype=torch.float16),
    torch.tensor([[[60000.0, 60000.0]]], dtype=torch.float16),  # sum overflows
    torch.tensor([[[0.5, 2.0]]]).expand(1, 4, 2),  # zero-stride view
    BLUE,
    BLUE * NAN,
]


@pytest.mark.parametrize("source", SOURCES, ids=repr)
def test_a_cast_python_value_is_judged_as_its_tensor_would_be(source):
    value = cast_to_tensor(source)
    expected = _outcome(_reference_non_finite, "glow", value)
    assert _outcome(reject_non_finite, "glow", value, cast_from=source) == expected
    expected = _outcome(_reference_unit_interval, "opacity", value, given=source)
    assert (
        _outcome(
            reject_outside_unit_interval,
            "opacity",
            value,
            given=source,
            cast_from=source,
        )
        == expected
    )


@pytest.mark.parametrize("value", TENSORS, ids=lambda t: f"{t.dtype}{tuple(t.shape)}")
def test_a_tensor_is_judged_as_the_elementwise_check_judges_it(value):
    assert _outcome(reject_non_finite, "location", value) == _outcome(
        _reference_non_finite, "location", value
    )
    assert _outcome(reject_outside_unit_interval, "opacity", value) == _outcome(
        _reference_unit_interval, "opacity", value
    )


def test_a_plain_number_never_builds_a_tensor_to_be_validated(monkeypatch):
    """The point of the number path: no tensor op runs for a valid number."""
    import algan.utils.tensor_utils as tensor_utils

    def forbidden(*args, **kwargs):
        raise AssertionError("the tensor check ran for a plain number")

    monkeypatch.setattr(tensor_utils, "_all_finite", forbidden)
    monkeypatch.setattr(tensor_utils, "_within_unit_interval", forbidden)
    value = cast_to_tensor(0.25)
    assert reject_non_finite("glow", value, cast_from=0.25) is value
    assert reject_outside_unit_interval("opacity", value, cast_from=0.25) is value


def test_the_number_path_follows_the_default_dtype():
    """``1e39`` fits float64, so it is finite once the default dtype is float64."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        value = cast_to_tensor(1.0e39)
        assert reject_non_finite("glow", value, cast_from=1.0e39) is value
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    ("attr", "bad", "message"),
    [
        ("opacity", NAN, "opacity must be finite; got a NaN value: nan."),
        ("opacity", 1.5, "opacity must be between 0 and 1 inclusive; got 1.5."),
        ("glow", INF, "glow must be finite; got a infinite value: tensor"),
        ("glow", 1.0e39, "glow must be finite; got a infinite value: tensor"),
    ],
)
def test_the_assignment_funnel_still_names_the_bad_value(attr, bad, message):
    square = Square()
    with pytest.raises(AlganConfigurationError) as raised:
        setattr(square, attr, bad)
    assert str(raised.value).startswith(message)
