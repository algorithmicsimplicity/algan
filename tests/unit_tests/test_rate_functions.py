"""Contracts of the public rate functions.

``rate_funcs`` is a leaf module every animation goes through, and a rate
function that stops hitting its endpoints silently changes the final state of
every animation authored with it rather than raising.  These tests pin the
endpoints, the range and the monotonicity so a rewrite of one curve cannot
quietly move where animations land.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from algan import DEFAULT_RATE_FUNC, Off, Scene, Square, Sync, rate_funcs
from algan.constants.spatial import RIGHT

SAMPLES = torch.linspace(0.0, 1.0, 65)

# Every callable the module exports that maps progress to progress. ``inversed``
# is a combinator over one of these, not one itself.
CURVES = sorted(
    name
    for name, value in vars(rate_funcs).items()
    if not name.startswith("_")
    and callable(value)
    and name not in {"inversed"}
    and getattr(value, "__module__", None) == rate_funcs.__name__
)


def _evaluate(func, t):
    return torch.as_tensor(func(t)).reshape(-1).double()


def test_the_public_curve_set_is_not_silently_empty():
    # A refactor that moved these to another module would otherwise turn every
    # test below into a vacuous pass.
    assert len(CURVES) >= 8
    assert {"identity", "linear", "smooth", "ease_out_expo"} <= set(CURVES)


@pytest.mark.parametrize("name", CURVES)
def test_rate_function_starts_at_zero_and_finishes_at_one(name):
    func = getattr(rate_funcs, name)
    values = _evaluate(func, SAMPLES)
    assert values[0] == pytest.approx(0.0, abs=1e-5), f"{name} does not start at 0"
    assert values[-1] == pytest.approx(1.0, abs=1e-5), f"{name} does not end at 1"


@pytest.mark.parametrize("name", CURVES)
def test_rate_function_is_finite_and_stays_inside_the_unit_interval(name):
    values = _evaluate(getattr(rate_funcs, name), SAMPLES)
    assert torch.isfinite(values).all(), f"{name} is not finite on [0, 1]"
    assert values.min() >= -1e-5, f"{name} undershoots 0"
    assert values.max() <= 1 + 1e-5, f"{name} overshoots 1"


@pytest.mark.parametrize("name", CURVES)
def test_rate_function_never_runs_an_animation_backwards(name):
    values = _evaluate(getattr(rate_funcs, name), SAMPLES)
    assert (values.diff() >= -1e-5).all(), f"{name} is not monotone increasing"


@pytest.mark.parametrize("name", CURVES)
def test_rate_function_accepts_a_scalar_progress_tensor(name):
    # Contexts evaluate the curve on whatever shape the timeline produced,
    # including a single frame time.
    value = float(
        torch.as_tensor(getattr(rate_funcs, name)(torch.tensor(0.5))).reshape(-1)[0]
    )
    assert 0.0 - 1e-5 <= value <= 1.0 + 1e-5


def test_linear_is_the_identity_curve_not_a_copy_of_it():
    assert rate_funcs.linear is rate_funcs.identity
    assert _evaluate(rate_funcs.identity, SAMPLES) == pytest.approx(
        SAMPLES.double(), abs=1e-6
    )


def test_inversed_mirrors_a_curve_through_the_diagonal():
    mirrored = rate_funcs.inversed(rate_funcs.ease_out_expo)
    forward = _evaluate(rate_funcs.ease_out_expo, SAMPLES)
    backward = _evaluate(mirrored, SAMPLES)
    assert backward[0] == pytest.approx(0.0, abs=1e-5)
    assert backward[-1] == pytest.approx(1.0, abs=1e-5)
    assert backward == pytest.approx(1 - forward.flip(0), abs=1e-5)


def test_default_rate_func_is_one_of_the_published_curves():
    assert callable(DEFAULT_RATE_FUNC)
    assert DEFAULT_RATE_FUNC in {getattr(rate_funcs, name) for name in CURVES}


def test_smooth_takes_its_inflection_as_a_keyword_default():
    # ``smooth`` is the default curve and is also called with an explicit
    # inflection by the indication animations.
    signature = inspect.signature(rate_funcs.smooth)
    assert signature.parameters["inflection"].default == 10.0
    sharp = _evaluate(rate_funcs.smooth, SAMPLES)
    gentle = torch.as_tensor(rate_funcs.smooth(SAMPLES, 2.0)).reshape(-1).double()
    # A steeper inflection spends longer near the ends and crosses faster.
    assert sharp[len(sharp) // 4] < gentle[len(gentle) // 4]


@pytest.mark.parametrize("name", ["linear", "smooth", "ease_out_expo"])
def test_a_context_rate_func_reshapes_the_path_but_not_its_endpoints(name):
    """The whole point of a rate function: same start and end, different middle."""
    func = getattr(rate_funcs, name)
    with Scene() as scene:
        with Off():
            square = Square().spawn()
        with Sync(run_time=1.0, rate_func=func):
            square.move(RIGHT * 2)

        times = torch.tensor([0.0, 0.5, 1.0])
        scene.timeline_manager.set_state_to_times(times)
        travelled = [
            float(square.location[index].reshape(-1, 3)[:, 0].mean()) for index in range(3)
        ]

    assert travelled[0] == pytest.approx(0.0, abs=1e-3)
    assert travelled[2] == pytest.approx(2.0, abs=1e-3)
    assert 0.0 <= travelled[1] <= 2.0
    expected_midpoint = 2.0 * float(torch.as_tensor(func(0.5)).reshape(-1)[0])
    assert travelled[1] == pytest.approx(expected_midpoint, abs=0.05)
