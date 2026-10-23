"""Edge cases of animation-context timing.

A feature test, outside the fast suite.
"""

from __future__ import annotations

import pytest
import torch

from algan import RIGHT, Lag, Square
from algan.errors import AlganConfigurationError
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _square():
    return Square(add_to_scene=False).spawn(False)


@pytest.mark.parametrize("ratio", [-0.5, -1, -1e-9])
def test_lag_rejects_negative_ratio(ratio):
    """REGRESSION. ``Lag(-0.5)`` moved the cursor backwards, so the second of
    two moves started at scene time -0.5 and its first half was never shown;
    mid-scene it started before the block, on top of earlier animations.
    """
    with pytest.raises(AlganConfigurationError, match="negative"):
        Lag(ratio)


def test_lag_zero_ratio_still_allowed():
    a, b = _square(), _square()
    with Lag(0):
        a.move(RIGHT)
        b.move(RIGHT)
    SceneManager.instance().current_scene.timeline_manager.set_state_to_times(
        torch.tensor([0.0], dtype=torch.get_default_dtype())
    )
    torch.testing.assert_close(a.location, b.location)
