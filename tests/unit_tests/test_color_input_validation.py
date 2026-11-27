"""Color inputs that were stored as given and drew the wrong thing.

- An integer tensor or NumPy array in 0-255 (``np.array([255, 128, 0],
  dtype=np.uint8)``, how image libraries hand a color over) passed straight
  through, as integer tuples did before they were refused.
- A negative glow was accepted by ``Color`` and by its ``glow`` setter. Glow
  is additive emission, so a negative value has no rendered meaning.

Record-only; no render.
"""

import numpy as np
import pytest
import torch

from algan import Square
from algan.constants.color import Color, to_color
from algan.errors import AlganConfigurationError, InvalidColorError


@pytest.mark.parametrize(
    "value",
    [
        (255, 0, 0),
        [255, 128, 0],
        (255, 128, 0, 255),
        np.array([255, 128, 0], dtype=np.uint8),
        torch.tensor([255, 128, 0]),
        torch.tensor([[255, 128, 0], [0, 0, 1]]),
    ],
)
def test_integer_channels_above_one_are_refused(value):
    with pytest.raises(InvalidColorError, match="255"):
        to_color(value)
    with pytest.raises(InvalidColorError, match="255"):
        Color(value)


def test_a_one_row_integer_array_gets_the_scaled_value_and_hex_code():
    with pytest.raises(InvalidColorError, match="'#FF8000'"):
        to_color(np.array([255, 128, 0], dtype=np.uint8))


def test_the_constructor_refuses_integer_channels():
    with pytest.raises(InvalidColorError, match="255"):
        Square(color=torch.tensor([255, 0, 0]))


def test_integer_zeros_and_ones_and_float_buffers_stay_legal():
    assert to_color(torch.tensor([1, 0, 1])).shape[-1] == 5
    assert to_color(np.array([1, 0, 1])).shape[-1] == 5
    assert to_color(torch.tensor([2.0, 0.5, 0.0])).tolist()[:3] == [2.0, 0.5, 0.0]
    # A 3-D integer tensor is an image, not a list of colors: not refused.
    image = torch.full((2, 2, 3), 200)
    assert to_color(image)[..., :3].eq(200).all()


def test_negative_glow_is_refused():
    with pytest.raises(AlganConfigurationError, match="glow"):
        Color((1, 0, 0), glow=-1)
    with pytest.raises(AlganConfigurationError, match="glow"):
        Color((1.0, 0.0, 0.0, -0.5, 1.0))
    color = Color((1.0, 0.0, 0.0))
    with pytest.raises(AlganConfigurationError, match="glow"):
        color.glow = -2
    assert color.glow.item() == 0
    assert Color((1, 0, 0), glow=3).glow.item() == 3
