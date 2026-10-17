"""An integer from NumPy or torch is an integer.

``np.random.permutation`` and ``torch.randperm`` hand out NumPy integers and
integer tensors, and so does arithmetic on any array. Indexing a text with one
-- ``for i in np.random.permutation(len(text)): text[i]`` -- raised ``batch
indices must be integers or slices, not int64``, and a pack indexed with one
raised ``'numpy.int64' object is not iterable``. A Sphere given one as its
resolution failed to unpack it as a pair, and a video resolution or a grid's
row count was refused as not an integer.
"""

import numpy as np
import pytest
import torch

from algan import HD, RIGHT, Group, Scene, Sphere, Square, Text, batch_mobs

INTEGERS = [np.int64(1), np.int32(1), torch.tensor(1), torch.tensor([1])]
KINDS = ["numpy int64", "numpy int32", "0-d tensor", "1-element tensor"]


@pytest.mark.parametrize("index", INTEGERS, ids=KINDS)
def test_a_text_is_indexed_by_any_integer(index):
    with Scene():
        text = Text("hello")
        torch.testing.assert_close(text[index].get_center(), text[1].get_center())


@pytest.mark.parametrize("index", INTEGERS, ids=KINDS)
def test_a_pack_is_indexed_by_any_integer(index):
    with Scene():
        pack = batch_mobs([Square(), Square().move(RIGHT * 3)])
        torch.testing.assert_close(pack[index].get_center(), pack[1].get_center())


def test_glyphs_are_picked_in_a_random_order():
    with Scene():
        text = Text("hello")
        order = np.random.default_rng(0).permutation(len(text))
        centers = [text[i].get_center() for i in order]
        for i, center in zip(order, centers):
            torch.testing.assert_close(center, text[int(i)].get_center())


def test_a_sphere_takes_any_integer_resolution():
    with Scene():
        assert Sphere(resolution=np.int64(12)).grid_width == (
            Sphere(resolution=12).grid_width
        )


def test_video_settings_take_numpy_integers():
    settings = HD.set(
        resolution=(np.int64(320), np.int64(180)), frames_per_second=np.int32(24)
    )
    assert settings.resolution == (320, 180)
    assert settings.frames_per_second == 24
    assert all(type(value) is int for value in (*settings.resolution, settings.fps))


def test_a_grid_takes_any_integer_row_count():
    with Scene():
        expected = Group(*[Square() for _ in range(4)]).arrange_in_grid(2)
        grid = Group(*[Square() for _ in range(4)]).arrange_in_grid(np.int64(2))
        for member, reference in zip(grid, expected):
            torch.testing.assert_close(member.get_center(), reference.get_center())
