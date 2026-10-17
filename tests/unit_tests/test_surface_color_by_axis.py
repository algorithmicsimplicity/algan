"""``Surface.set_color_by_axis`` maps one coordinate through a color scale.

It read the coordinate as a strided view of the vertex positions and handed
that to ``torch.searchsorted``, which warns about a non-contiguous query: every
script that colored a surface by height printed a PyTorch performance warning
about Algan's internals.
"""

import torch

from algan import BLUE, YELLOW, Scene, Surface


def _ramp(uv):
    return torch.cat((uv[..., :1], uv[..., 1:], uv[..., :1]), -1)


def test_coloring_by_height_hands_searchsorted_a_packed_query(monkeypatch):
    original = torch.searchsorted
    queries = []

    def searchsorted(sorted_sequence, values, *args, **kwargs):
        queries.append(values)
        return original(sorted_sequence, values, *args, **kwargs)

    monkeypatch.setattr(torch, "searchsorted", searchsorted)
    with Scene():
        Surface(_ramp, add_to_scene=False).set_color_by_axis(colorscale=[BLUE, YELLOW])

    assert queries
    assert all(query.is_contiguous() for query in queries)


def test_the_low_end_takes_the_first_color_and_the_high_end_the_last():
    with Scene():
        surface = Surface(_ramp, add_to_scene=False)
        surface.set_color_by_axis(colorscale=[BLUE, YELLOW], axis=0)
        heights = surface.grid.location.reshape(-1, 3)[:, 0]
        colors = surface.grid.color.reshape(-1, surface.grid.color.shape[-1])

    blue = torch.as_tensor(BLUE, dtype=colors.dtype).reshape(-1)[:3]
    yellow = torch.as_tensor(YELLOW, dtype=colors.dtype).reshape(-1)[:3]
    torch.testing.assert_close(colors[heights.argmin(), :3], blue)
    torch.testing.assert_close(colors[heights.argmax(), :3], yellow)
