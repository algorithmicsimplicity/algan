"""Group layout: ``arrange_in_grid``.

Render-free: layouts are recorded inside ``Off()`` and read back directly.
"""

import pytest

from algan import Group, Off, Scene, SceneManager, Square


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.mark.parametrize(
    ("members", "num_rows", "rows_used"),
    [(4, 3, 2), (6, 4, 3), (9, 3, 3), (5, 2, 2)],
)
def test_arrange_in_grid_centres_the_rows_it_actually_uses(
    members, num_rows, rows_used
):
    # num_cols comes from ceil(members / num_rows), which can leave fewer
    # rows than asked for (4 members, num_rows=3 -> 2 columns -> 2 rows).
    # The grid was still sized and centred for num_rows, so it sat half a
    # row off the Group's pivot.
    with Scene(), Off():
        g = Group([Square(size=1) for _ in range(members)])
        g.arrange_in_grid(num_rows, row_buffer=0)
        ys = {round(float(c.location[0, 0, 1]), 4) for c in g.children}
        assert len(ys) == rows_used
        assert float(g.get_center()[0, 0, 1]) == pytest.approx(
            float(g.location[0, 0, 1]), abs=1e-5
        )
