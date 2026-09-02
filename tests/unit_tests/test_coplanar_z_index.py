"""``z_index`` as the coplanar draw order for 2-D circuits.

Exactly coplanar circuits produce the same hit distance, so the resolve ranks
them by an internal index that follows neither creation order nor hierarchy --
it follows the batch merge's grouping, which splits filled circuits from
stroked ones. That is why a Manim ``Vector`` drawn over a ``NumberPlane`` used
to put its stroked shaft *under* the grid and its filled tip *over* it.

``BezierCircuitCubic.z_index`` is the override: the renderer biases a circuit
toward the camera by that many coplanarity bins, which is enough to decide the
order and far too small to show.
"""

import numpy as np
import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, GREEN, RED, WHITE
from algan.mobs.shapes_2d import Square
from algan.scene import Scene
from algan.settings.video_settings import LD


def _read_frame(path):
    """The rendered PNG as an ``[H, W, C]`` uint8 array."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image)


def _render(tmp_path, name, z_index):
    """Two coplanar unit squares, red under blue, blue given ``z_index``.

    Both sit at z = 0, so nothing but the draw order decides the centre pixel.
    """
    path = tmp_path / f"{name}.png"
    with Scene() as scene:
        with Off():
            red = Square(size=2.0, color=RED, stroke_width=0)
            blue = Square(size=2.0, color=BLUE, stroke_width=0)
            blue.z_index = z_index
        red.spawn(animate=False)
        blue.spawn(animate=False)
        scene.save_frame(str(path), video_settings=LD)
    frame = _read_frame(path)
    height, width = frame.shape[:2]
    return frame[height // 2, width // 2].astype(int)


def test_z_index_defaults_to_zero_and_is_a_plain_scalar():
    with Scene(), Off():
        square = Square()
    assert square.z_index == 0.0
    # Not timeline-backed: it selects between orderings, it is not a pose.
    assert "z_index" not in square.animatable_attrs


def test_setting_z_index_propagates_to_the_whole_sub_hierarchy():
    """Manim's ``set_z_index`` defaults to ``family=True`` and a composite has
    to stack as one thing -- an arrow whose tip and shaft disagree is the bug
    this exists to fix.
    """
    manim = pytest.importorskip("manim")
    from algan.mobs.manim_mob import ManimMob

    arrow = manim.Arrow(start=manim.LEFT, end=manim.RIGHT)
    with Scene(), Off():
        mob = ManimMob(arrow)
    assert mob.submobjects, "an Arrow carries its tip as a submobject"

    mob.z_index = 3
    assert mob.z_index == 3.0
    assert all(sub.z_index == 3.0 for sub in mob.submobjects)


def test_manim_z_index_is_carried_across_on_import():
    manim = pytest.importorskip("manim")
    from algan.mobs.manim_mob import ManimMob

    source = manim.Square()
    source.set_z_index(4)
    with Scene(), Off():
        mob = ManimMob(source)
    assert mob.z_index == 4.0


def test_z_index_decides_which_coplanar_circuit_draws_in_front(tmp_path):
    """The behavioural guarantee: same plane, same depth, order by z_index."""
    tied = _render(tmp_path, "tied", 0)
    lifted = _render(tmp_path, "lifted", 1)
    sunk = _render(tmp_path, "sunk", -1)

    # Whichever way the tie falls by default, one bin either side is decisive.
    assert lifted[2] > lifted[0], f"z_index=1 must put blue in front, got {lifted}"
    assert sunk[0] > sunk[2], f"z_index=-1 must put blue behind, got {sunk}"
    assert not np.array_equal(lifted, sunk)
    assert np.array_equal(tied, lifted) or np.array_equal(tied, sunk)


def test_one_bin_of_bias_does_not_move_the_geometry(tmp_path):
    """The bias is a tie-break, not a transform.

    A circuit lifted by one bin must occupy the same pixels it did -- only the
    contested ones may change. Rendered against an empty background so nothing
    else can win a pixel, the silhouette has to be identical.
    """

    def silhouette(name, z_index):
        path = tmp_path / f"{name}.png"
        with Scene() as scene:
            with Off():
                square = Square(size=2.0, color=WHITE, stroke_width=0)
                square.z_index = z_index
            square.spawn(animate=False)
            scene.save_frame(str(path), video_settings=LD)
        return torch.from_numpy(_read_frame(path).copy())[..., :3].sum(-1)

    base = silhouette("bias_off", 0)
    lifted = silhouette("bias_on", 1)
    assert torch.equal((base > 0), (lifted > 0))


def test_the_bias_runs_along_each_circuits_own_eye_ray():
    """A large bias must not move a circuit on screen, wherever it sits in frame.

    ``z_index`` was written for single tie-bins -- 1e-4 world units, small
    enough that any direction hides the difference. A non-planar patch's border
    asks for ``BORDER_DEPTH_BIAS_BINS_PER_UNIT`` (2e4) bins per world unit of
    its patch's bulge, i.e. whole world units, and sliding every circuit along
    ONE shared axis is a translation: that preserves screen position only for
    geometry on that axis, and off-axis it magnifies about the screen centre.
    A saddle tile's stroke landed 7.2 px outboard of the outline it drew.
    Displaced along its own eye ray a circuit's centre cannot move at all.
    """
    import torch.nn.functional as F

    from algan.rendering.raytracing.primitives import RayTracedBezierCircuitPrimitive

    prim = RayTracedBezierCircuitPrimitive.__new__(RayTracedBezierCircuitPrimitive)
    prim._has_z_index = True
    # One circuit on the optical axis and two well off it, which is where the
    # shared-axis translation used to slide the geometry.
    centres = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 2.0, 0.0], [-4.0, 1.5, 1.0]]])
    prim.mob_center = centres.clone()
    prim.z_index = torch.full((1, 3, 1), 1.0e4)  # one world unit of bias
    prim.num_segments_per_object = torch.ones(3, dtype=torch.long)
    corners = centres.unsqueeze(-2).expand(-1, -1, 4, -1).clone()
    cam_o = torch.tensor([[0.0, 0.0, 20.0]])
    sp = torch.tensor([[0.0, 0.0, 10.0]])

    moved = prim._apply_z_index_bias(corners, cam_o, sp)

    before = centres - cam_o.view(-1, 1, 3)
    after = moved[..., 0, :] - cam_o.view(-1, 1, 3)
    # Same ray out of the eye is the same pixel, however far along it the
    # circuit ends up.
    parallel = torch.cross(
        F.normalize(before, p=2, dim=-1), F.normalize(after, p=2, dim=-1), dim=-1
    )
    assert float(parallel.abs().max()) < 1e-5
    # And it did move, toward the camera, by the bias it asked for.
    closer = before.norm(p=2, dim=-1) - after.norm(p=2, dim=-1)
    assert float(closer.min()) > 0.99
    assert float(closer.max()) < 1.01


def _stack_centre_pixel(tmp_path, name, order):
    """Author coplanar shapes that all cover the frame centre, in ``order``.

    Each entry is ``("fill" | "stroke", colour)``. The two kinds land in
    different merge blocks, which is the split that used to scatter a
    composite Mob's parts across whatever they overlapped.
    """
    from algan.mobs.shapes_2d import Line

    path = tmp_path / f"{name}.png"
    with Scene() as scene:
        with Off():
            mobs = []
            for kind, colour in order:
                if kind == "fill":
                    mobs.append(Square(size=1.6, color=colour, stroke_width=0))
                else:
                    mobs.append(
                        Line(
                            start=(-1.5, 0.0, 0.0),
                            end=(1.5, 0.0, 0.0),
                            color=colour,
                            stroke_width=40,
                        )
                    )
        for mob in mobs:
            mob.spawn(animate=False)
        scene.save_frame(str(path), video_settings=LD)
    frame = _read_frame(path)
    height, width = frame.shape[:2]
    return frame[height // 2, width // 2].astype(int)


def _dominant(pixel):
    return int(np.argmax(pixel[:3]))


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        # Last authored wins, whichever block it lands in. Both sequences
        # alternate blocks, so neither can be satisfied by the merged layout
        # alone -- this is what the resolved draw order buys.
        ((("stroke", RED), ("fill", GREEN), ("stroke", BLUE)), 2),
        ((("fill", RED), ("stroke", GREEN), ("fill", BLUE)), 2),
        ((("stroke", BLUE), ("fill", GREEN), ("stroke", RED)), 0),
        ((("fill", BLUE), ("stroke", GREEN), ("fill", RED)), 0),
    ],
)
def test_coplanar_shapes_draw_in_author_order(tmp_path, order, expected):
    """The default guarantee: coplanar 2-D geometry stacks as it was authored.

    ``expected`` is the RGB channel index of the last-authored colour.
    """
    name = "_".join(k for k, _ in order) + str(expected)
    pixel = _stack_centre_pixel(tmp_path, name, order)
    assert _dominant(pixel) == expected, (
        f"last-authored colour must win the shared pixel, got {pixel}"
    )


def test_draw_order_walks_each_tree_whole_rather_than_by_depth():
    """Roots in creation order, parent before children, trees not interleaved.

    The depth-descending sort this replaced kept parents ahead of children but
    put a deep node of one tree ahead of a shallow node of another, which is
    how an arrow ended up straddling a grid it crossed.
    """
    manim = pytest.importorskip("manim")
    from algan.mobs.manim_mob import ManimMob

    with Scene() as scene:
        with Off():
            grid = ManimMob(manim.VGroup(*(manim.Line() for _ in range(4))))
            arrow = ManimMob(manim.Arrow(start=manim.LEFT, end=manim.RIGHT))
        rank, _ = scene._authored_draw_order()

    def ranks(mob):
        out = [rank[id(mob)]]
        for sub in mob.submobjects:
            out.extend(ranks(sub))
        return out

    grid_ranks, arrow_ranks = ranks(grid), ranks(arrow)
    assert max(grid_ranks) < min(arrow_ranks), (
        "the whole first tree must precede the whole second"
    )
    assert rank[id(arrow)] < min(rank[id(s)] for s in arrow.submobjects), (
        "a parent must precede its children (Manim's family pre-order)"
    )


def test_draw_bias_costs_block_alternations_not_mobs():
    """The bias budget is what keeps the displacement invisible.

    A run of same-block circuits shares one bias -- only a block boundary
    costs a bin -- so the span stays far below the circuit count.
    """
    with Scene() as scene:
        with Off():
            same_block = [Square(stroke_width=0, color=RED) for _ in range(8)]
            alternating = []
            from algan.mobs.shapes_2d import Line

            for i in range(8):
                alternating.append(
                    Square(stroke_width=0, color=RED)
                    if i % 2
                    else Line(color=BLUE, stroke_width=4)
                )
        _, bias = scene._authored_draw_order()

    run = [bias[id(m)] for m in same_block]
    assert len(set(run)) == 1, f"one block must cost one bias, got {sorted(set(run))}"

    span = max(bias.values()) - min(bias.values())
    assert span < len(bias), f"bias span {span} must stay under the circuit count"
    alt = [bias[id(m)] for m in alternating]
    assert len(set(alt)) > 1, "alternating blocks must be separated"
