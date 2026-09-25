"""Row-span candidates for circuits (``raster_circuit_span_candidates``).

A long diagonal stroke's screen box is almost all empty, so the sparse
discovery expands a qualifying circuit row by row along its projected outline
instead of chunking its whole box (``sheet_compact_taichi._circuit_reach``).
Only the candidate SET may change: every frame must be bit-identical to the box
expansion's. The scene is chosen to be hostile to the reach bound -- a wide
field of view, strokes and fills on planes tilted far out of the image plane,
thick strokes, a hairline and a curved stroke -- since the stroke's screen
reach grows with both the tilt and the ray's angle off the optical axis.
"""

import math

import numpy as np

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, WHITE
from algan.constants.spatial import DOWN, IN, LEFT, OUT, RIGHT, UP
from algan.mobs.group import Group
from algan.mobs.manim_adapters import Arc
from algan.mobs.shapes_2d import Circle, Line, Polygon, Square
from algan.rendering.raytracing import raster_pipeline
from algan.scene import Scene
from algan.settings import SETTINGS
from algan.settings.video_settings import LD

VIDEO = LD.set(resolution=(320, 180))


def _scene_mobs():
    mobs = []
    for k in range(8):
        line = Line(LEFT * 5, RIGHT * 5, stroke_width=3 + 2 * k, color=WHITE)
        line.rotate(20 * k, OUT).rotate(25 + 6 * k, RIGHT).rotate(12 * k - 45, UP)
        mobs.append(line.move(UP * (k - 4) * 0.5))
    for sx in (-1, 1):
        for sy in (-1, 1):
            disc = Circle(radius=0.9, color=BLUE, fill_opacity=1, stroke_width=6)
            disc.rotate(60 * sx, UP).rotate(55 * sy, RIGHT)
            mobs.append(disc.move_to(RIGHT * 6.2 * sx + UP * 3.4 * sy))
            frame = Square(size=1.2, stroke_width=8, fill_opacity=0).rotate(35, OUT)
            mobs.append(
                frame.rotate(70 * sy, UP).move_to(RIGHT * 4.6 * sx + UP * 2.5 * sy)
            )
    arc = Arc(radius=2.5, start_angle=0, angle=288, stroke_width=14).rotate(50, RIGHT)
    mobs.append(arc.move_to(LEFT * 3.5 + DOWN * 2.8))
    star = Polygon(
        *[
            (RIGHT * math.cos(a) + UP * math.sin(a)) * (1.4 if i % 2 == 0 else 0.6)
            for i, a in enumerate(np.linspace(0, 2 * math.pi, 10, endpoint=False))
        ],
        fill_opacity=1,
        stroke_width=4,
    )
    mobs.append(star.rotate(65, UP).move_to(RIGHT * 3.8 + DOWN * 3.0 + IN))
    mobs.append(Line(LEFT * 9 + UP * 0.05, RIGHT * 9 + DOWN * 0.05, stroke_width=0.5))
    return mobs


def _render(tmp_path, spans, monkeypatch):
    rows = []
    real = raster_pipeline._pair_expand_rows

    def counted(*args, **kwargs):
        out = real(*args, **kwargs)
        rows.append(0 if out is None else int(out.shape[0]))
        return out

    monkeypatch.setattr(raster_pipeline, "_pair_expand_rows", counted)
    path = tmp_path / f"spans_{int(spans)}.png"
    with (
        SETTINGS.raytracing.experimental.override(raster_circuit_span_candidates=spans),
        Scene() as scene,
    ):
        with Off():
            scene.get_camera().set_fov(100)
            mobs = _scene_mobs()
            Group(*mobs).scale(2.4)
        for mob in mobs:
            mob.spawn(animate=False)
        scene.save_frame(str(path), video_settings=VIDEO)
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image).copy(), sum(rows)


def test_circuit_spans_render_exactly_the_box_frame(tmp_path, monkeypatch):
    boxes, box_rows = _render(tmp_path, False, monkeypatch)
    spans, span_rows = _render(tmp_path, True, monkeypatch)
    # Something was drawn, and the spans were actually taken.
    assert (boxes.max(-1) > 20).mean() > 0.01
    assert span_rows < box_rows
    assert np.array_equal(spans, boxes)
