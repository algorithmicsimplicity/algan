"""``shadow_identity_reject`` off renders, through both of its launch sites.

The switch is the A/B control for identity-aware shadow rejection
(``test_shadow_identity_epsilon.py`` pins its policy). Off, the shadow trace is
handed a dummy for ``tri_obj``, and the dummy was flat: but the kernel binds
``tri_obj`` through the arena as the [frames, triangles] table it is, so every
shadowed render with the switch off raised ``ArenaBindingError`` ("tri_obj has 1
dimensions, but the kernel's binding reads 2") before drawing anything. Two
call sites passed it: the sheet resolve's events for primary hits
(``raster_pipeline``) and the drain loop's deferred events for bounce hits
(``tracer``). A mirror under a lit cube reaches both -- the mirror's own hits
are primaries, the cube seen in it is a bounce -- and the test counts the
launches of each, so it cannot pass without reaching them.

Off is the pre-identity acceptance (the absolute ``min_hit_distance`` floor on
every hit). Nothing in this scene is judged differently by that floor and the
scene-scaled one, so the two arms must draw the same frame.

Unmarked, so outside ``--fast``: only the identity switch can break it.
"""

import torch

from algan import (
    BLACK,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    Cube,
    DirectionalLight,
    MeshLambertMaterial,
    MeshStandardMaterial,
    Off,
    Scene,
    Surface,
    VideoSettings,
)
from algan.rendering.raytracing import raster_taichi

VIDEO = VideoSettings(resolution=(160, 90), frames_per_second=10, supersampling=1)


def _frame(identity, monkeypatch):
    """The frame, and how many shadow traces ran for primary and for bounce hits."""
    launches = {0: 0, 1: 0}
    trace = raster_taichi.raster_shadow_trace
    secondary = raster_taichi._RASTER_SHADOW_TRACE_PARAMS.index("secondary")

    def counted(*args):
        launches[int(args[secondary])] += 1
        return trace(*args)

    monkeypatch.setattr(raster_taichi, "raster_shadow_trace", counted)
    SETTINGS.raytracing.set(shadows=True)
    SETTINGS.raytracing.experimental.shadow_identity_reject = identity
    with Scene(video_settings=VIDEO) as scene:
        scene.set_background(BLACK)
        with Off():
            Scene.clear_lights()
            DirectionalLight(location=UP * 7 + LEFT * 5 + IN * 3, target=UP).spawn()
            mirror = Surface(
                coord_function=lambda uv: torch.cat(
                    (
                        (uv[..., :1] - 0.5) * 6,
                        torch.zeros_like(uv[..., :1]),
                        (uv[..., 1:2] - 0.5) * 6,
                    ),
                    -1,
                ),
                grid_width=2,
                grid_height=2,
                color=WHITE,
            ).set_material(MeshStandardMaterial(color=WHITE, metalness=1, roughness=0))
            mirror.two_sided = True
            mirror.rotate(17, RIGHT + OUT * 0.37, about=ORIGIN)
            mirror.spawn()
            Cube(color=WHITE).set_material(MeshLambertMaterial()).scale(0.6).move_to(
                UP * 1.2
            ).spawn()
            scene.camera.rotate(-40, RIGHT, about=ORIGIN)
            scene.camera.rotate(30, UP, about=ORIGIN)
        frame = torch.cat([f.cpu() for f in scene.get_frames(0, 1)]).float()[0, ..., :3]
    monkeypatch.undo()
    return frame, launches


def test_identity_rejection_off_renders_the_same_frame(monkeypatch):
    on, _ = _frame(True, monkeypatch)
    off, launches = _frame(False, monkeypatch)
    assert launches[0] > 0, "the mirror's own hits should trace shadows"
    assert launches[1] > 0, "the cube seen in the mirror should trace shadows"
    assert float((on - off).abs().max()) <= 2
