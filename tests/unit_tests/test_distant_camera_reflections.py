"""A mirror under a distant camera reflects what it faces, in every route.

A primary hit formed as ``ro + t_hit * rd`` is rounded at the eye's magnitude,
in every direction including off the surface, and ``set_near_orthographic``
puts the eye thousands of units out, where float32 positions are ~1e-3 apart:
far more than the lift a reflection or refraction starts with. Spawned from
there, a continuation began under the mirror and met it again, and a grey
mirror reflecting a plain background speckled with its own double
reflection: up to 57 levels off on 45% of the mirror in the classic wavefront
route, and up to 10-20 on a reflective 2-D circuit in every route -- through
the classic route's continuation sites, the sheet route's circuit hit point,
and the sub-pixel re-samples (``_jittered_surface_sample``) the sheet route
and the path tracer spawn their circuit reflections from.

Each route renders the mirror at the default camera and at distance 10,000;
the far mirror's pixels must all show the near one's single reflection. The
flat-green twin of each scene masks the mirror, less a pixel's antialiased
border. Unmarked, so outside ``--fast``: only the renderer's spawn points can
break it.
"""

import pytest
import torch

from algan import (
    BLUE,
    GREEN,
    GREY,
    ORIGIN,
    OUT,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    MeshBasicMaterial,
    MeshStandardMaterial,
    Off,
    Scene,
    Square,
    Surface,
    VideoSettings,
)

VIDEO = VideoSettings(resolution=(160, 90), frames_per_second=10, supersampling=1)
ROUTES = {
    "sheet": {"samples": 1, "experimental": {}},
    "classic wavefront": {"samples": 1, "experimental": {"hybrid_raster": False}},
    "path tracer": {"samples": 4, "experimental": {}},
}
_MASKS = {}


def _mirror(kind, material):
    if kind == "circuit":
        mirror = Square(size=5, color=WHITE, fill_opacity=1, stroke_width=0)
        mirror.set_material(material)
        mirror.rotate(-90, RIGHT)
    else:
        mirror = Surface(
            coord_function=lambda uv: torch.cat(
                (
                    (uv[..., :1] - 0.5) * 5,
                    torch.zeros_like(uv[..., :1]),
                    (uv[..., 1:2] - 0.5) * 5,
                ),
                -1,
            ),
            grid_width=2,
            grid_height=2,
            color=WHITE,
        ).set_material(material)
        mirror.two_sided = True
    return mirror


def _frame(kind, distance, material, route="sheet"):
    """One frame of the tilted mirror against a plain background."""
    config = ROUTES[route]
    SETTINGS.raytracing.set(shadows=False, samples_per_pixel=config["samples"])
    for name, value in config["experimental"].items():
        setattr(SETTINGS.raytracing.experimental, name, value)
    with Scene(video_settings=VIDEO) as scene:
        scene.set_background(BLUE)
        with Off():
            mirror = _mirror(kind, material)
            mirror.rotate(17, RIGHT + OUT * 0.37, about=ORIGIN)
            mirror.spawn()
            if distance:
                scene.camera.set_near_orthographic(distance=distance)
            scene.camera.rotate(-40, RIGHT, about=ORIGIN)
            scene.camera.rotate(30, UP, about=ORIGIN)
        return torch.cat([f.cpu() for f in scene.get_frames(0, 1)]).float()[0, ..., :3]


def _mask(kind, distance):
    """The mirror's pixels, from its flat-green twin, less a pixel's border."""
    if (kind, distance) not in _MASKS:
        flat = _frame(kind, distance, MeshBasicMaterial(color=GREEN))
        inside = (flat - flat[0, 0]).abs().amax(-1) > 30
        core = inside.clone()
        core[1:] &= inside[:-1]
        core[:-1] &= inside[1:]
        core[:, 1:] &= inside[:, :-1]
        core[:, :-1] &= inside[:, 1:]
        _MASKS[kind, distance] = core
    return _MASKS[kind, distance]


@pytest.mark.parametrize("kind", ["mesh", "circuit"])
@pytest.mark.parametrize("route", list(ROUTES))
def test_a_mirror_under_a_distant_camera_reflects_once(route, kind):
    grey_mirror = MeshStandardMaterial(color=GREY, metalness=1, roughness=0)
    near = _frame(kind, 0, grey_mirror, route)[_mask(kind, 0)]
    far = _frame(kind, 10000, grey_mirror, route)[_mask(kind, 10000)]
    assert near.shape[0] > 1000
    assert far.shape[0] > 1000
    reflection = near.median(0).values
    # The near mirror is one colour: the background, reflected once.
    assert float((near - reflection).abs().max()) <= 3
    off = (far - reflection).abs().amax(-1)
    assert float(off.max()) <= 3, (
        f"{int((off > 3).sum())} of {far.shape[0]} mirror pixels off by up to "
        f"{float(off.max()):.0f}"
    )
