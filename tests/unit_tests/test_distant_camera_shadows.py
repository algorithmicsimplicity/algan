"""Shadows under a distant camera: no shadow ray starts below the surface it shades.

A near-orthographic camera (``Camera.set_near_orthographic``) puts its eye thousands of
units out, where float32 positions are about 1e-3 apart. A point formed as
``eye + t * dir`` is rounded that coarsely in every direction, off the surface as well
as along it, while a shadow ray's origin is lifted only 256 ULPs of the scene's own
coordinates off it (about 6e-5 at unit scale; ``transport_taichi._offset_ray_origin``).
Three places formed shadow origins that way: the sheet route's sub-pixel footprint
(``raster_taichi._pixel_footprint``), the classic wavefront's shadow position (the inline
fan, which shades its primary hits), and the path tracer's hit point. Their origins
sank below the surface and the rays hit the surface they had left, so an evenly lit
plane came out speckled -- a quarter or a half of a pixel's light lost (sheet route),
whole pixels black (wavefront), the plane a quarter darker overall (path tracer) --
and, under a moving camera, the speckle was re-rolled every frame (the
backpropagation video's loss landscape, its eye 16,000 units out).

Each route renders one lit plane twice, from the default camera and from 10,000 units
out, and the far plane must be as evenly lit as the near one. The plane is tilted on
purpose: an axis-aligned plane's normal is exact, its dot products round nothing, and
it stayed clean before the fix. Before it, the far renders here had 79% (sheet) and
47% (wavefront) of the plane's pixels off, by up to 56 and 187 levels, and the path
tracer's plane was 35 levels darker.

Unmarked, so outside ``--fast``: only the shadow paths named above can break it.
"""

import pytest
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
    DirectionalLight,
    MeshLambertMaterial,
    Off,
    Scene,
    Surface,
    VideoSettings,
)

VIDEO = VideoSettings(resolution=(160, 90), frames_per_second=10, supersampling=1)
FAR = 10000.0

ROUTES = {
    "sheet": {"samples": 1, "experimental": {}},
    "classic wavefront": {"samples": 1, "experimental": {"hybrid_raster": False}},
    "path tracer": {"samples": 4, "experimental": {}},
}


def _plane():
    plane = Surface(
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
    ).set_material(MeshLambertMaterial())
    plane.two_sided = True
    plane.rotate(17, RIGHT + OUT * 0.37, about=ORIGIN)
    return plane


def _plane_values(route, distance):
    """The plane's interior pixels (mean of RGB), rendered by ``route``.

    ``distance`` is how far out the near-orthographic eye goes; 0 keeps the
    default camera.
    """
    config = ROUTES[route]
    SETTINGS.raytracing.set(
        shadows=True, samples_per_pixel=config["samples"], denoise=False
    )
    for name, value in config["experimental"].items():
        setattr(SETTINGS.raytracing.experimental, name, value)
    with Scene(video_settings=VIDEO) as scene:
        scene.set_background(BLACK)
        with Off():
            Scene.clear_lights()
            DirectionalLight(location=UP * 7 + LEFT * 5 + IN * 3, target=UP).spawn()
            _plane().spawn()
            camera = scene.camera
            if distance:
                camera.set_near_orthographic(distance=distance)
            camera.rotate(-40, RIGHT, about=ORIGIN)
            camera.rotate(30, UP, about=ORIGIN)
        frame = torch.cat([f.cpu() for f in scene.get_frames(0, 1)]).float()[0]
    rgb = frame[..., :3]
    lit = (rgb.amax(-1) > 8).float()
    # Interior: lit pixels all eight of whose neighbours are lit, so no silhouette.
    unlit_near = torch.nn.functional.max_pool2d((1 - lit)[None, None], 3, 1, padding=1)[
        0, 0
    ]
    interior = (lit > 0) & (unlit_near == 0)
    assert int(interior.sum()) > 2000, "the plane should fill much of the frame"
    return rgb[interior].mean(-1)


@pytest.mark.parametrize("route", list(ROUTES))
def test_a_distant_camera_lights_a_plane_as_evenly_as_a_near_one(route):
    near = _plane_values(route, 0.0)
    far = _plane_values(route, FAR)
    if ROUTES[route]["samples"] == 1:
        # Deterministic: one value across the whole plane, the near camera's.
        level = near.median()
        assert float((near - level).abs().max()) <= 2
        off = (far - level).abs()
        assert float(off.max()) <= 2, (
            f"{int((off > 2).sum())} of {far.numel()} pixels of an evenly lit plane are "
            f"off by up to {float(off.max()):.0f} levels under the distant camera"
        )
    else:
        # The path tracer samples, so compare distributions: its plane was a
        # quarter darker overall from the far camera.
        assert abs(float(far.median()) - float(near.median())) <= 2
        assert float(torch.quantile(far, 0.02)) >= float(torch.quantile(near, 0.02)) - 6
