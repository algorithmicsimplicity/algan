"""Shading defects the sheet resolve review found, each pinned by a render.

Each case renders through the sheet route, once with one input corrupted, and
asserts on the frame:

* a clear glass sphere on a white background lost the light passing beside it
  at its silhouette, a dark one-pixel outline; and once that was fixed, its far
  sheet still re-claimed coverage the near one had left to the background,
  because the one-mesh ceiling bound on opaque meshes only;
* a NaN sheet area painted its pixel opaque;
* an environment map brighter than 1 wrapped round in the byte frame buffer;
* a coloured transmitter over part of a pixel tinted some of what it did not
  cover.

Feature tests of the resolve, which nothing outside it can break, so none is
marked fast.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from algan import (
    BLACK,
    OUT,
    PURE_GREEN,
    PURE_RED,
    RED,
    RIGHT,
    WHITE,
    Off,
    Scene,
    Sphere,
    Square,
    VideoSettings,
)
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.rendering.raytracing import raster_pipeline, sheet_resolve_taichi
from algan.rendering.raytracing.glossy_prefilter_taichi import ACC_GEO
from algan.rendering.raytracing.raster_taichi import (
    _AA_NUM_SAMPLES as NUM_SAMPLES,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_ONE_SOLID_BIT as ONE_SOLID,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import min_alpha
from algan.rendering.raytracing.sheets import resolve_pixel_reference
from algan.rendering.shaders.materials import MeshBasicMaterial, MeshPhysicalMaterial
from algan.settings import SETTINGS


def _luminance(path):
    """Mean of the encoded RGB channels, per pixel, as float."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    return im.mean(-1)


def _clear_glass(**kw):
    return MeshPhysicalMaterial(
        color=WHITE, roughness=0.0, metalness=0.0, transmission=1.0, ior=1.5, **kw
    )


#: The emission's one-mesh block, held before any test wraps it.
_ONE_MESH_PIXEL_CAPS = raster_pipeline._one_mesh_pixel_caps


def _force_one_solid(monkeypatch, solid):
    """Make the emission flag one-solid pixels (``solid``) or not, whatever the
    batch says; ``False`` is the emission as it was before the flag existed.
    Each call replaces the last: the wrapper calls the unwrapped block.
    """

    def forced(*args, **kwargs):
        kwargs["solid"] = solid
        forced.calls += 1
        return _ONE_MESH_PIXEL_CAPS(*args, **kwargs)

    forced.calls = 0
    monkeypatch.setattr(raster_pipeline, "_one_mesh_pixel_caps", forced)
    return forced


def _render_clear_glass(path, monkeypatch):
    """A clear sphere on white at 192x108. Returns the luminance, and the
    resolve's sheet records and leftover geometric share per covered pixel,
    read straight after its shading launch -- before any continuation adds to
    the accumulator.
    """
    real = sheet_resolve_taichi.sheet_resolve_shade
    params = sheet_resolve_taichi._SHEET_RESOLVE_SHADE_PARAMS
    records = {}

    def recorded(*args):
        launch = dict(zip(params, args))
        if int(launch["mode"]) != 0 or records:
            return real(*args)
        out = real(*args)
        n = int(launch["num_covered"])
        records["geo"] = launch["pix_accum"][:n, ACC_GEO].cpu().tolist()
        for name in ("sheet_offsets", "sheet_cov", "sheet_msk", "sheet_cap"):
            records[name] = launch[name].cpu().tolist()
        return out

    monkeypatch.setattr(sheet_resolve_taichi, "sheet_resolve_shade", recorded)
    SETTINGS.raytracing.set(shadows=False)
    with Scene() as scene:
        Sphere(radius=1.4).set_material(_clear_glass()).spawn(animate=False)
        scene.save_frame(
            str(path), VideoSettings((192, 108), 1, supersampling=1), background=WHITE
        )
    assert records, "the render made no shading launch to read"
    return _luminance(path), records


def test_clear_glass_on_white_has_no_dark_rim(tmp_path, monkeypatch):
    """A clear sphere on a white background shows no dark outline.

    With no lights, every continuation a smooth white glass sphere spawns --
    reflection, refraction, and the light passing beside it in a partially
    covered silhouette pixel -- ends on the same white, so the sphere should
    all but vanish: a 4x supersampled render of it (``analytic_aa`` off, the
    wavefront fallback) never drops below 250.

    Two defects darkened the rim. The glass branch made the primary ray the
    reflection whenever ``alpha * R >= 1 - alpha``, which a grazing silhouette
    pixel usually satisfies, and dropped the pass-through: darkest pixel 204,
    with 18 pixels below 230. Then the far sheet re-claimed the near sheet's
    ``corr < 1`` residue -- the one-mesh ceiling bound on opaque meshes only --
    and spent it on a ray into the sphere through its back face, which at that
    grazing angle totally reflects: 16 pixels below 245 and 26 below 250. Now
    2 and 4. What remains (darkest 238, at a pixel whose far sheet claims
    nothing) is the near sheet's one shading point at the limb, against the
    reference's many.
    """
    lum, _records = _render_clear_glass(tmp_path / "glass_on_white.png", monkeypatch)
    dark = int((lum < 225).sum())
    assert dark == 0, (
        f"{dark} pixels of a clear sphere on white are darker than 225 "
        f"(darkest {lum.min():.0f}): the silhouette is dropping its pass-through"
    )
    dim = int((lum < 245).sum())
    assert dim <= 4, (
        f"{dim} pixels of a clear sphere on white are darker than 245 "
        f"(darkest {lum.min():.0f}): its far sheet is claiming the near one's "
        "residue (16 did before the one-solid ceiling)"
    )


def _oracle_background(covs, msks, caps, refracted):
    """What the oracle leaves the background on one pixel of matte-alpha
    sheets, every one of them refracted or none.
    """
    _claims, T = resolve_pixel_reference(
        covs,
        msks,
        [False] * len(covs),
        [1.0] * len(covs),
        caps=caps,
        min_alpha=min_alpha,
        refracts=[refracted] * len(covs),
    )
    return sum(T) / NUM_SAMPLES


def test_the_refracting_resolve_follows_the_oracle_on_clear_glass(
    tmp_path, monkeypatch
):
    """The resolve's refraction variant composites a glass solid's coverage as
    ``sheets.resolve_pixel_reference`` says, one-solid ceiling included.

    ``test_sheet_resolve_parity`` holds the kernel to the oracle on unlit
    translucent triangles, which never refract, so the ceiling's binding half
    -- behind a sheet the walk refracted at full alpha -- needs glass in the
    scene. Here every sheet of the clear sphere refracts, so on each
    silhouette pixel the leftover geometric share the walk commits (what no
    sheet claimed) must be the oracle's background share with every sheet
    marked refracted: within 2e-5 on all of them, and on many it is not what
    the oracle says without the rule.
    """
    _lum, rec = _render_clear_glass(tmp_path / "glass.png", monkeypatch)
    off = rec["sheet_offsets"]
    compared = moved = 0
    worst = (0.0, None)
    for p, geo in enumerate(rec["geo"]):
        a, b = off[p], off[p + 1]
        sheets = [rec[k][a:b] for k in ("sheet_cov", "sheet_msk", "sheet_cap")]
        assert all(m & ONE_SOLID for m in sheets[1]), f"pixel {p} is not one solid"
        expected = _oracle_background(*sheets, refracted=True)
        # Only a walk that kept its primary: a pixel the glass fully covers
        # turns it into the reflection, which commits no geometric share.
        if not 0.01 < expected < 0.99:
            continue
        compared += 1
        gap = abs(geo - expected)
        if gap > worst[0]:
            worst = (gap, p)
        if abs(_oracle_background(*sheets, refracted=False) - expected) > 1e-3:
            moved += 1
    assert compared > 100, f"only {compared} silhouette pixels to compare"
    assert worst[0] <= 2e-5, f"pixel {worst[1]} is {worst[0]:.3g} off the oracle"
    assert moved > 10, f"the ceiling changed only {moved} of {compared} pixels"


def test_solids_the_walk_does_not_refract_render_as_before(tmp_path, monkeypatch):
    """The one-solid flag changes nothing until the walk refracts at a
    full-alpha sheet, so a solid it does not refract renders exactly as it did
    without the flag -- and the clear glass beside them does not.

    Five spheres on white in one batch that refracts: clear glass, opaque,
    translucent, glass at opacity 0.6 and red glass at an index of 1 (a plain
    transmitter). Each is a solid whose far sheet shows THROUGH its near one --
    the opaque one excepted, whose own one-mesh ceiling the flag must not
    disturb -- so the ceiling must not bind: every pixel outside the clear
    sphere is byte-identical with the flag forced off. Then the clear sphere
    alone with no bounces left, flagged regardless: the walk refracts nothing,
    and the frame is identical too.
    """

    def five_spheres(path, solid):
        forced = _force_one_solid(monkeypatch, solid)
        with Scene() as scene:
            spheres = [
                Sphere(radius=0.9).set_material(_clear_glass()),
                Sphere(radius=0.9, color=RED),
                Sphere(radius=0.9, color=RED).set_opacity(0.5),
                Sphere(radius=0.9).set_material(_clear_glass()).set_opacity(0.6),
                Sphere(radius=0.9).set_material(
                    MeshPhysicalMaterial(
                        color=RED,
                        roughness=0.0,
                        metalness=0.0,
                        transmission=1.0,
                        ior=1.0,
                    )
                ),
            ]
            for sphere, x in zip(spheres, (-5.6, -2.8, 0.0, 2.8, 5.6)):
                sphere.move_to(RIGHT * x).spawn(animate=False)
            scene.save_frame(
                str(path),
                VideoSettings((192, 108), 1, supersampling=1),
                background=WHITE,
            )
        assert forced.calls, "the emission's one-mesh block never ran"
        return np.asarray(Image.open(path).convert("RGB")).astype(int)

    SETTINGS.raytracing.set(shadows=False)
    flagged = five_spheres(tmp_path / "flagged.png", True)
    plain = five_spheres(tmp_path / "plain.png", False)
    moved = np.abs(flagged - plain).max(-1) > 0
    # The clear sphere spans columns 8..32 at this size; the next starts at 46.
    assert moved[:, :40].any(), "the flag did not reach the clear glass sphere"
    stray = np.argwhere(moved[:, 40:])
    assert not len(stray), (
        f"{len(stray)} pixels of solids the walk does not refract moved, first "
        f"at (x, y) = ({stray[0][1] + 40}, {stray[0][0]})"
    )

    SETTINGS.raytracing.set(max_bounces=0)
    lone = []
    for solid in (True, False):
        forced = _force_one_solid(monkeypatch, solid)
        path = tmp_path / f"no_bounces_{solid}.png"
        with Scene() as scene:
            Sphere(radius=1.4).set_material(_clear_glass()).spawn(animate=False)
            scene.save_frame(
                str(path), VideoSettings((96, 54), 1, supersampling=1), background=WHITE
            )
        assert forced.calls, "the emission's one-mesh block never ran"
        lone.append(path.read_bytes())
    assert lone[0] == lone[1], "glass with no bounces left moved under the flag"


def test_a_nan_sheet_area_paints_nothing(tmp_path, monkeypatch):
    """A sheet whose exact area is NaN claims nothing.

    The resolve clamps an area with ``min(area, 1)``, and on the CPU backend
    that returns 1 for a NaN, so one bad area painted its pixel opaque. Nothing
    upstream produces one today (the compaction clamps its areas), so this hands
    the resolve NaN for EVERY sheet of a white square on black. Before, all 460
    covered pixels came out pure white, silhouette pixels included; now none is
    painted. The oracle, ``sheets.resolve_pixel_reference``, drops a NaN area
    the same way (``test_sheet_compaction``).
    """
    from algan.rendering.raytracing import raster_pipeline

    launch = raster_pipeline.shade_sparse_raster_coverage
    poisoned = []

    def with_nan_areas(coverage, *args, **kwargs):
        coverage["sheet_cov"].fill_(float("nan"))
        poisoned.append(int(coverage["sheet_cov"].numel()))
        return launch(coverage, *args, **kwargs)

    monkeypatch.setattr(raster_pipeline, "shade_sparse_raster_coverage", with_nan_areas)
    SETTINGS.raytracing.set(shadows=False)
    out = tmp_path / "nan_areas.png"
    with Scene() as scene:
        Square(color=WHITE).scale(1.5).rotate(17, OUT).spawn(animate=False)
        scene.save_frame(
            str(out), VideoSettings((96, 54), 1, supersampling=1), background=BLACK
        )
    assert max(poisoned, default=0) > 100, "the resolve never saw the square"
    lum = _luminance(out)
    painted = int((lum > 0).sum())
    assert painted == 0, f"{painted} pixels were painted by sheets with NaN areas"


def test_a_bright_environment_saturates_the_byte_buffer(tmp_path):
    """An environment map brighter than 1 shows as white on the byte buffer.

    With post-process tonemapping off the frame buffer is uint8, and the sheet
    route prefills it with the environment map, pixel by pixel. The prefill
    stored ``value * 255`` unclamped, so a map at 1.5 (382) wrapped to 126: a
    grey sky behind a scene lit by a white one. It now rounds and clamps the
    way the host's solid-colour prefill does. ``linear_color_space`` has to be
    off as well, because the linear space refuses the byte buffer outright.
    """
    SETTINGS.raytracing.set(shadows=False, linear_color_space=False, tonemapping=False)
    SETTINGS.raytracing.experimental.set(post_process_tonemap=False)
    out = tmp_path / "bright_sky.png"
    with Scene() as scene:
        scene.set_environment_map(torch.full((4, 8, 3), 1.5), ambient=False)
        Sphere(radius=0.5, color=RED).spawn(animate=False)
        scene.save_frame(str(out), VideoSettings((64, 36), 1, supersampling=1))
    im = np.asarray(Image.open(out).convert("RGB"))
    corners = im[[0, 0, -1, -1], [0, -1, 0, -1]]
    assert (corners == 255).all(), (
        f"the sky's corners read {corners.tolist()}, not white: the "
        "environment prefill wrapped in the byte buffer"
    )


def test_a_tinted_transmitter_tints_only_what_it_covers(tmp_path):
    """Over part of a pixel, a coloured transmitter leaves the rest untinted.

    A pure green triangle that passes half its light, over a pure red
    background: no red gets through the green, so the red in the frame is the
    part of each pixel the triangle leaves uncovered, exactly as behind an
    opaque black triangle of the same shape. The walk keeps one visibility per
    sample and carries the tint in the throughput, and it used to blend the
    tint in linearly with the sheet's coverage, which darkened the uncovered
    part too: 82 of the 94 silhouette pixels passed less red than behind black
    by more than 2 levels, by up to 27, before; none after.

    Index 1 keeps the triangle off the glass branch, which refracts through a
    ray of its own instead.
    """
    SETTINGS.raytracing.set(shadows=False)
    corners = torch.tensor([[-2.3, -1.4, 0.0], [2.1, -1.9, 0.0], [0.3, 1.7, 0.0]])
    frames = {}
    for name, material in (
        (
            "tinted",
            MeshPhysicalMaterial(
                color=PURE_GREEN, transmission=0.5, ior=1.0, metalness=0.0
            ),
        ),
        ("black", MeshBasicMaterial(color=BLACK)),
    ):
        out = tmp_path / f"{name}.png"
        with Scene() as scene:
            with Off():
                Scene.clear_lights()
                tri = TriangleTriangulated(corners, color=PURE_GREEN)
                tri.set_material(material)
                tri.spawn(animate=False)
            scene.save_frame(
                str(out),
                VideoSettings((96, 54), 1, supersampling=1),
                background=PURE_RED,
            )
        frames[name] = np.asarray(Image.open(out).convert("RGB")).astype(np.int64)
    red = frames["black"][..., 0]
    silhouette = (red > 5) & (red < 250)
    assert int(silhouette.sum()) > 50, "the triangle's edges are not in frame"
    off = np.abs(frames["tinted"][..., 0] - red)[silhouette]
    assert int((off > 2).sum()) == 0, (
        f"{int((off > 2).sum())} silhouette pixels pass red unlike an opaque "
        f"black twin, by up to {int(off.max())}: the tint reaches past the "
        "sheet's coverage"
    )
