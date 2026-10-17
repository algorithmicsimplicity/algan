"""``forced_mirror_scatter``: the example custom scatter, against the scatter contract.

``fragment_shaders.forced_mirror_scatter`` is the documented complete example of
a custom ray continuation (``FragmentStage(..., scatter=...)``): commit 15% of
the shaded colour, bounce 85% along the mirror direction. This file pins what it
hands the renderer, read through the contract in ``shading_taichi``:

* an opaque hit commits ``0.15 * alpha`` of the shaded colour and bounces
  ``0.85 * alpha`` along the mirror of the ray about the shading normal, and
  absorbs once the ray is out of bounces;
* the bounce leaves the surface on the side the ray arrived from, even where a
  smooth-shaded silhouette tips the interpolated normal past the facet;
* the bounce origin is lifted by a scale-aware epsilon -- clear of the surface
  at large coordinates, and far closer than a fixed ``1e-3`` at small ones;
* coverage is honoured: the ``1 - alpha`` that misses a partly covering surface
  continues through it, in the probe and in a rendered frame.

The probe calls the scatter ``@ti.func`` directly; the render is one SMOKE_TEST
frame of a half-covering mirror in front of a red wall. A feature test, outside
the fast suite: only a change to this scatter or to the contract breaks it.

Note the absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time, and stringifying
them stops the kernel compiling.
"""

import cv2
import pytest
import torch
import torch.nn.functional as F

from algan import (
    BLACK,
    IN,
    OUT,
    PURE_GREEN,
    PURE_RED,
    SMOKE_TEST,
    Off,
    Scene,
    Square,
    Surface,
)
from algan.rendering.shaders.fragment_shaders import (
    _scatter_forced_mirror,
    forced_mirror_scatter,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.scene_manager import SceneManager
from algan.taichi_compat import ti
from algan.utils.color_space import srgb_to_linear

SHADED = (0.2, 0.4, 0.6, 0.1)


@ti.kernel
def _scatter_probe(
    refraction: ti.template(),
    rd_in: ti.types.ndarray(),
    n_in: ti.types.ndarray(),
    fn_in: ti.types.ndarray(),
    hit_in: ti.types.ndarray(),
    alpha_in: ti.types.ndarray(),
    bounces_in: ti.types.ndarray(),
    params: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(rd_in.shape[0]):
        rd = ti.math.vec3(rd_in[i, 0], rd_in[i, 1], rd_in[i, 2]).normalized()
        n = ti.math.vec3(n_in[i, 0], n_in[i, 1], n_in[i, 2])
        fn = ti.math.vec3(fn_in[i, 0], fn_in[i, 1], fn_in[i, 2])
        hit = ti.math.vec3(hit_in[i, 0], hit_in[i, 1], hit_in[i, 2])
        shaded = ti.math.vec4(SHADED[0], SHADED[1], SHADED[2], SHADED[3])
        (contrib, pass_w, refl_o, refl_d, refl_w, trans_o, trans_d, trans_w) = (
            _scatter_forced_mirror(
                rd,
                n,
                fn,
                hit,
                shaded,
                ti.math.vec3(0.5, 0.5, 0.5),
                alpha_in[i],
                -1.0,
                1.0,
                0.0,
                params,
                0,
                i,
                bounces_in[i],
                refraction,
            )
        )
        for c in ti.static(range(4)):
            out[i, c] = contrib[c]
        for c in ti.static(range(3)):
            out[i, 4 + c] = pass_w[c]
            out[i, 7 + c] = refl_o[c]
            out[i, 10 + c] = refl_d[c]
            out[i, 13 + c] = refl_w[c]
            out[i, 16 + c] = trans_o[c]
            out[i, 19 + c] = trans_d[c]
            out[i, 22 + c] = trans_w[c]


def scatter(
    rd, n_interp, face_n, hit=(0.3, 0.2, 0.0), alpha=1.0, bounces=3, refraction=1
):
    """Run the scatter once and return its eight branches by contract name."""
    init_taichi()

    def row(v):
        return torch.tensor([v], dtype=torch.float32)

    out = torch.zeros(1, 25, dtype=torch.float32)
    _scatter_probe(
        refraction,
        row(rd),
        row(n_interp),
        row(face_n),
        row(hit),
        torch.tensor([alpha], dtype=torch.float32),
        torch.tensor([bounces], dtype=torch.int32),
        torch.zeros(1, 1, 12, dtype=torch.float32),
        out,
    )
    o = out[0]
    return {
        "contrib": o[0:4],
        "pass_w": o[4:7],
        "refl_orig": o[7:10],
        "refl_dir": o[10:13],
        "refl_w": o[13:16],
        "trans_orig": o[16:19],
        "trans_dir": o[19:22],
        "trans_w": o[22:25],
    }


def continuations(s, refraction=1):
    """What the deterministic renderer follows from a scatter's branches.

    The contract in ``shading_taichi``: a positive ``refl_w`` bounces the ray
    and ``pass_w`` is used only when ``refl_w`` is zero; a positive ``trans_w``
    additionally splits off a second ray, where the split pool is compiled in
    (``refraction != 0``). Returns ``[(kind, weight, origin, direction)]``.
    """
    followed = []
    if float(s["refl_w"].max()) > 0.0:
        followed.append(("bounce", s["refl_w"], s["refl_orig"], s["refl_dir"]))
    else:
        followed.append(("through", s["pass_w"], None, None))
    if refraction and float(s["trans_w"].max()) > 0.0:
        followed.append(("split", s["trans_w"], s["trans_orig"], s["trans_dir"]))
    return followed


def _unit(v):
    return F.normalize(torch.tensor(v, dtype=torch.float32), dim=0)


def _mirror(rd, n):
    rd, n = _unit(rd), _unit(n)
    return rd - 2.0 * torch.dot(rd, n) * n


# ---------------------------------------------------------------------------
# An opaque mirror
# ---------------------------------------------------------------------------


def test_an_opaque_hit_commits_fifteen_percent_and_bounces_the_rest():
    """Face-on: 15% of the shaded colour (glow included) stays, 85% bounces
    straight back, achromatically, and nothing passes through or splits.
    """
    s = scatter(rd=(0.0, 0.0, -1.0), n_interp=(0.0, 0.0, 1.0), face_n=(0.0, 0.0, 1.0))
    torch.testing.assert_close(s["contrib"], 0.15 * torch.tensor(SHADED))
    torch.testing.assert_close(s["refl_w"], torch.full((3,), 0.85))
    torch.testing.assert_close(s["refl_dir"], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(s["pass_w"], torch.zeros(3))
    torch.testing.assert_close(s["trans_w"], torch.zeros(3))
    assert [kind for kind, *_ in continuations(s)] == ["bounce"]


@pytest.mark.parametrize(
    ("rd", "n_interp"),
    [
        ((1.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        ((0.3, -0.2, -1.0), (0.2, 0.1, 1.0)),
        ((-0.6, 0.4, -0.7), (-0.1, 0.25, 1.0)),
    ],
)
def test_the_bounce_is_the_mirror_about_the_shading_normal(rd, n_interp):
    """Away from a silhouette the reflection follows the smooth normal, not
    the facet -- what makes a smooth-shaded mesh mirror like the surface it
    approximates -- and leaves on the side the ray came from.
    """
    face_n = (0.0, 0.0, 1.0)
    s = scatter(rd=rd, n_interp=n_interp, face_n=face_n)
    torch.testing.assert_close(s["refl_dir"], _mirror(rd, n_interp), rtol=0, atol=1e-5)
    assert float(torch.dot(s["refl_dir"], _unit(face_n))) > 0.0


def test_out_of_bounces_the_mirror_absorbs():
    """``bounces_left == 0`` leaves the committed 15% and continues nothing."""
    s = scatter(
        rd=(0.0, 0.0, -1.0), n_interp=(0.0, 0.0, 1.0), face_n=(0.0, 0.0, 1.0), bounces=0
    )
    torch.testing.assert_close(s["contrib"], 0.15 * torch.tensor(SHADED))
    for name in ("refl_w", "pass_w", "trans_w"):
        torch.testing.assert_close(s[name], torch.zeros(3), msg=name)


# ---------------------------------------------------------------------------
# Where the bounce starts and which way it goes
# ---------------------------------------------------------------------------


def test_a_silhouette_hit_reflects_away_from_the_surface():
    """REGRESSION. At a smooth-shaded silhouette the bounce went into the solid.

    Near a silhouette the interpolated normal tips past the facet the ray
    actually hit, so ``n_interp . rd`` turns positive on a front-facing hit.
    The scatter read the side off that dot product and flipped the normal, so
    the bounce was mirrored about the flipped frame -- a direction below the
    facet's own horizon -- and its origin was pushed ``1e-3`` along the
    flipped normal, i.e. inside the solid. The ray then struck the object's
    own far side. It now takes the side and, where the mirror direction would
    not clear the facet, the direction from the geometric normal
    (``shading_taichi._reflect_frame``), as the built-in scatter does.
    """
    face_n = (0.0, 0.0, 1.0)
    rd = (0.95, 0.0, -0.3)  # grazing, but onto the front of the facet
    n_interp = (0.6, 0.0, 0.8)  # tipped past the silhouette
    assert float(torch.dot(_unit(rd), _unit(n_interp))) > 0.0
    assert float(torch.dot(_unit(rd), _unit(face_n))) < 0.0

    s = scatter(rd=rd, n_interp=n_interp, face_n=face_n, hit=(0.5, 0.1, 0.25))
    hit = torch.tensor([0.5, 0.1, 0.25])
    assert float(s["refl_dir"][2]) > 0.0, (
        f"the bounce {s['refl_dir'].tolist()} heads into the surface it hit"
    )
    assert float(s["refl_orig"][2] - hit[2]) > 0.0, "the bounce starts inside the solid"
    torch.testing.assert_close(s["refl_dir"].norm(), torch.tensor(1.0))


def test_the_bounce_origin_clears_the_surface_far_from_the_origin():
    """REGRESSION. The bounce origin was offset a fixed ``1e-3``, which at large
    coordinates is below float32's spacing and so moved it nowhere.

    At ``x = 1e5`` neighbouring float32 values are ``2**-7`` apart, so
    ``hit + n * 1e-3`` rounded back to the hit point itself: the reflected ray
    started on the surface and could hit it again (self-intersection acne on a
    scene authored at large coordinates). The offset is now taken in integer
    float space (``transport_taichi._offset_ray_origin``), so it scales with
    the representable spacing at the hit.
    """
    for axis in range(3):
        n = [0.0, 0.0, 0.0]
        n[axis] = 1.0
        hit = [0.25, -0.5, 0.75]
        hit[axis] = 1.0e5
        rd = [0.0, 0.0, 0.0]
        rd[axis] = -1.0
        s = scatter(rd=tuple(rd), n_interp=tuple(n), face_n=tuple(n), hit=tuple(hit))
        lift = float(s["refl_orig"][axis]) - hit[axis]
        assert lift > 0.0, f"axis {axis}: the origin did not leave the surface"


def test_the_bounce_origin_offset_is_small_at_small_coordinates():
    """REGRESSION. The same fixed ``1e-3`` offset was too large at small scales.

    On a scene authored at small coordinates ``1e-3`` is a visible distance:
    a reflection launched that far off the surface skipped anything closer to
    it than that -- the light leaking through thin geometry that
    ``transport_taichi._offset_ray_origin`` was written to stop. The
    scale-aware offset is a few hundred ULPs of the hit point, which at
    ``0.5`` is a few times ``1e-5``.
    """
    hit = (0.5, 0.5, 0.5)
    s = scatter(
        rd=(0.0, 0.0, -1.0), n_interp=(0.0, 0.0, 1.0), face_n=(0.0, 0.0, 1.0), hit=hit
    )
    lift = float(s["refl_orig"][2]) - hit[2]
    assert 0.0 < lift < 1e-4
    torch.testing.assert_close(s["refl_orig"][:2], torch.tensor(hit[:2]))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.9])
def test_a_partly_covering_mirror_keeps_what_shows_through_it(alpha):
    """REGRESSION. A partly covering forced mirror dropped everything behind it.

    The scatter returned ``pass_w = 1 - alpha`` alongside a positive
    ``refl_w``, but the contract uses ``pass_w`` only when ``refl_w`` is zero,
    so the deterministic renderer bounced and threw the pass-through away: a
    half-transparent (or fading-in) mirror rendered as an opaque dark panel,
    while the path tracer, which samples all three branches, showed the scene
    behind it. The reflection now takes the split slot when there is
    something behind it, as the built-in scatter does for a semi-transparent
    reflector, so every share of the ray except the committed 15% continues.
    """
    s = scatter(
        rd=(0.0, 0.0, -1.0),
        n_interp=(0.0, 0.0, 1.0),
        face_n=(0.0, 0.0, 1.0),
        alpha=alpha,
        refraction=1,
    )
    followed = continuations(s, refraction=1)
    total = sum(float(weight.max()) for _kind, weight, *_ in followed)
    assert total == pytest.approx(1.0 - 0.15 * alpha, abs=1e-6), followed
    by_kind = {kind: weight for kind, weight, *_ in followed}
    assert "through" in by_kind, "the surface is partly absent: rays must pass"
    torch.testing.assert_close(by_kind["through"], torch.full((3,), 1.0 - alpha))
    (_kind, weight, origin, direction) = [c for c in followed if c[0] != "through"][0]
    torch.testing.assert_close(weight, torch.full((3,), 0.85 * alpha))
    torch.testing.assert_close(direction, torch.tensor([0.0, 0.0, 1.0]))
    assert float(origin[2]) > 0.0


@pytest.mark.parametrize(("alpha", "kept"), [(0.5, "through"), (0.8, "bounce")])
def test_without_the_split_pool_the_heavier_continuation_is_kept(alpha, kept):
    """REGRESSION. With only one ray to spend, the scatter spent it on the
    reflection whatever the shares, so at ``alpha = 0.5`` it kept the 42.5%
    bounce and dropped the 50% behind. It now keeps the heavier of the two, so
    the share it drops is always the smaller -- the built-in rule.
    """
    s = scatter(
        rd=(0.0, 0.0, -1.0),
        n_interp=(0.0, 0.0, 1.0),
        face_n=(0.0, 0.0, 1.0),
        alpha=alpha,
        refraction=0,
    )
    followed = continuations(s, refraction=0)
    assert [kind for kind, *_ in followed] == [kept]


# ---------------------------------------------------------------------------
# What a user sees
# ---------------------------------------------------------------------------


def _patch_mean(img, rows, cols):
    """Mean linear-light RGB of an image region (OpenCV loads BGR bytes)."""
    rgb = torch.from_numpy(img[rows, cols, ::-1].copy()).float() / 255.0
    return srgb_to_linear(rgb).reshape(-1, 3).mean(0)


def test_a_half_covering_forced_mirror_shows_what_is_behind_it(tmp_path):
    """REGRESSION. Rendered, a half-transparent forced mirror hid what was behind it.

    A black, unlit, ``opacity=0.5`` panel wearing ``forced_mirror_scatter``
    stands face-on in front of a red wall, with a large green panel behind the
    camera for it to reflect. Half of each camera ray misses the panel, so the
    wall must come through it at half strength in linear light, and the 85%
    mirror share of the half that hits must show the green: ``(0.5, 0.425, 0)``.
    The deterministic renderer followed only the reflection, so the panel came
    out green on black, with no trace of the wall behind it.
    """
    SceneManager.reset()
    try:
        with Scene(video_settings=SMOKE_TEST) as scene:
            scene.set_background(BLACK)
            Scene.clear_lights()
            with Off():
                Square(size=6.0, color=PURE_RED).move(IN * 2.0).spawn(animate=False)
                # Behind the camera, which sits at z = 20: seen only in the mirror.
                behind_camera = Square(size=30.0, color=PURE_GREEN).move(OUT * 30.0)
                behind_camera.spawn(animate=False)
                panel = Surface(color=BLACK, opacity=0.5).scale(1.5)
                panel.set_fragment_shader(forced_mirror_scatter)
                panel.spawn(animate=False)
            result = scene.save_frame(
                tmp_path / "half_mirror.png", overwrite=True, post_processes=()
            )
    finally:
        SceneManager.reset()
    img = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    # The panel spans 3 of the frame's ~7.7 units (12 px); the wall 6 (24 px).
    through_panel = _patch_mean(img, slice(cy - 3, cy + 3), slice(cx - 3, cx + 3))
    beside_panel = _patch_mean(img, slice(cy - 2, cy + 2), slice(cx + 7, cx + 10))
    assert float(beside_panel[0]) > 0.9, f"the bare wall is not red: {beside_panel}"
    assert float(beside_panel[1]) < 0.02
    ratio = float(through_panel[0] / beside_panel[0])
    assert ratio == pytest.approx(0.5, abs=0.04), (
        f"the wall comes through the half-covering panel at {ratio:.3f} of its "
        "strength; half of every ray misses the panel"
    )
    assert float(through_panel[1]) == pytest.approx(0.85 * 0.5, abs=0.04), (
        f"the mirror share shows green at {float(through_panel[1]):.3f}"
    )
