"""Render-level parity of the in-kernel Manim stage against Manim itself.

The torch-side ``manim_shader`` is pinned against the vendored
``get_shaded_rgb`` by ``test_manim_shader.py``, but almost every rendered
frame goes through the in-kernel port ``_stage_manim`` (material id 0)
instead. This file renders a real frame through that stage and compares
individual pixels against the vendored Manim function, evaluated at the exact
world position each sampled pixel's centre ray hits.

Geometry: axis-aligned ``Cube`` solids (``Polyhedron`` subclasses, so their
face normals are proven outward and their shading is declared one-sided --
the stage receives the geometric normal, not a viewer-flipped one). Their
face vertex normals are zero, so the kernel's degenerate-normal fallback
hands the stage the geometric face normal, which for an axis-aligned face is
known exactly. Every sampled face is perpendicular to the view axis, so one
pixel spans the same small world step it does on a calibration face -- on a
steeply inclined face one pixel spans many times more world distance, and
sub-byte evaluation detail would swamp the comparison.

- Test 1 places the light BETWEEN two screen-parallel front planes: the
  centred cube's front face (z = 2) is turned away from it and exercises
  the halved negative lobe of ``0.5 * (n . to_light) ** 3``, while a second
  cube pushed behind the light (front plane z = -4) catches the positive
  lobe. The camera looks straight down the axis, so the pixel at the centre
  of the image samples the exact centre of the centred cube's front face.
- Test 2 renders solids with no material of their own after
  ``Scene.use_manim_defaults()`` and checks them against Manim's own light
  position: the centred cube's front face is lit, and a second cube hangs
  below the eye line so its top face is visible while facing away from
  Manim's low light.

The expected byte is ``floor(255 * clamp(get_shaded_rgb(...), 0, 1) + 0.5)``
-- the same round-half-up the encoder applies -- computed by importing the
vendored function, never by re-implementing it. Tolerance is 1 byte per
channel: it covers rounding and the sRGB decode/encode round trip, plus the
sub-byte effect of a partially covering fragment being evaluated at its
owned-sample centroid instead of the pixel centre (a pixel straddling a
face's triangulation diagonal).

Not marked ``fast``: this pays a real render.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image

from algan import (
    BLACK,
    ORIGIN,
    OUT,
    SETTINGS,
    WHITE,
    Cube,
    Off,
    PointLight,
    Scene,
)
from algan.constants.color import Color
from algan.external_libraries.manim.utils.color.core import get_shaded_rgb
from algan.manim_defaults import MANIM_LIGHT_SOURCE
from algan.rendering.shaders.material_shaders import manim_shader
from algan.rendering.shaders.materials import ManimMaterial
from algan.scene_manager import SceneManager
from algan.settings.video_settings import VideoSettings

#: Even on both axes -- deliberately. The tracer's half-screen extents are
#: integer floors (``float(width // 2)`` at the render call sites), so on an
#: odd resolution every kernel-side sample sits up to half a pixel away from
#: the analytic projection this harness replicates, which is worth whole
#: bytes wherever the shading gradient is steep. On an even resolution the
#: two conventions coincide exactly.
_RESOLUTION = (160, 160)

#: Side length of both cubes. Front faces sit 2 units in front of a cube
#: centre.
_CUBE_SIDE = 4.0

#: Non-grey albedo with three distinct channels, far enough from both ends of
#: the range that no expected value under either rig reaches the [0, 1]
#: clamp.
_ALBEDO_RGB = (0.72, 0.38, 0.13)

#: Test 1's own stated light: off-axis, and sandwiched between the centred
#: cube's front plane (z = 2, which it is behind, so that face turns away
#: and exercises the halved negative lobe) and the forward cube's front plane
#: (z = -4, which it is in front of, so that face catches the positive lobe).
#: OUTWARD is +z, so "in front" is the larger z.
_TEST_LIGHT = (4.0, 3.0, -2.0)

#: Camera distance and the vertical fov that frames 8 world units at the
#: origin plane. The test states these itself rather than borrowing Algan's
#: stock camera placement (the numbers happen to match Manim's rig).
_CAMERA_DISTANCE = 20.0
_FIELD_OF_VIEW_DEGREES = math.degrees(2.0 * math.atan(4.0 / _CAMERA_DISTANCE))

_CUBE_A_CENTRE = torch.tensor((0.0, 0.0, 0.0))
# Test 1's second cube sits behind the light, far enough right that its lit
# front face clears the centred cube's silhouette.
_CUBE_B_CENTRE_FAR = torch.tensor((4.0, 0.0, -6.0))
# Test 2 hangs it low: Manim's own light sits below the origin (its y is -9),
# so the top face is visible from underneath while facing away from it.
_CUBE_B_CENTRE_LOW = torch.tensor((0.0, -4.5, 0.0))

_TOLERANCE_BYTES = 1
_NEIGHBOURHOOD_RADIUS = 2


def _video_settings():
    return VideoSettings(
        resolution=_RESOLUTION,
        frames_per_second=1,
        supersampling=1,
    )


@pytest.fixture
def restored_global_settings():
    """Restore what ``use_manim_defaults()`` repoints process-globally.

    The conftest autouse fixture already snapshots ``SETTINGS`` around every
    test; this mirrors the explicit save/restore ``test_manim_defaults.py``
    uses, so the mutation the feature under test performs is visible and
    undone at the point of use.
    """
    style = SETTINGS.style
    saved_style = (style.default_material, style.background.clone())
    saved_tonemapping = SETTINGS.raytracing.tonemapping
    # use_manim_defaults also selects Manim's display-referred working space,
    # which changes what every LATER test in the session renders -- this test
    # file's own expectations included. Restored to the value that was there,
    # not to the documented default, so a leak from an earlier test is exposed
    # rather than silently repaired.
    saved_linear = SETTINGS.raytracing.linear_color_space
    yield
    SETTINGS.style.set(default_material=saved_style[0], background=saved_style[1])
    SETTINGS.raytracing.set(
        tonemapping=saved_tonemapping, linear_color_space=saved_linear
    )


def _face_descriptors(manim_defaults_rig):
    """The two sampled faces per rig, each fully stated: centre point on its
    plane, outward unit normal, the two in-face axes, and the world segment
    the face's triangulation diagonal runs along.

    ``Polyhedron`` fans each quad face into triangles ``(f0, f1, f2)`` and
    ``(f0, f2, f3)``, so the diagonal joins the face's first and third
    corners. A pixel straddling that diagonal is composited from TWO
    partially covering fragments, each evaluated at its own owned-sample
    centroid, so the sampling loop asserts every checked pixel stays
    strictly inside one triangle.
    """
    half = _CUBE_SIDE / 2
    # OUTWARD is +z, so a cube's camera-facing plane is at bz + half.
    front_face_of = lambda bx, by, bz: {  # noqa: E731
        "centre": (bx, by, bz + half),
        "normal": (0.0, 0.0, 1.0),
        "axis_u": (1.0, 0.0, 0.0),
        "axis_v": (0.0, 1.0, 0.0),
        "diagonal_from": (bx - half, by - half, bz + half),
        "diagonal_to": (bx + half, by + half, bz + half),
    }
    if manim_defaults_rig:
        bx, by, bz = _CUBE_B_CENTRE_LOW.tolist()
        second_face = (
            "top face of the low cube (turned away from Manim's light)",
            {
                "centre": (bx - 1.0, by + half, bz - 0.5),
                "normal": (0.0, 1.0, 0.0),
                "axis_u": (1.0, 0.0, 0.0),
                "axis_v": (0.0, 0.0, -1.0),
                "diagonal_from": (bx - half, by + half, bz + half),
                "diagonal_to": (bx + half, by + half, bz - half),
                "single_triangle": True,
            },
        )
        first_label = "front face of the centred cube (lit by Manim's light)"
    else:
        # Off the quad centre and well clear of the f0-f2 diagonal (which
        # runs through it), so every checked pixel sits in one triangle.
        fx, fy, fz = _CUBE_B_CENTRE_FAR.tolist()
        second_face = (
            "front face of the forward cube (in front of the test light)",
            dict(
                front_face_of(fx, fy, fz),
                centre=(fx - 1.0, fy + 1.0, fz + half),
                single_triangle=True,
            ),
        )
        first_label = "front face of the centred cube (turned away from the light)"
    return [
        (first_label, front_face_of(0.0, 0.0, 0.0)),
        second_face,
    ]


def _make_side_of(descriptor, normal):
    """Signed distance from a face's triangulation diagonal: + on one
    triangle's side, - on the other's, 0 on the diagonal itself.
    """
    if "diagonal_from" not in descriptor:
        return None
    p0 = torch.tensor(descriptor["diagonal_from"], dtype=torch.float64)
    p1 = torch.tensor(descriptor["diagonal_to"], dtype=torch.float64)
    edge_dir = (p1 - p0) / (p1 - p0).norm()
    perp = torch.linalg.cross(normal, edge_dir)

    def side_of(point):
        return ((point - p0) * perp).sum().item()

    return side_of


def _render_scene(tmp_path, name, *, manim_defaults_rig):
    """Build the two-cube scene, render one still, and measure it.

    Returns one record per checked face: the sampled pixel coordinates, the
    exact world sample position, the expected bytes from the vendored
    function over a small neighbourhood, and the rendered bytes for that same
    neighbourhood.
    """
    width, height = _RESOLUTION
    output_path = tmp_path / f"{name}.png"
    SceneManager.reset()
    try:
        with Scene(video_settings=_video_settings()) as scene:
            scene.set_background(BLACK)
            # Stated inputs rather than ambient defaults: single-sample
            # deterministic tracer, Manim-faithful output transform (no
            # curve, unit exposure), no shadow fans (Manim's model has none;
            # the stage's visibility gate reads 1 either way).
            SETTINGS.raytracing.set(
                samples_per_pixel=1,
                tonemapping=False,
                tonemap_exposure=1.0,
                shadows=False,
            )
            with Off():
                if manim_defaults_rig:
                    space = SETTINGS.raytracing.linear_color_space
                    scene.use_manim_defaults()
                    # Put the working colour space back to whatever this
                    # SESSION has been rendering in. use_manim_defaults selects
                    # Manim's display-referred space, which is a PROCESS-START
                    # decision: it is folded into the kernels through ti.static,
                    # so flipping it after something has already rendered
                    # leaves the compiled kernels in the old space while the
                    # host-side half of the pipeline moves to the new one, and
                    # the two disagree by ~24/255. This file renders several
                    # scenes in one process, so it has to pick one space and
                    # keep it; what it is testing is the shading LAW, which
                    # matches get_shaded_rgb to a byte in either space (both
                    # tests pass under ALGAN_LINEAR_COLOR=0 as well).
                    SETTINGS.raytracing.set(linear_color_space=space)
                else:
                    # Not Algan's stock rig: this test states its own light.
                    scene.clear_lights()
                    PointLight(location=torch.tensor(_TEST_LIGHT), color=WHITE).spawn(
                        animate=False
                    )
                    camera = scene.get_camera()
                    camera.move_to(OUT * _CAMERA_DISTANCE)
                    camera.look_at(ORIGIN)
                    camera.set_fov(_FIELD_OF_VIEW_DEGREES)

                cubes = []
                for centre in (
                    _CUBE_A_CENTRE,
                    _CUBE_B_CENTRE_LOW if manim_defaults_rig else _CUBE_B_CENTRE_FAR,
                ):
                    # Constructed at the origin and moved with move_to, the
                    # way every scene in this repository places a Polyhedron:
                    # the recorded move propagates to the face geometry,
                    # which the location= constructor kwarg does not.
                    cube = Cube(
                        size=_CUBE_SIDE,
                        fill_color=Color(_ALBEDO_RGB),
                        fill_opacity=1.0,
                    )
                    if not manim_defaults_rig:
                        cube.set_material(ManimMaterial())
                    cube.move_to(centre)
                    cube.spawn(animate=False)
                    cubes.append(cube)

            lights = scene.get_light_sources()
            assert len(lights) == 1
            # What the face mobs themselves carry. A bare solid keeps
            # ``shader is None`` -- the Manim shading the render below
            # verifies can then only have come from the default material
            # reaching the geometry at primitive-build time, which is the
            # end-to-end claim under test.
            shaders = {
                getattr(mob, "shader", None) for mob in cubes[0].faces.get_descendants()
            }
            if manim_defaults_rig:
                assert isinstance(SETTINGS.style.default_material, ManimMaterial)
                expected_light = torch.tensor(MANIM_LIGHT_SOURCE).reshape(3).double()
                assert shaders == {None}
            else:
                assert shaders == {manim_shader}
                expected_light = torch.tensor(_TEST_LIGHT, dtype=torch.float64)
            assert lights[0].location.reshape(3).tolist() == pytest.approx(
                expected_light.tolist(), abs=1e-5
            )

            # The camera model the renderer consumes, read off the Scene the
            # way _materialize_render_state reads it.
            camera = scene.get_camera()
            ro = camera.location.detach().reshape(3).double()
            sp = camera.screen.location.detach().reshape(3).double()
            sb = camera._get_render_screen_basis().detach().reshape(3, 3).double()
            dual = torch.linalg.inv(sb)
            d0, d1 = dual[:, 0], dual[:, 1]
            n2 = sb[2]
            half_h = height / 2

            def ray_through(x_cont, y_cont):
                # World ray through a continuous pixel position; x grows
                # rightward, y grows UPWARD from the bottom edge (the
                # composite buffer is bottom-up before the output flip).
                # Inverts raytrace_kernels_taichi._generate_ray exactly.
                u = (x_cont - width / 2) / half_h
                v = (y_cont - half_h) / half_h
                target = sp + u * d0 + v * d1
                direction = target - ro
                return direction / direction.norm()

            def hit_on_plane(direction, plane_point, plane_normal):
                offset = ((plane_point - ro) * plane_normal).sum()
                t = offset / (direction * plane_normal).sum()
                return ro + t * direction

            def continuous_pixel(world_point):
                # Forward projection matching raster_pipeline._project_points.
                denom = ((world_point - ro) * n2).sum()
                scale = ((sp - ro) * n2).sum() / denom
                hit = ro + scale * (world_point - ro)
                rel = hit - sp
                u = (rel * sb[0]).sum()
                v = (rel * sb[1]).sum()
                return (u * half_h + width / 2).item(), (v * half_h + half_h).item()

            def pixel_of(world_point):
                px_cont, py_cont = continuous_pixel(world_point)
                col = int(math.floor(px_cont))
                row = height - 1 - int(math.floor(py_cont))
                return col, row

            def sample_at(col, row, plane_point, plane_normal):
                px_cont = col + 0.5
                py_cont = height - row - 0.5
                ray = ray_through(px_cont, py_cont)
                return hit_on_plane(ray, plane_point, plane_normal)

            def corner_samples(col, py_index, plane_point, plane_normal):
                points = []
                for dx in (0.0, 1.0):
                    for dy in (0.0, 1.0):
                        ray = ray_through(col + dx, py_index + dy)
                        points.append(hit_on_plane(ray, plane_point, plane_normal))
                return points

            def expected_bytes(sample, plane_normal):
                shaded = get_shaded_rgb(
                    np.array(_ALBEDO_RGB, dtype=np.float64),
                    sample.numpy(),
                    plane_normal.numpy(),
                    expected_light.numpy(),
                )
                return np.floor(np.clip(shaded, 0.0, 1.0) * 255.0 + 0.5)

            records = []
            window = range(-_NEIGHBOURHOOD_RADIUS, _NEIGHBOURHOOD_RADIUS + 1)
            for label, descriptor in _face_descriptors(manim_defaults_rig):
                centre = torch.tensor(descriptor["centre"], dtype=torch.float64)
                normal = torch.tensor(descriptor["normal"], dtype=torch.float64)
                axis_u = torch.tensor(descriptor["axis_u"], dtype=torch.float64)
                axis_v = torch.tensor(descriptor["axis_v"], dtype=torch.float64)
                col, row = pixel_of(centre)

                side_of = _make_side_of(descriptor, normal)

                # Every pixel asserted below must lie wholly inside this
                # face: all four corner rays of every neighbourhood pixel
                # have to land strictly within its bounds. When
                # ``single_triangle`` holds, they must additionally stay on
                # one side of the triangulation diagonal, so each compared
                # pixel is ONE fully covering fragment evaluated exactly at
                # the pixel centre -- no centroid-offset blend, no other
                # surface, no background. The exception is a quad face's own
                # centre sample: the f0-f2 diagonal passes through the
                # centre of EVERY quad face by construction, so that pixel
                # is always two fragments deep; on a face perpendicular to
                # the view axis the two fragments' shades differ by well
                # under the tolerance (measured; see the module docstring),
                # and being mid-face is the whole point of that assertion.
                single_triangle = bool(descriptor.get("single_triangle", False))
                margin_world = float("inf")
                for dr in window:
                    for dc in window:
                        neighbour_row = row + dr
                        neighbour_col = col + dc
                        py_index = height - 1 - neighbour_row
                        corners = corner_samples(
                            neighbour_col, py_index, centre, normal
                        )
                        for point in corners:
                            local = point - centre
                            along_u = abs((local * axis_u).sum().item())
                            along_v = abs((local * axis_v).sum().item())
                            reach = max(along_u, along_v)
                            margin_world = min(margin_world, _CUBE_SIDE / 2 - reach)
                            assert reach < _CUBE_SIDE / 2, (
                                f"{label}: pixel ({neighbour_col}, "
                                f"{neighbour_row}) is not fully covered by "
                                "this face"
                            )
                            if single_triangle and side_of is not None:
                                centre_side = side_of(
                                    sample_at(
                                        neighbour_col, neighbour_row, centre, normal
                                    )
                                )
                                corner_side = side_of(point)
                                assert corner_side != 0.0, (
                                    f"{label}: pixel ({neighbour_col}, "
                                    f"{neighbour_row}) touches the "
                                    "triangulation diagonal"
                                )
                                assert corner_side * centre_side > 0.0, (
                                    f"{label}: pixel ({neighbour_col}, "
                                    f"{neighbour_row}) straddles the "
                                    "triangulation diagonal"
                                )

                centre_sample = sample_at(col, row, centre, normal)
                # The sampled position has to be the face centre to within a
                # pixel: that is what makes "evaluated at the world position
                # of the sampled pixel" stateable from the geometry alone.
                offset = (centre_sample - centre).norm().item()
                assert offset <= margin_world, (
                    f"{label}: centre pixel samples {offset:.4f} world units "
                    f"from the face centre; nearest edge is {margin_world:.4f}"
                )

                expected_window = {}
                for dr in window:
                    for dc in window:
                        sample = sample_at(col + dc, row + dr, centre, normal)
                        expected_window[(dc, dr)] = expected_bytes(sample, normal)

                records.append(
                    {
                        "label": label,
                        "col": col,
                        "row": row,
                        "sample": centre_sample,
                        "offset_from_centre": offset,
                        "expected_window": expected_window,
                        "rendered_window": {},
                    }
                )

            result = scene.save_frame(
                output_path,
                video_settings=_video_settings(),
                at=0.0,
                overwrite=True,
                post_processes=(),
            )
            assert result.status == "rendered"
    finally:
        SceneManager.reset()

    with Image.open(output_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.int64)

    for record in records:
        for key in record["expected_window"]:
            rendered = pixels[record["row"] + key[1], record["col"] + key[0]]
            record["rendered_window"][key] = rendered
    return records


def _print_and_check(record):
    """Print one record's measured numbers and assert them against the
    vendored function's bytes, over the centre pixel and its whole
    neighbourhood.
    """
    rendered = record["rendered_window"][(0, 0)]
    expected = record["expected_window"][(0, 0)]
    difference = np.abs(rendered - expected)
    print(f"[{record['label']}]")
    print(f"  pixel             : col={record['col']} row={record['row']}")
    print(
        "  sampled world pos :",
        np.array2string(record["sample"].numpy(), precision=6),
    )
    print(f"  rendered RGB      : {rendered.tolist()}")
    print(f"  expected RGB      : {expected.tolist()}   (get_shaded_rgb)")
    print(f"  |difference|      : {difference.tolist()}")
    message = (
        f"{record['label']}: rendered {rendered.tolist()} deviates from "
        f"get_shaded_rgb {expected.tolist()} by {difference.tolist()} "
        f"(tolerance {_TOLERANCE_BYTES} per channel)"
    )
    assert difference.max() <= _TOLERANCE_BYTES, message

    window_worst = 0
    for key, expected_val in record["expected_window"].items():
        rendered_val = record["rendered_window"][key]
        window_worst = max(window_worst, int(np.abs(rendered_val - expected_val).max()))
    print(f"  worst over 5x5    : {window_worst}")
    assert window_worst <= _TOLERANCE_BYTES
    return rendered


def _albedo_bytes():
    return np.floor(np.array(_ALBEDO_RGB) * 255.0 + 0.5)


def test_in_kernel_stage_matches_get_shaded_rgb(tmp_path, restored_global_settings):
    """Explicit rig: one white intensity-1 PointLight placed by this test.

    Both assertions come out of one render. The light sits between the two
    front planes, so the centred cube's front face is turned away from it --
    the halved negative lobe darkens that face below the authored colour by
    exactly what ``get_shaded_rgb`` predicts -- while the forward cube's
    front face catches the positive lobe.
    """
    dark_record, lit_record = _render_scene(
        tmp_path, "manim_stage_explicit_rig", manim_defaults_rig=False
    )
    dark = _print_and_check(dark_record)
    lit = _print_and_check(lit_record)

    albedo = _albedo_bytes()
    # Direction checks that would catch a flipped row order or a sign error
    # in the lobe: the facing-the-light face brightens above the authored
    # colour, the turned-away face darkens below it.
    assert (lit > albedo).all()
    assert (dark < albedo).all()


def test_use_manim_defaults_reaches_bare_solids(tmp_path, restored_global_settings):
    """End to end: solids with NO material of their own, rendered after
    ``Scene.use_manim_defaults()``, shade the way Manim shades -- against the
    rig's own light position, mirrored into Algan coordinates like everything
    else the method installs.
    """
    lit_record, dark_record = _render_scene(
        tmp_path, "manim_stage_default_material", manim_defaults_rig=True
    )
    lit = _print_and_check(lit_record)
    dark = _print_and_check(dark_record)

    albedo = _albedo_bytes()
    assert (lit > albedo).all()
    assert (dark < albedo).all()
