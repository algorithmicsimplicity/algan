"""What a 3-D Mob with no material of its own renders as, hop by hop.

The process default is ``SETTINGS.style.default_material``, installed at
import as a ``DiffuseMaterial``; a triangle primitive built for a mob that set
no material takes the default material's shader *and* its parameter values.
This file pins each link: the imported default itself, the packed material id
of a bare solid, the two families deliberately kept unlit so 2-D content stays
flat (``ImageMob`` and triangulated circuit fills), circuits' own immunity,
the settings validation, and a configured default material's parameters
actually reaching the packed block -- plus an explicit mob value beating it.

Tensor assertions on the primitive build, no render and no Taichi. Feature
tests of this plumbing rather than of anything the timeline or Scene can
break, so none of it is marked ``fast``.
"""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS, Circle, ImageMob, Off, Scene, Sphere
from algan.errors import AlganConfigurationError
from algan.rendering.primitives.bezier_circuit_primitive import BezierCircuitPrimitive
from algan.rendering.raytracing.settings import _MAT_SLOTS
from algan.rendering.raytracing.shading_taichi import _MID_LAMBERT, _MID_UNLIT
from algan.rendering.shaders.material_shaders import lambert_shader
from algan.rendering.shaders.materials import DiffuseMaterial, MeshStandardMaterial


@pytest.fixture
def style():
    """Save/restore the whole style section around a test."""
    saved = SETTINGS.snapshot()
    try:
        yield SETTINGS.style
    finally:
        SETTINGS.restore(saved)


def _primitives(mob):
    build = getattr(mob, "get_render_primitives", None)
    got = build() if build is not None else None
    if got is None:
        # A container builds nothing itself; walk into what does.
        return [p for child in mob.children for p in _primitives(child)]
    return got if isinstance(got, list) else [got]


def _packed_material_ids(mob):
    ids = []
    for primitive in _primitives(mob):
        pack = getattr(primitive, "_pack_material", None)
        if pack is None:
            continue  # a circuit primitive carries no material block at all
        mat_id, _mat = pack()
        ids.extend(mat_id.reshape(-1).tolist())
    return ids


def test_the_imported_default_is_a_diffuse_material():
    assert isinstance(SETTINGS.style.default_material, DiffuseMaterial)
    assert SETTINGS.style.default_material.shader is lambert_shader


def test_a_bare_3d_mob_packs_the_lambert_id():
    with Scene(), Off():
        sphere = Sphere(radius=0.5)
        sphere.spawn(animate=False)

        primitives = _primitives(sphere)
        assert primitives, "the bare solid built no render primitive"
        assert all(p.shader is lambert_shader for p in primitives)

        ids = _packed_material_ids(sphere)
        assert ids
        assert set(ids) == {_MID_LAMBERT}


def test_an_image_mob_is_unlit():
    # A picture shows its own colours, not a lighting gradient: it must reach
    # the renderer as unlit whatever material the process default carries.
    with Scene(), Off():
        image = ImageMob(torch.rand(8, 6, 4))
        image.spawn(animate=False)

        ids = _packed_material_ids(image)
        assert ids
        assert set(ids) == {_MID_UNLIT}


def test_a_triangulated_circuit_fill_is_unlit():
    # A triangulated circuit's fill is a bezier circuit that happens to be
    # triangulated (this is what TexTriangulated glyph fills and plots' curves
    # are made of); its untriangulated twin is unlit, and circuits stay unlit.
    from algan.mobs.triangulated_bezier_circuit import TriangulatedBezierCircuit

    # One closed square outline as four cubic segments.
    square = torch.tensor(
        [
            [
                [-0.5, -0.5, 0.0],
                [-0.17, -0.5, 0.0],
                [0.17, -0.5, 0.0],
                [0.5, -0.5, 0.0],
            ],
            [[0.5, -0.5, 0.0], [0.5, -0.17, 0.0], [0.5, 0.17, 0.0], [0.5, 0.5, 0.0]],
            [[0.5, 0.5, 0.0], [0.17, 0.5, 0.0], [-0.17, 0.5, 0.0], [-0.5, 0.5, 0.0]],
            [
                [-0.5, 0.5, 0.0],
                [-0.5, 0.17, 0.0],
                [-0.5, -0.17, 0.0],
                [-0.5, -0.5, 0.0],
            ],
        ]
    )
    with Scene(), Off():
        fill = TriangulatedBezierCircuit([square])
        fill.spawn(animate=False)

        ids = _packed_material_ids(fill)
        assert ids
        assert set(ids) == {_MID_UNLIT}


def test_a_plain_circle_never_reaches_the_default():
    # What determines a circuit's shading is its primitive kind: a Circle
    # renders as analytic-coverage circuit primitives, which carry no material
    # block at all -- not a shader attribute of its own.
    with Scene(), Off():
        circle = Circle(radius=1.0)
        circle.spawn(animate=False)

        primitives = _primitives(circle)
        assert primitives
        assert all(isinstance(p, BezierCircuitPrimitive) for p in primitives)
        assert all(getattr(p, "_pack_material", None) is None for p in primitives)


def test_a_non_material_default_is_rejected(style):
    # A plain function has no .shader attribute, so this must fail loudly
    # rather than detonate later at first primitive build.
    with pytest.raises(AlganConfigurationError):
        SETTINGS.style.set(default_material=lambert_shader)


def test_the_default_materials_parameters_reach_the_packed_block(style):
    SETTINGS.style.set(default_material=MeshStandardMaterial(roughness=0.3))
    start, _width = _MAT_SLOTS["roughness"]

    with Scene(), Off():
        bare = Sphere(radius=0.5)
        bare.spawn(animate=False)
        _mat_id, mat = _primitives(bare)[0]._pack_material()
        assert bool(torch.all(mat[:, :, start] == 0.3)), (
            "a configured default material's parameter value must reach the "
            "packed material block, not silently fall back to the built-in "
            "default"
        )

        # An explicit per-mob value still beats the seed.
        explicit = Sphere(radius=0.5).set_material(MeshStandardMaterial(roughness=1.0))
        explicit.spawn(animate=False)
        _mat_id, mat = _primitives(explicit)[0]._pack_material()
        assert bool(torch.all(mat[:, :, start] == 1.0))
