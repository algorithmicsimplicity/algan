"""``set_shader`` checks a custom vertex shader against how it is called.

The renderer passes the nine fixed values by position, then each material
parameter by position. A shader that cannot take them so failed only once the
first frame was shaded; one that left out a fixed parameter was given the
wrong values without a word.
"""

from __future__ import annotations

import warnings
from functools import partial

import pytest

from algan import SMOKE_TEST, MeshStandardMaterial, Scene, Sphere
from algan.errors import AlganConfigurationError, AlganWarning
from algan.rendering.shaders import material_shaders, pbr_shaders


def _short(memory, vertex_location, vertex_normal, albedo_color, bands=4):
    return albedo_color


def _keyword_bands(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity,
    ambient_light_intensity,
    *,
    bands=4,
):
    return albedo_color


def _keyword_extras(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity,
    ambient_light_intensity,
    **extras,
):
    return albedo_color


def toon(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity,
    bands=4,
):
    return albedo_color


def no_normal(
    memory,
    vertex_location,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity,
    ambient_light_intensity,
    bands=4,
):
    return albedo_color


def my_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity,
    ambient_light_intensity,
    banding=4.0,
):
    return albedo_color


@pytest.mark.parametrize(
    ("shader", "problem"),
    [
        (_short, r"_short\(memory, .+, bands=4\) takes 5 parameters\. "),
        (_keyword_bands, r"_keyword_bands takes bands=4 by keyword only\. "),
        (_keyword_extras, r"_keyword_extras takes \*\*extras by keyword only\. "),
        (
            partial(my_shader, banding=8.0),
            r"partial takes banding=8\.0 by keyword only\. ",
        ),
    ],
    ids=["too few", "keyword-only", "var-keyword", "partial keyword"],
)
def test_a_shader_the_renderer_cannot_call_is_refused_when_it_is_set(shader, problem):
    with (
        Scene(video_settings=SMOKE_TEST) as scene,
        pytest.raises(
            AlganConfigurationError,
            match=problem + r"Algan calls a vertex shader with nine fixed parameters, "
            r"by position: memory, vertex_location, .+, ambient_light_intensity\.",
        ),
    ):
        Sphere(scene=scene).set_shader(shader)


def test_a_shader_missing_a_fixed_parameter_warns_at_the_call():
    with Scene(video_settings=SMOKE_TEST) as scene:
        sphere = Sphere(scene=scene)
        with pytest.warns(
            AlganWarning,
            match=r"^toon will be given the wrong values: bands is given "
            r"ambient_light_intensity\. Algan calls",
        ) as caught:
            sphere.set_shader(toon)
    assert [w.filename for w in caught] == [__file__]
    # It is still installed, as before.
    assert sphere.shader is toon


def test_every_fixed_parameter_a_shift_moves_is_named():
    with (
        Scene(video_settings=SMOKE_TEST) as scene,
        pytest.warns(
            AlganWarning,
            match=r"no_normal will be given the wrong values: albedo_color is "
            r"given vertex_normal; camera_location is given albedo_color; .+; "
            r"ambient_light_intensity is given light_intensity; bands is given "
            r"ambient_light_intensity\.",
        ),
    ):
        Sphere(scene=scene).set_shader(no_normal)


def test_a_fixed_parameter_after_the_nine_is_named():
    def appended(
        memory,
        vertex_location,
        vertex_normal,
        albedo_color,
        camera_location,
        light_origin,
        light_color,
        bands,
        light_intensity,
        ambient_light_intensity,
    ):
        return albedo_color

    with (
        Scene(video_settings=SMOKE_TEST) as scene,
        pytest.warns(
            AlganWarning,
            match=r"light_intensity is given ambient_light_intensity; "
            r"ambient_light_intensity is taken as a material parameter\.",
        ),
    ):
        Sphere(scene=scene).set_shader(appended)


def _renamed(mem, pos, nrm, col, cam, lo, lc, li, amb, bands=4):
    return col


def _renamed_with_defaults(mem, pos, nrm, col, cam, lo, lc, li=1.0, amb=0.5, bands=4):
    return col


def _ambient_default(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity,
    ambient_light_intensity=1.0,
    bands=4,
):
    return albedo_color


def _any_number(*values):
    return values[3]


class _Callable:
    def __call__(
        self,
        memory,
        vertex_location,
        vertex_normal,
        albedo_color,
        camera_location,
        light_origin,
        light_color,
        light_intensity,
        ambient_light_intensity,
        bands=4,
    ):
        return albedo_color


@pytest.mark.parametrize(
    "shader",
    [
        my_shader,
        _renamed,
        _renamed_with_defaults,
        _ambient_default,
        _any_number,
        _Callable(),
        *(
            getattr(module, name)
            for module in (material_shaders, pbr_shaders)
            for name in dir(module)
            if name.endswith("_shader") and not name.startswith("_")
        ),
    ],
)
def test_a_shader_that_takes_the_nine_in_order_is_set_without_a_word(shader):
    with Scene(video_settings=SMOKE_TEST) as scene, warnings.catch_warnings():
        warnings.simplefilter("error", AlganWarning)
        sphere = Sphere(scene=scene).set_shader(shader)
    assert sphere.shader is shader


def test_a_material_sets_its_shader_without_a_word():
    with Scene(video_settings=SMOKE_TEST) as scene, warnings.catch_warnings():
        warnings.simplefilter("error", AlganWarning)
        Sphere(scene=scene).set_material(MeshStandardMaterial(roughness=0.2))
