"""Material arguments are checked when the material is made.

Render-free. A material scalar outside its meaning (a negative roughness, an
IOR of zero) is refused with an error naming it, a value outside its
documented range is accepted with a warning, and an argument the renderer
accepts for three.js parity but does nothing with warns once, naming it, when
the material is applied.
"""

import math
import warnings

import pytest

from algan import BLUE, RED, Scene, Square
from algan.errors import AlganConfigurationError, UnsupportedFeatureWarning
from algan.rendering.shaders.materials import (
    MeshBasicMaterial,
    MeshDepthMaterial,
    MeshLambertMaterial,
    MeshNormalMaterial,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    MeshToonMaterial,
)


@pytest.mark.parametrize(
    ("cls", "kwargs", "name"),
    [
        (MeshStandardMaterial, {"roughness": -0.5}, "roughness"),
        (MeshStandardMaterial, {"metalness": -1.0}, "metalness"),
        (MeshStandardMaterial, {"emissive_intensity": -2.0}, "emissive_intensity"),
        (MeshStandardMaterial, {"roughness": math.nan}, "roughness"),
        (MeshStandardMaterial, {"metalness": "shiny"}, "metalness"),
        (MeshLambertMaterial, {"env_map_intensity": math.inf}, "env_map_intensity"),
        (MeshPhongMaterial, {"shininess": -5.0}, "shininess"),
        (MeshPhysicalMaterial, {"clearcoat": -1.0}, "clearcoat"),
        (MeshPhysicalMaterial, {"transmission": -1.0}, "transmission"),
        (MeshPhysicalMaterial, {"sheen": -0.1}, "sheen"),
        (MeshPhysicalMaterial, {"specular_intensity": -1.0}, "specular_intensity"),
        (MeshPhysicalMaterial, {"thickness": -1.0}, "thickness"),
        (MeshPhysicalMaterial, {"attenuation_distance": math.nan}, "attenuation"),
        # These two used to end in a bare ZeroDivisionError.
        (MeshPhysicalMaterial, {"ior": -1.0}, "ior"),
        (MeshPhysicalMaterial, {"reflectivity": 2.5}, "reflectivity"),
        (MeshPhysicalMaterial, {"ior": 0.0}, "ior"),
        (MeshPhysicalMaterial, {"reflectivity": -0.5}, "reflectivity"),
        (MeshToonMaterial, {"bands": 0}, "bands"),
        (MeshDepthMaterial, {"near": 5.0, "far": 1.0}, "near"),
    ],
)
def test_a_meaningless_material_scalar_is_refused_by_name(cls, kwargs, name):
    with pytest.raises(AlganConfigurationError, match=name):
        cls(**kwargs)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (MeshStandardMaterial, {"metalness": 3.0}),
        (MeshStandardMaterial, {"roughness": 1.5}),
        (MeshPhysicalMaterial, {"clearcoat": 4.0}),
        (MeshPhysicalMaterial, {"transmission": 1.2}),
    ],
)
def test_a_value_past_the_top_of_its_range_warns_and_is_kept(cls, kwargs):
    ((name, value),) = kwargs.items()
    with pytest.warns(UnsupportedFeatureWarning, match=f"{name}={value:g}"):
        material = cls(**kwargs)
    assert getattr(material, name) == value


def test_an_index_below_air_warns_and_is_kept():
    """A relative index below 1 (an air bubble in water) is not refused."""
    with pytest.warns(UnsupportedFeatureWarning, match="ior=0.75 is below 1"):
        bubble = MeshPhysicalMaterial(ior=0.75)
    assert bubble.ior == 0.75


def test_valid_materials_are_accepted_silently_and_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        physical = MeshPhysicalMaterial(
            roughness=0.0,
            metalness=1,
            clearcoat=1.0,
            ior=1.0,
            transmission=1,
            sheen=8.0,  # a deliberate boost, as the neural-net shells use
            attenuation_distance=math.inf,
        )
        from_reflectivity = MeshPhysicalMaterial(reflectivity=0.5)
        MeshToonMaterial(bands=1)
        MeshDepthMaterial(near=0.0, far=1.0)
    assert physical.metalness == 1
    assert isinstance(physical.metalness, int)
    assert from_reflectivity.ior == pytest.approx(1.2 / 0.8)


@pytest.mark.parametrize(
    ("material", "argument"),
    [
        (MeshBasicMaterial(visible=False), "visible=False"),
        (MeshStandardMaterial(tone_mapped=False), "tone_mapped=False"),
        (MeshPhysicalMaterial(iridescence=1.0), "iridescence is not rendered"),
        (MeshPhysicalMaterial(iridescence_ior=1.8), "iridescence_ior"),
        (MeshNormalMaterial(color=BLUE), "color does nothing"),
        (MeshDepthMaterial(color=BLUE), "color does nothing"),
    ],
)
def test_an_argument_no_renderer_reads_warns_once_naming_it(material, argument):
    with Scene(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Square(color=RED).set_material(material)
    named = [w for w in caught if argument in str(w.message)]
    assert len(named) == 1, [str(w.message) for w in caught]
    assert issubclass(named[0].category, UnsupportedFeatureWarning)


def test_default_arguments_do_not_warn():
    with Scene(), warnings.catch_warnings():
        warnings.simplefilter("error")
        Square().set_material(MeshPhysicalMaterial())
        Square().set_material(MeshNormalMaterial())
        Square().set_material(MeshBasicMaterial(opacity=0.5))
        # three.js's own spelling of a see-through material, which is also
        # what Algan renders, so it has nothing to report.
        Square().set_material(MeshStandardMaterial(transparent=True, opacity=0.5))
