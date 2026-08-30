"""Every authored colour crosses into linear light exactly once.

Under the linear working space (``SETTINGS.raytracing.linear_color_space``,
default on) the sRGB OETF runs at the byte write, so an authored colour has to be
*decoded* on the way in or the round trip is not the identity. When it is not,
nothing errors: every colour simply renders too bright, by a factor that varies
per channel, so saturated colours wash out and hues shift. An authored 0.5 grey
comes out 188 instead of 128.

Three routes carry authored colour into the render kernel and each is a separate
opportunity to miss the decode:

* the per-vertex colour arrays (``tri_colors`` and the two circuit arrays);
* the **colour texture maps** -- which is where most content actually goes.
  Constant-property promotion (``ALGAN_PROMOTE_CONSTANTS``, on by default)
  renders a mob whose colour and material are uniform from a shared 1x1 colour
  map instead of per-vertex rows, so an ordinary cube in an ordinary colour never
  touches ``tri_colors`` at all;
* the colour slots of the material parameter block -- ``emissive``, ``specular``,
  ``specular_color``, ``sheen_color``.

The first two tests here are the end-to-end statement of the invariant, rendered:
a surface that neither reflects nor is lit must come back off the frame buffer
the colour it was authored. They render because that is the only place the round
trip is observable -- the decode and the encode are at opposite ends of the
pipeline, and a merge-level assertion would pin one half against the other's
absence. They are small (48x48, one frame, no anti-aliasing) and they are feature
tests of the render boundary rather than of anything the timeline or the Scene
can break, so they stay out of the fast suite.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from algan import (
    MeshBasicMaterial,
    MeshStandardMaterial,
    Off,
    Prism,
    Scene,
    SceneManager,
    VideoSettings,
)
from algan.rendering.raytracing import scene_builder as sb
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.settings import _MAT_SLOTS

#: Deliberately three different channels, none of them a fixed point of the
#: transfer function, so a missing decode cannot hide in a grey.
_AUTHORED = (0.5, 0.25, 0.75)
_AUTHORED_BYTES = (128, 64, 191)


def _render_centre_pixel(tmp_path, material):
    """The centre pixel of a slab filling the frame, as (R, G, B) bytes."""
    SceneManager.instance().reset()
    video = VideoSettings((48, 48), 30, supersampling=1)
    with rt_settings_off_tonemap():
        with Off():
            slab = Prism(width=12.0, height=12.0, depth=0.5)
            slab.set_material(material)
            slab.move_to(torch.tensor((0.0, 0.0, 0.0)))
            slab.spawn(animate=False)
        out = tmp_path / "probe.png"
        Scene.save_frame(str(out), video)
    im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert im is not None, "the probe render produced no file"
    h, w = im.shape[:2]
    return tuple(int(v) for v in im[h // 2, w // 2, 2::-1])


class rt_settings_off_tonemap:  # noqa: N801 - a context manager, not a class API
    """Tonemapping off for the duration, restored afterwards.

    It is off by default, but a curve in the pipeline would be a second transfer
    function and these tests are about the first one.
    """

    def __enter__(self):
        from algan.settings import SETTINGS

        self._previous = SETTINGS.raytracing.tonemapping
        SETTINGS.raytracing.set(tonemapping=False)
        return self

    def __exit__(self, *exc):
        from algan.settings import SETTINGS

        SETTINGS.raytracing.set(tonemapping=self._previous)
        return False


@pytest.mark.skipif(
    not rt_settings.linear_color_space,
    reason="the display-referred pipeline neither decodes nor encodes",
)
def test_an_unlit_authored_colour_renders_as_itself(tmp_path):
    """The round trip, on the route promotion actually takes.

    A uniformly-coloured, uniformly-shaded slab is promoted to a 1x1 colour map
    in ``scene["textures"]``; that map used not to be decoded, so this rendered
    (188, 137, 225) -- brighter and hue-shifted -- while ``tri_colors``, the one
    route that was decoded, sat empty.
    """
    got = _render_centre_pixel(tmp_path, MeshBasicMaterial(color=_AUTHORED))
    assert all(abs(a - b) <= 1 for a, b in zip(got, _AUTHORED_BYTES)), got


@pytest.mark.skipif(
    not rt_settings.linear_color_space,
    reason="the display-referred pipeline neither decodes nor encodes",
)
def test_an_emissive_colour_renders_as_itself(tmp_path):
    """Emissive is light the surface adds, so it is authored colour too.

    It rides the material parameter block rather than any colour array, which is
    the third route and was the third one missed. The base colour is black so
    nothing the lights do can contribute.
    """
    got = _render_centre_pixel(
        tmp_path,
        MeshStandardMaterial(
            color=(0.0, 0.0, 0.0), roughness=1.0, metalness=0.0, emissive=_AUTHORED
        ),
    )
    assert all(abs(a - b) <= 2 for a, b in zip(got, _AUTHORED_BYTES)), got


def test_only_the_colour_slots_of_the_material_block_are_decoded():
    """A scalar coefficient sharing the block must not move.

    ``ior`` is the one that shows it: 1.5 is not a fixed point of the transfer
    function, where roughness 1 and metalness 0 both are.
    """
    if not rt_settings.linear_color_space:
        pytest.skip("the display-referred pipeline does not decode at all")
    width = max(start + w for start, w in _MAT_SLOTS.values())
    mat = torch.zeros((1, 2, width))
    e_start, e_width = _MAT_SLOTS["emissive"]
    mat[..., e_start : e_start + e_width] = 0.5
    ior_slot, _ = _MAT_SLOTS["ior"]
    mat[..., ior_slot] = 1.5
    rough_slot, _ = _MAT_SLOTS["roughness"]
    mat[..., rough_slot] = 0.5
    scene = {"tri_mat": mat, "tri_mat_id": torch.zeros((1, 2), dtype=torch.int32)}

    sb._decode_merged_colors(scene)

    assert np.isclose(float(mat[0, 0, e_start]), 0.21404, atol=1e-4)
    assert float(mat[0, 0, ior_slot]) == 1.5
    assert float(mat[0, 0, rough_slot]) == 0.5


def test_a_custom_pipeline_block_is_left_alone():
    """A custom fragment pipeline packs its own layout into the same array, so
    those slots are not colours and decoding them would corrupt the pipeline's
    parameters.
    """
    if not rt_settings.linear_color_space:
        pytest.skip("the display-referred pipeline does not decode at all")
    width = max(start + w for start, w in _MAT_SLOTS.values())
    mat = torch.full((1, 1, width), 0.5)
    scene = {
        "tri_mat": mat,
        "tri_mat_id": torch.full(
            (1, 1), rt_settings._USER_PIPELINE_BASE, dtype=torch.int32
        ),
    }

    sb._decode_merged_colors(scene)

    assert torch.allclose(mat, torch.full_like(mat, 0.5))


def test_the_decode_is_a_no_op_under_the_display_referred_pipeline(monkeypatch):
    """With ``linear_color_space`` off there is no OETF at the byte write, so
    there must be no decode either -- the two halves are one switch.
    """
    monkeypatch.setattr(rt_settings, "linear_color_space", False)
    width = max(start + w for start, w in _MAT_SLOTS.values())
    scene = {
        "tri_colors": torch.full((1, 1, 3, 5), 0.5),
        "tri_mat": torch.full((1, 1, width), 0.5),
        "tri_mat_id": torch.zeros((1, 1), dtype=torch.int32),
    }

    sb._decode_merged_colors(scene)

    assert torch.allclose(scene["tri_colors"], torch.full((1, 1, 3, 5), 0.5))
    assert torch.allclose(scene["tri_mat"], torch.full_like(scene["tri_mat"], 0.5))
