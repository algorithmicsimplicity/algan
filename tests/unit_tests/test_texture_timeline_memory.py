"""A colour texture is an ordinary animated attribute whose channel width is a
whole flattened image, so every per-row and per-frame cost in the timeline is
multiplied by ``H * W * 5``. That made a docs example ask for 87 GB in one
allocation. These tests pin the four things that keep it proportionate --
without giving up the per-texel interpolation that makes the attribute
animatable in the first place.

The render suites cannot cover this: ``tests/fast/scene.py`` has no textured
geometry at all, and the only pixel-compared texture scenes live in
``tests/full_renders``, which is outside the fast budget.
"""

import math

import torch

from algan import Scene, Surface
from algan.animation_timeline.animation_contexts import Off, Seq
from algan.animation_timeline.timeline import AttributeTimeline
from algan.constants import easings
from algan.scene_manager import SceneManager


def _texture(width, height, channel):
    """A ``[W, H, 5]`` opaque texture with one colour channel saturated."""
    texture = torch.zeros(width, height, 5)
    texture[..., channel] = 1.0
    texture[..., 4] = 1.0
    return texture


def test_wide_attribute_reserves_by_bytes_not_by_row_count():
    """A 256-row reservation is gigabytes when a row is a whole image."""
    narrow = AttributeTimeline(3)
    wide = AttributeTimeline(1774 * 887 * 5)

    assert narrow.current_state.shape[-2] == 256, (
        "ordinary attributes must keep their full speculative reservation"
    )
    # A row this wide is 31 MB, so the byte budget buys none: the reservation
    # falls to the 2-row floor that add() needs, and grows on demand from there.
    assert wide.current_state.shape[-2] == 2, (
        f"reserved {wide.current_state.shape[-2]} rows of a whole image each"
    )
    reserved_bytes = wide.current_state.numel() * wide.current_state.element_size()
    would_have_been = 256 * wide.current_state.shape[-1] * 4
    assert reserved_bytes * 100 < would_have_been, (
        f"reserved {reserved_bytes / 1e9:.2f} GB where a flat 256 rows would "
        f"have taken {would_have_been / 1e9:.2f} GB"
    )


def test_narrow_attributes_are_byte_for_byte_unchanged():
    """The byte budget must not shrink any attribute the engine registers."""
    for channels in (1, 3, 5, 9, 1024):
        timeline = AttributeTimeline(channels)
        assert timeline.current_state.shape[-2] == 256, (
            f"D={channels} should sit below the byte budget's break-even"
        )


def test_materialization_allocates_no_row_beyond_the_last_in_use():
    """Every row id in circulation is < pointer, so pointer rows is enough."""
    scene = SceneManager.reset()
    surface = Surface(
        color_texture=_texture(4, 4, 0), grid_height=4, grid_width=4
    ).spawn()
    scene.wait(1)

    timeline = scene.timeline_manager
    attr_timeline = timeline.attr_to_timeline[surface._color_texture_attr]
    times = torch.tensor([0.25, 0.75])
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(times)
        rows = attr_timeline.active_state.shape[1]
        timeline.clear_buffers()

    assert rows == attr_timeline.pointer, (
        f"materialized {rows} rows for {attr_timeline.pointer} in use -- a spare "
        f"row costs a whole extra image on every frame of the batch"
    )


def test_colour_texture_still_interpolates_per_texel():
    """The reason the texels live on the timeline at all: assigning a new
    same-resolution texture inside an animation context crossfades them.
    """
    SceneManager.reset()
    surface = Surface(
        color_texture=_texture(4, 4, 0), grid_height=4, grid_width=4
    ).spawn()
    with Seq(runtime=1, easing=easings.identity):
        surface.color_texture = _texture(4, 4, 2)

    scene = SceneManager.instance().current_scene
    timeline = scene.timeline_manager
    attr_timeline = timeline.attr_to_timeline[surface._color_texture_attr]
    row = int(attr_timeline.mob_id_to_inds[surface.id][0])

    times = torch.linspace(0.0, 2.0, 9)
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(times)
        state = attr_timeline.active_state[:, row].clone().view(-1, 4, 4, 5)
        timeline.clear_buffers()

    first, last = state[..., 0].mean(dim=(1, 2)), state[..., 2].mean(dim=(1, 2))
    assert first[0] > 0.99, "did not start on the first texture"
    assert last[0] < 0.01, "the second texture was already blended in at t=0"
    assert first[-1] < 0.01, "the first texture never faded out"
    assert last[-1] > 0.99, "did not end on the second texture"
    assert ((first > 0.01) & (first < 0.99)).any(), (
        "no partially blended frame -- the crossfade was lost"
    )
    assert torch.allclose(first + last, torch.ones_like(first), atol=1e-5)


def test_batch_sizer_prices_the_texture_and_notices_a_late_assignment():
    """The frame-batch search is the only thing bounding how many copies of a
    texture are materialized at once, and its cache key has to carry the
    texture: a surface priced before its texture was assigned would otherwise
    keep serving the texture-free estimate forever.
    """
    SceneManager.reset()
    surface = Surface(grid_height=4, grid_width=4).spawn()
    untextured = surface._get_memory_used_per_timestep()

    surface.color_texture = _texture(64, 32, 0)
    textured = surface._get_memory_used_per_timestep()

    texels = 64 * 32 * 5
    assert textured >= untextured + texels * 4, (
        f"estimate rose by only {textured - untextured} bytes for a texture "
        f"whose materialized state alone is {texels * 4} bytes per frame"
    )


def test_presence_probe_does_not_materialize_the_texture():
    """Asking whether a surface is textured must not clone the image."""
    SceneManager.reset()
    plain = Surface(grid_height=4, grid_width=4).spawn()
    textured = Surface(
        color_texture=_texture(4, 4, 0), grid_height=4, grid_width=4
    ).spawn()

    assert not plain._has_color_texture
    assert textured._has_color_texture

    scene = SceneManager.instance().current_scene
    attr_timeline = scene.timeline_manager.attr_to_timeline[
        textured._color_texture_attr
    ]
    reads = []
    original_get = AttributeTimeline.get

    def counting_get(self, key, copy=True):
        if self is attr_timeline:
            reads.append(copy)
        return original_get(self, key, copy=copy)

    AttributeTimeline.get = counting_get
    try:
        assert textured._has_color_texture
        assert not plain._has_color_texture
    finally:
        AttributeTimeline.get = original_get

    assert reads == [], f"presence probes read the texture buffer {len(reads)} times"


def test_render_primitive_reads_the_texture_without_cloning_it():
    """get_render_primitives asked for the texture three times per build, each
    a full clone. It reads it once, uncopied, and premultiplies out of place.
    """
    SceneManager.reset()
    surface = Surface(
        color_texture=_texture(8, 4, 0), grid_height=4, grid_width=4
    ).spawn()

    scene = SceneManager.instance().current_scene
    attr_timeline = scene.timeline_manager.attr_to_timeline[surface._color_texture_attr]
    copies = []
    original_get = AttributeTimeline.get

    def counting_get(self, key, copy=True):
        if self is attr_timeline and copy:
            copies.append(key)
        return original_get(self, key, copy=copy)

    AttributeTimeline.get = counting_get
    try:
        primitive = surface.get_render_primitives()
    finally:
        AttributeTimeline.get = original_get

    assert primitive.texture_map is not None
    assert copies == [], f"the texture was cloned {len(copies)} times for one build"
    # Premultiplying opacity must not write through the uncopied view.
    materialized = attr_timeline.current_state[
        :, attr_timeline.mob_id_to_inds[surface.id]
    ]
    assert torch.equal(materialized.view(8, 4, 5)[..., 0], torch.ones(8, 4)), (
        "the timeline's own texels were modified by the primitive build"
    )


def test_scene_render_still_works_end_to_end_with_a_texture():
    """Guards the whole chain above against a shape error the unit tests miss."""
    SceneManager.reset()
    Surface(color_texture=_texture(4, 4, 2), grid_height=4, grid_width=4).spawn()
    result = Scene.save_frame()
    assert result.rendered
