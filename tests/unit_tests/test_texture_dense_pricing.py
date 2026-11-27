"""A color texture's window is priced for the window being fetched.

A Surface prices its color texture off its previous primitive build: a
window that build found constant, or described as segments, costs about one
image a batch, where a dense animated window costs several images a frame.
The first window of a cross-fade the segment description declines (an updater
depends on the Surface, say) is dense and animated, yet it was priced as the
still window before it. Sized two to three times too long, its per-frame
texels, decode, mips and merge copy overran the render device, which under
WDDM pages into shared system memory rather than failing: four such batches
were 100 s of a 145 s render.

``AnimationTimeline.dense_window_changes`` answers, from the recording, for
the window the batch sizer is choosing; these tests pin its answers, the
second price the sizer takes with it, and the shorter fetch that results.

Feature tests for the texture path: unmarked, so outside the fast suite.
"""

import math

import pytest
import torch

from algan import Scene, Surface
from algan.animation_timeline import timeline as timeline_module
from algan.animation_timeline.animation_contexts import AnimationContext, Off
from algan.mobs.shapes_3d import Sphere
from algan.scene_manager import SceneManager


def _tex(width, height, seed=0):
    g = torch.Generator().manual_seed(seed)
    tex = torch.rand(width, height, 5, generator=g)
    tex[..., 3] = 0.0
    tex[..., 4] = 1.0
    return tex


def _build(scene, surface, times):
    """Materialize ``times`` and build the primitive, as a fetch does, so the
    Surface carries what that window's build observed.
    """
    timeline = scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(times, active_mobs=list(scene.actors))
        seg = timeline.segment_window_for(surface._color_texture_attr, surface.id)
        surface.get_render_primitives()
        timeline.clear_buffers()
    return seg


def _crossfade(make, *, updater, instant=False, size=8):
    """Still on [1, 3), fading (or stepping) to a second map from 3 to 5."""
    scene = SceneManager.reset()
    surface = make(_tex(size, size, 1)).spawn()
    if updater:
        # Depends on the Surface without ever touching its texture: the
        # segment description declines the Mob all the same.
        surface.add_updater(lambda mob, t: mob.set(opacity=1.0))
    Scene.wait(2)
    if instant:
        with Off():
            surface.color_texture = _tex(size, size, 2)
        Scene.wait(2)
    else:
        with AnimationContext(runtime=2):
            surface.color_texture = _tex(size, size, 2)
    Scene.wait(1)
    return scene, surface


def _plane(tex):
    return Surface(color_texture=tex, grid_height=4, grid_width=4)


def _sphere(tex):
    return Sphere(radius=1.0, color_texture=tex)


STILL = torch.linspace(1.2, 2.8, 5)
FADING = torch.linspace(2.6, 4.2, 5)
AFTER = torch.linspace(5.2, 5.8, 4)


def test_a_declined_crossfade_window_is_dense_and_changes():
    scene, surface = _crossfade(_plane, updater=True)
    timeline = scene.timeline_manager
    attr = surface._color_texture_attr

    assert timeline.dense_window_changes(attr, surface.id, FADING)
    # The same Mob's still windows are dense too (the updater declines them)
    # but nothing recorded writes the rows there: the build collapses them,
    # and the previous build's price is right.
    assert not timeline.dense_window_changes(attr, surface.id, STILL)
    assert not timeline.dense_window_changes(attr, surface.id, AFTER)
    # What the probe promised is what the fetch then gets.
    assert _build(scene, surface, FADING) is None
    assert not surface._texture_window_collapsed
    assert not surface._texture_window_lerp


def test_a_described_crossfade_window_is_not_dense():
    scene, surface = _crossfade(_plane, updater=False)
    timeline = scene.timeline_manager
    attr = surface._color_texture_attr

    assert not timeline.dense_window_changes(attr, surface.id, FADING)
    assert not timeline.dense_window_changes(attr, surface.id, STILL)
    # The probe built no description and left none behind.
    assert timeline.segment_window_for(attr, surface.id) is None
    seg = _build(scene, surface, FADING)
    assert seg is not None
    assert seg.endpoints.shape[0] > 1


def test_an_instant_reassignment_inside_a_declined_window_changes_it():
    scene, surface = _crossfade(_plane, updater=True, instant=True)
    timeline = scene.timeline_manager
    attr = surface._color_texture_attr

    # The step lands at t = 3: a window holding frames on both sides of it
    # holds both maps; one that starts on it, or ends before it, holds one.
    assert timeline.dense_window_changes(attr, surface.id, torch.linspace(2.6, 3.4, 5))
    assert not timeline.dense_window_changes(
        attr, surface.id, torch.linspace(3.0, 3.8, 5)
    )
    assert not timeline.dense_window_changes(attr, surface.id, STILL)


def test_other_attributes_and_unknown_mobs_answer_false():
    scene, surface = _crossfade(_plane, updater=True)
    timeline = scene.timeline_manager

    assert not timeline.dense_window_changes("opacity", surface.id, FADING)
    assert not timeline.dense_window_changes(
        surface._color_texture_attr, surface.id + 10_000, FADING
    )
    assert not timeline.dense_window_changes(
        surface._color_texture_attr, surface.id, FADING[:0]
    )


def test_the_sizer_takes_a_second_price_for_a_texture_read_as_cheap():
    # A closed surface's dense window is the window plus its per-frame wrap
    # pad; collapsed, it is the window alone.
    scene, surface = _crossfade(_sphere, updater=True)
    _build(scene, surface, STILL)
    assert surface._texture_window_collapsed
    frame = 8 * 8 * 5 * 4
    assert surface._color_texture_bytes_per_timestep() == frame

    actor_mem, dense_actor_mem = scene._price_actors_per_timestep([(surface, 0.0)])

    assert len(actor_mem) == 1
    (spawn, keys, extra, extra_render) = dense_actor_mem[0]
    assert keys == ((surface._color_texture_attr, surface.id),)
    assert (spawn, extra, extra_render) == (0.0, frame, 0)
    # The scope is closed again: outside the sizer the estimate reads the
    # previous build, as it always has.
    assert scene.timeline_manager._sizing_keys is None
    assert surface._color_texture_bytes_per_timestep() == frame


def test_a_texture_already_priced_dense_has_no_second_price():
    scene, surface = _crossfade(_sphere, updater=True)
    _build(scene, surface, FADING)
    assert not surface._texture_window_collapsed

    _, dense_actor_mem = scene._price_actors_per_timestep([(surface, 0.0)])

    assert dense_actor_mem == []


def test_the_kill_switch_prices_off_the_previous_build_alone(monkeypatch):
    scene, surface = _crossfade(_sphere, updater=True)
    _build(scene, surface, STILL)
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset({"densepricing"}))

    actor_mem, dense_actor_mem = scene._price_actors_per_timestep([(surface, 0.0)])

    assert dense_actor_mem == []
    assert actor_mem[0][1] >= 8 * 8 * 5 * 4


@pytest.mark.parametrize("disabled", [False, True])
def test_a_fetch_that_reaches_a_declined_crossfade_is_sized_dense(
    monkeypatch, disabled
):
    """The fetch starts on still frames and could run into the fade. Priced
    off the still build alone it takes every frame its budget allows at the
    collapsed price; priced for the window, a fetch that reaches the fade
    has to fit at the dense price, and this one stops just short of it.
    """
    # A map wide enough that its per-frame pad copy is a third of the
    # Sphere's whole price.
    scene, surface = _crossfade(_sphere, updater=True, size=64)
    if disabled:
        monkeypatch.setattr(
            timeline_module, "_OPT_DISABLED", frozenset({"densepricing"})
        )
    fps = scene.frames_per_second
    scene.scene_times.append([0, round(6 * fps)])
    scene._initialize_frames()
    _build(scene, surface, STILL)
    assert surface._texture_window_collapsed

    actor_mem, dense_actor_mem = scene._price_actors_per_timestep([(surface, 0.0)])
    still_price = actor_mem[0][1]
    start = round(1.5 * fps)
    fade_starts = round(3 * fps)
    # Room for the still frames up to the fade and a few fading ones at the
    # collapsed price, but at the dense price not even for the still ones.
    frames = (fade_starts - start) + 6
    budget = still_price * frames

    with scene._batch_prep_context():
        _, end, _ = scene._get_batch_of_primitives(
            start, round(6 * fps), [surface], budget
        )

    if disabled:
        assert end - start == frames
    else:
        extra = dense_actor_mem[0][2]
        assert extra == 64 * 64 * 5 * 4
        assert budget // (still_price + extra) < fade_starts - start
        # Every window that ends before the fade's first frame is still
        # priced as still, so that is where the fetch ends.
        assert end - start == fade_starts - start
