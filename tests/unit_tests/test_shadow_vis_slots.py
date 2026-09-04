"""The per-batch size of the shadow-visibility payload.

``max_shadow_lights`` is a compile-time *cap* on how many lights can cast
ray-traced shadows. It used to also be the length of the per-fragment ``vis`` /
``lvis`` vector in all three shading kernels -- and that vector is indexed by a
runtime light ordinal, so it does not stay in registers: it becomes a per-thread
local-memory array, re-initialised on every drained surface. At the default cap
that is 192 bytes per thread to carry, for the default one-light rig, twelve.

``shadow_vis_slots`` sizes it per batch instead. The whole change is invisible
only because of the invariant tested here: **the slot count is never below what
the batch needs**, so the in-kernel bound ``li < slots`` accepts exactly the
lights ``li < max_shadow_lights`` accepted before. If that ever stops holding, a
light silently loses its shadow.
"""

from __future__ import annotations

import pytest

from algan.rendering.raytracing.shading_taichi import (
    SHADOW_VIS_CHANNELS,
    max_shadow_lights,
    shadow_vis_slots,
)

# Past the cap as well, since that is where truncation takes over.
COUNTS = list(range(0, 2 * max_shadow_lights + 3))


@pytest.mark.parametrize("num_lights", COUNTS)
def test_every_light_the_cap_would_shadow_still_has_a_slot(num_lights):
    """The acceptance test in the kernels must not narrow.

    Before: a light was shadowed iff ``li < max_shadow_lights``. After: iff
    ``li < shadow_vis_slots(num_lights)``. For those to agree on every light
    the batch actually has, the slot count has to cover the batch -- or, once
    the batch exceeds the cap, to equal the cap so the same lights truncate.
    """
    slots = shadow_vis_slots(num_lights)
    assert slots >= min(num_lights, max_shadow_lights)


@pytest.mark.parametrize("num_lights", COUNTS)
def test_slots_stay_within_the_cap(num_lights):
    """The cap is still the ceiling -- it is what the kernels' truncation
    counting and the tracer's warning are stated against.
    """
    assert 1 <= shadow_vis_slots(num_lights) <= max_shadow_lights


@pytest.mark.parametrize("num_lights", COUNTS)
def test_slots_are_powers_of_two(num_lights):
    """Bucketed, so the number of compiled kernel variants stays bounded.

    Sizing the vector by the exact light count would compile a fresh variant of
    three megakernels for every distinct count a scene happens to use, and each
    of those costs tens of seconds of Taichi compile.
    """
    slots = shadow_vis_slots(num_lights)
    assert slots == max_shadow_lights or (slots & (slots - 1)) == 0


def test_the_default_one_light_rig_gets_one_slot():
    """The case that motivates this: `default_scene_initializer` spawns a
    single PointLight, and the payload was carrying sixteen lights' worth.
    """
    assert shadow_vis_slots(1) == 1
    assert SHADOW_VIS_CHANNELS * shadow_vis_slots(1) == 3


def test_slots_never_shrink_as_lights_are_added():
    """Monotone, so adding a light can never take a slot away from one that
    already had one.
    """
    previous = 0
    for num_lights in COUNTS:
        slots = shadow_vis_slots(num_lights)
        assert slots >= previous
        previous = slots


def test_a_batch_with_no_lights_still_gets_a_valid_vector():
    """A zero-length ``ti.Vector`` is not constructible, and a shadowless batch
    still instantiates the kernel.
    """
    assert shadow_vis_slots(0) >= 1
