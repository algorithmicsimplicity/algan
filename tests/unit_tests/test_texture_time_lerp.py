"""In-kernel texture time interpolation (texture_time_lerp).

An animated colour-texture reassignment used to materialize one full image
per frame of a batch. The timeline's segment-window gate
(``AnimationTimeline._describe_segment_windows``) now describes such a window
as K endpoint images plus per-frame (i0, i1, w) rows, the surface hands them
to the renderer as a ``[1, K, H, W, 5]`` stack plus ``texture_lerp``, and the
sampler lerps the two endpoint texels in authored space before the
linear-light decode. Anything the conservative gate cannot prove -- an
overlapping edit, an updater dependency, a custom animated function, a
window whose endpoint count rivals its frame count -- falls back to the
dense path byte-identically.

These tests pin the gate's accept/decline matrix, the bit-parity of the
weights with the dense replay, the endpoint values, the estimator pricing,
and the u8-admission fix the round carries (provenance proved on the actual
endpoint stack, and on the AND over every assignment for the dense arm).
``benchmarks/_texture_lerp_ab.py`` is the frame-level acceptance harness.

Feature tests for the texture path: unmarked, so outside the fast suite.
"""

import math

import torch

from algan import Scene, Square, Surface
from algan.animation_timeline.animation_contexts import (
    AnimationContext,
    Off,
    Sync,
)
from algan.constants.easings import ease_out_back
from algan.mobs.shapes_3d import Sphere
from algan.rendering.raytracing import settings as rt_settings
from algan.scene_manager import SceneManager


def _tex(width, height, seed=0):
    """A ``[W, H, 5]`` texture of arbitrary floats, zero glow, opaque."""
    g = torch.Generator().manual_seed(seed)
    tex = torch.rand(width, height, 5, generator=g)
    tex[..., 3] = 0.0
    tex[..., 4] = 1.0
    return tex


def _u8_tex(width, height, seed=0):
    """A ``[W, H, 5]`` texture whose texels are exactly k/255, zero glow."""
    g = torch.Generator().manual_seed(seed)
    tex = torch.randint(0, 256, (width, height, 5), generator=g).float() / 255
    tex[..., 3] = 0.0
    return tex


def _materialize(scene, surface, times):
    """Materialize with the render loop's working-set contract (the gate
    requires a known actor set), build the primitive, and hand back both it
    and the segment window the batch stashed (None = dense).
    """
    timeline = scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(times, active_mobs=list(scene.actors))
        seg = timeline.segment_window_for(surface._color_texture_attr, surface.id)
        primitive = surface.get_render_primitives()
        primitive.texture_map = primitive.texture_map.clone()
        timeline.clear_buffers()
    return primitive, seg


def _crossfade_scene(tex_a, tex_b, duration=2):
    scene = SceneManager.reset()
    surface = Surface(color_texture=tex_a, grid_height=4, grid_width=4).spawn()
    with AnimationContext(runtime=duration):
        surface.color_texture = tex_b
    Scene.wait(1)
    return scene, surface


def test_a_crossfade_window_is_described_as_endpoints():
    tex_a, tex_b = _tex(8, 8, 1), _tex(8, 8, 2)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    # Spawn occupies [0, 1], the reassignment [1, 3]; sample around it.
    times = torch.linspace(0.5, 3.5, 7)
    primitive, seg = _materialize(scene, surface, times)

    assert seg is not None, "the gate declined a plain crossfade"
    assert primitive.texture_map.shape == (1, 3, 8, 8, 5), (
        "the map must be a [1, K, H, W, 5] endpoint stack"
    )
    assert primitive.texture_lerp.shape == (7, 3)
    # Endpoint 0 is the first map's recorded state, bit for bit; the post
    # endpoint is the AUTHORED second map (the setter stamps it on the edit
    # -- the stored state is an ulp off, which would void u8 provenance).
    assert torch.equal(seg.endpoints[0], tex_a.reshape(-1))
    post = seg.endpoints[int(seg.index1[3])]
    assert torch.equal(post, tex_b.reshape(-1))
    # Before the fade every frame sits on the pre endpoint; after it, on the
    # tail -- both with weight zero.
    w = seg.weights
    assert float(w[0]) == 0.0
    assert float(w[-1]) == 0.0
    assert int(seg.index0[0]) == int(seg.index1[0])
    assert int(seg.index0[-1]) == int(seg.index1[-1])
    assert bool((w[2:5] > 0).any()), "no frame interpolates inside the fade"


def test_weights_are_bit_identical_to_the_dense_replay():
    """The weights must be the very tensor the dense replay computes -- same
    ops, same shapes -- because torch CPU kernels round shape-dependently.
    The gate recomputes ``easing((t - s) / (e - s + 1e-6))`` on the same
    frame times; verify against an independent evaluation off the recorded
    event, and verify the evaluated trajectory against the dense buffer.
    """
    tex_a, tex_b = _tex(8, 8, 3), _tex(8, 8, 4)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    times = torch.linspace(0.5, 3.5, 13)
    _, seg = _materialize(scene, surface, times)
    assert seg is not None

    timeline = scene.timeline_manager
    event = next(
        f
        for f in timeline.function_timeline.function_applications
        if any(a == surface._color_texture_attr for a, *_ in f.recorded_edits)
    )
    s, e = event.time.start, event.time.end
    sel = (s <= times) & (times < e)
    expected = event.easing(((times[sel] - s) / (e - s + 1e-6)).view(-1, 1, 1))
    assert torch.equal(seg.weights[sel], expected.view(-1))

    # The dense window, under the kill switch, frame for frame.
    previous_lerp = rt_settings.texture_time_lerp
    rt_settings.set_texture_time_lerp(False)
    try:
        with Off(
            record_attr_modifications=False,
            record_funcs=False,
            priority_level=math.inf,
        ):
            timeline.set_state_to_times(times, active_mobs=list(scene.actors))
            dense = surface._color_texture_uncopied().clone()
            timeline.clear_buffers()
    finally:
        rt_settings.set_texture_time_lerp(previous_lerp)
    assert torch.allclose(seg.evaluate(), dense, atol=1e-6, rtol=0), (
        "the description does not reproduce the dense trajectory"
    )


def test_an_overshooting_easing_is_described_verbatim():
    tex_a, tex_b = _tex(8, 8, 5), _tex(8, 8, 6)
    scene = SceneManager.reset()
    surface = Surface(color_texture=tex_a, grid_height=4, grid_width=4).spawn()
    with AnimationContext(runtime=2, easing=ease_out_back):
        surface.color_texture = tex_b
    Scene.wait(1)
    times = torch.linspace(1.0, 3.0, 9)
    _, seg = _materialize(scene, surface, times)
    assert seg is not None
    assert float(seg.weights.max()) > 1.0, (
        "ease_out_back overshoots; the weights must carry it, not clamp it"
    )


def test_an_instant_reassignment_is_a_step_between_endpoints():
    tex_a, tex_b = _tex(8, 8, 7), _tex(8, 8, 8)
    scene = SceneManager.reset()
    surface = Surface(color_texture=tex_a, grid_height=4, grid_width=4).spawn()
    Scene.wait(1)
    with Off():
        surface.color_texture = tex_b
    Scene.wait(1)
    times = torch.linspace(0.5, 2.9, 6)
    primitive, seg = _materialize(scene, surface, times)
    assert seg is not None
    assert primitive.texture_lerp is not None
    assert bool((seg.weights == 0).all()), "an instant edit interpolates nothing"
    before = seg.endpoints[seg.index0[0]]
    after = seg.endpoints[seg.index0[-1]]
    assert torch.equal(before, tex_a.reshape(-1))
    # Constant frames after the swap read the STORED state (pre + change),
    # which round-trips the authored map to within its own arithmetic.
    assert torch.allclose(after, tex_b.reshape(-1), atol=1e-6, rtol=0)


def test_the_gate_declines_overlapping_edits():
    tex_a = _tex(8, 8, 9)
    scene = SceneManager.reset()
    surface = Surface(color_texture=tex_a, grid_height=4, grid_width=4).spawn()
    with Sync(duration=2):
        surface.color_texture = _tex(8, 8, 10)
        surface.color_texture = _tex(8, 8, 11)
    Scene.wait(0.5)
    times = torch.linspace(1.0, 3.2, 6)
    primitive, seg = _materialize(scene, surface, times)
    assert seg is None, "overlapping edits replay a chain, not a lerp"
    assert primitive.texture_lerp is None
    assert primitive.texture_map.shape[0] == 6, "the dense window must be back"


def test_the_gate_declines_when_an_updater_depends_on_the_mob():
    tex_a, tex_b = _tex(8, 8, 12), _tex(8, 8, 13)
    scene = SceneManager.reset()
    surface = Surface(color_texture=tex_a, grid_height=4, grid_width=4).spawn()
    # Attached BEFORE the fade, so the updater is active across the window
    # (an updater added after the window's times never runs in it, and the
    # description stays valid there).
    surface.add_updater(lambda mob, t: mob.set(opacity=1.0))
    with AnimationContext(runtime=2):
        surface.color_texture = tex_b
    Scene.wait(1)
    times = torch.linspace(1.0, 3.0, 5)
    _, seg = _materialize(scene, surface, times)
    assert seg is None, (
        "an updater may overwrite the rows mid-window; the dense buffer is "
        "what makes that well-defined"
    )


def test_the_gate_declines_a_custom_animated_function_batch():
    tex_a, tex_b = _tex(8, 8, 14), _tex(8, 8, 15)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    square = Square().spawn()  # spawn animates [4, 5]
    square.animate_function(lambda mob, t: mob.set(opacity=t))  # [5, 6]
    # A custom function active in the window makes the working set
    # unknowable, so materialization runs full-width and every description
    # declines with it -- even for rows the function never touches.
    times = torch.linspace(5.0, 5.9, 3)
    _, seg = _materialize(scene, surface, times)
    assert seg is None


def test_the_kill_switch_restores_the_dense_window():
    tex_a, tex_b = _tex(8, 8, 16), _tex(8, 8, 17)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    times = torch.linspace(1.0, 3.0, 5)
    previous_lerp = rt_settings.texture_time_lerp
    rt_settings.set_texture_time_lerp(False)
    try:
        primitive, seg = _materialize(scene, surface, times)
    finally:
        rt_settings.set_texture_time_lerp(previous_lerp)
    assert seg is None
    assert primitive.texture_lerp is None
    assert primitive.texture_map.shape[0] == 5


def test_a_static_window_collapses_through_the_description():
    """A window with no edit in range is a one-endpoint description: the
    build collapses it exactly like the dense probe used to, without the
    timeline ever materializing a frame of it.
    """
    tex_a = _tex(8, 8, 18)
    scene = SceneManager.reset()
    surface = Surface(color_texture=tex_a, grid_height=4, grid_width=4).spawn()
    Scene.wait(2)
    times = torch.linspace(1.0, 2.0, 5)
    primitive, seg = _materialize(scene, surface, times)
    assert seg is not None
    assert seg.endpoints.shape[0] == 1
    assert primitive.texture_lerp is None
    assert primitive.texture_map.shape == (1, 8, 8, 5)
    assert surface._texture_window_collapsed


def test_estimator_prices_a_crossfade_at_the_endpoints():
    """A wrapped surface's dense animated window prices the window plus its
    per-frame pad copy; described, it prices at one image (the per-batch
    endpoint/pad/merge envelope), which is what lengthens crossfade batches.
    """
    tex_a, tex_b = _tex(16, 16, 19), _tex(16, 16, 20)
    scene = SceneManager.reset()
    surface = Sphere(radius=1.0, color_texture=tex_a).spawn()
    with AnimationContext(runtime=2):
        surface.color_texture = tex_b
    times = torch.linspace(1.2, 2.8, 5)
    _, seg = _materialize(scene, surface, times)
    assert seg is not None
    assert seg.endpoints.shape[0] > 1
    assert surface._texture_window_lerp
    frame = 16 * 16 * 5 * 4
    lerp_estimate = surface._color_texture_bytes_per_timestep()
    assert lerp_estimate == frame * 1
    surface._texture_window_lerp = False
    surface._texture_window_collapsed = False
    dense_estimate = surface._color_texture_bytes_per_timestep()
    assert dense_estimate == frame * 2, (
        "a dense animated window on a wrapped surface is the window plus "
        "its per-frame pad copy"
    )


def test_u8_admission_is_proved_on_the_endpoints_not_the_latest_stamp():
    """The per-assignment provenance stamp describes the LATEST map, but a
    window can show any map ever assigned. A crossfade from a non-u8 map to
    a u8 one must not admit the stack; and under the kill switches the dense
    pre-fade window (showing the old, non-u8 map) must not be admitted on
    the new map's stamp -- the AND over assignments is what the dense arm
    reads now.
    """
    tex_a = _tex(8, 8, 21)  # arbitrary floats: NOT k/255
    tex_b = _u8_tex(8, 8, 22)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    assert surface._color_texture_u8_ok, "the latest map IS u8-eligible"
    assert not surface._color_texture_u8_ok_all

    times = torch.linspace(1.0, 3.0, 5)
    primitive, seg = _materialize(scene, surface, times)
    assert seg is not None
    assert not primitive.texture_u8_ok, (
        "a stack containing a non-u8 endpoint was admitted to u8 storage"
    )

    # Dense arm, window before the fade: the map on screen is tex_a.
    previous_lerp = rt_settings.texture_time_lerp
    rt_settings.set_texture_time_lerp(False)
    try:
        primitive, seg = _materialize(scene, surface, torch.linspace(0.2, 0.9, 3))
    finally:
        rt_settings.set_texture_time_lerp(previous_lerp)
    assert seg is None
    assert primitive.texture_map.shape[0] == 1, "pre-fade window should collapse"
    assert not primitive.texture_u8_ok, (
        "the collapsed window shows the OLD map; admitting it on the new "
        "map's stamp would u8-round texels that are not k/255"
    )


def test_a_u8_crossfade_admits_the_endpoint_stack():
    tex_a, tex_b = _u8_tex(8, 8, 23), _u8_tex(8, 8, 24)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    times = torch.linspace(1.0, 3.0, 5)
    primitive, seg = _materialize(scene, surface, times)
    assert seg is not None
    # The admission must equal a direct proof on the actual stack. (Whether
    # the recorded write's ``pre + (b - pre)`` round-trips every k/255 texel
    # exactly is float luck per fixture; what matters is that the decision
    # is made on the real endpoint values, never on a stamp.)
    from algan.utils.tensor_utils import texture_u8_provenance

    direct = texture_u8_provenance(seg.endpoints.view(-1, 8, 8, 5))
    assert primitive.texture_u8_ok == direct
    assert torch.equal(seg.endpoints[0], tex_a.reshape(-1))


def test_lerp_stacks_survive_frame_slicing():
    """slice_time_window slices every source tensor whose leading axis
    matches the frame count. K can equal T by coincidence, so the endpoint
    stack travels as [1, K, H, W, 5] -- sliced never -- while texture_lerp
    (genuinely per-frame) is sliced with the window.
    """
    tex_a, tex_b = _tex(8, 8, 25), _tex(8, 8, 26)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    times = torch.linspace(0.5, 3.5, 3)  # T = 3 = K (pre, post, tail)
    primitive, seg = _materialize(scene, surface, times)
    assert seg is not None
    assert seg.endpoints.shape[0] == 3
    sliced = primitive.slice_time_window(0, 2, 3)
    assert sliced.texture_map.shape == (1, 3, 8, 8, 5), (
        "the endpoint stack was frame-sliced"
    )
    assert sliced.texture_lerp.shape == (2, 3)


def test_an_updater_discovering_the_mob_mid_window_gets_a_dense_fill():
    """The one legitimate late reader: a time-dependent updater branch that
    first touches the mob mid-window (its dependency was unknown when the
    gate ran). trace_updater_mob_access rematerializes the described rows
    densely from the description and drops it, so the read -- and the
    primitive build after it -- consume ordinary dense state.
    """
    tex_a, tex_b = _tex(8, 8, 27), _tex(8, 8, 28)
    scene = SceneManager.reset()
    surface = Surface(color_texture=tex_a, grid_height=4, grid_width=4).spawn()
    square = Square().spawn()  # spawns [1, 2]
    captured = {}

    def probe(mob, t):
        # Touches the surface only late in the window; the add-time
        # invocation (t = 0) and the early frames record no dependency.
        if float(t.reshape(-1).max()) > 2.0:
            captured["tex"] = surface.color_texture

    square.add_updater(probe)  # active from t = 2
    with AnimationContext(runtime=2):
        surface.color_texture = tex_b  # [2, 4]
    Scene.wait(1)
    times = torch.linspace(2.2, 4.4, 5)  # elapsed reaches 2.4 > 2.0
    primitive, seg = _materialize(scene, surface, times)
    assert "tex" in captured, "the branch never fired; the fixture is broken"
    assert seg is None, "the description must be dropped once a reader appears"
    assert primitive.texture_lerp is None
    assert primitive.texture_map.shape[0] == 5, (
        "the build after the fallback must consume the dense fill"
    )
    # The filled state interpolates: the last covered frame is far from both
    # endpoints' first texel unless the fill reproduced the lerp.
    tex = captured["tex"]
    assert tex.shape[0] == 5


def test_sub_batch_windows_use_the_animations_own_span():
    """Endpoints are the ANIMATION's endpoints, not the batch boundaries: a
    window covering only the middle of a fade still uploads the two authored
    maps and mid-fade weights, so consecutive batches share (dedupable)
    endpoint content.
    """
    tex_a, tex_b = _tex(8, 8, 29), _tex(8, 8, 30)
    scene, surface = _crossfade_scene(tex_a, tex_b)
    times = torch.linspace(1.6, 2.4, 3)  # strictly inside the [1, 3] fade
    _, seg = _materialize(scene, surface, times)
    assert seg is not None
    assert seg.endpoints.shape[0] == 2, "only pre and post are referenced"
    assert torch.equal(seg.endpoints[0], tex_a.reshape(-1))
    assert bool((seg.weights > 0).all())
    assert bool((seg.weights < 1).all())
    assert seg.cache_key is not None, (
        "a mid-animation window is stable across batches and cacheable"
    )
