"""The in-sampler texture opacity multiply and u8 colour-map storage.

texture_opacity_in_kernel moves the mob-opacity premultiply off the colour
map: the primitive carries per-frame scalars (``texture_opacity``) and the
map keeps its authored texels, so a fade of a static image collapses to a
one-frame map (the premultiply used to weld the fade into the widest
attribute in the engine, voiding texture_window_collapse). texture_u8_storage
then stores maps whose texels are provably ``k / 255`` (``texture_u8_ok``,
checked once at authoring) as RGBA bytes bit-packed into f32 lanes of the
shared bank, decoded in-kernel through a per-map LUT scattered from the
map's own direct decode -- the f32 arm's own bits, to within the one-ulp
SIMD-tail residue test_u8_lane_packing_round_trips documents.

The pixel-compared render suites cannot see most of this machinery
(``tests/fast/scene.py`` has no textured geometry); these tests pin the
host-side contracts, and ``benchmarks/_texture_opacity_ab.py`` is the
frame-level acceptance harness.
"""

import math

import torch

from algan import Scene, Surface
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.mobs.image_mob import ImageMob
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive
from algan.scene_manager import SceneManager
from algan.utils.color_space import srgb_to_linear
from algan.utils.tensor_utils import texture_u8_provenance


def _u8_texture(width, height, seed=0):
    """A ``[W, H, 5]`` texture whose texels are exactly k/255, zero glow."""
    g = torch.Generator().manual_seed(seed)
    tex = torch.randint(0, 256, (width, height, 5), generator=g).float() / 255
    tex[..., 3] = 0.0
    return tex


def _materialize(scene, surface, times):
    timeline = scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(times)
        primitive = surface.get_render_primitives()
        # The map (and under the legacy arm its premultiplied copy) must be
        # detached from the buffers before they are cleared.
        primitive.texture_map = primitive.texture_map.clone()
        timeline.clear_buffers()
    return primitive


def test_u8_provenance_detection():
    assert texture_u8_provenance(_u8_texture(8, 8))
    # An arbitrary float texel is not k/255.
    bad = _u8_texture(8, 8)
    bad[0, 0, 0] = 0.5001
    assert not texture_u8_provenance(bad)
    # Glow is not representable in the packed RGBA lane.
    glowing = _u8_texture(8, 8)
    glowing[..., 3] = 1.0
    assert not texture_u8_provenance(glowing)
    # Out-of-range and NaN both fail the round trip.
    assert not texture_u8_provenance(torch.full((4, 4, 5), 2.0))
    assert not texture_u8_provenance(torch.full((4, 4, 5), float("nan")))


def test_setter_stamps_and_restamps_provenance():
    SceneManager.reset()
    surface = Surface(color_texture=_u8_texture(4, 4), grid_height=4, grid_width=4)
    assert surface._color_texture_u8_ok
    # Texel arithmetic goes back through the setter, which re-proves.
    surface.color_texture = surface.color_texture * 0.31
    assert not surface._color_texture_u8_ok


def test_fade_of_a_static_image_keeps_a_one_frame_map():
    scene = SceneManager.reset()
    surface = Surface(
        color_texture=_u8_texture(4, 4), grid_height=4, grid_width=4
    ).spawn()
    with Sync(run_time=1):
        surface.opacity = 0.0

    # The spawn's own fade-in occupies [0, 1] and the fade-out [1, 2]; sample
    # across both so the opacity trajectory spans (near) 1 down to 0.
    primitive = _materialize(scene, surface, torch.linspace(0.5, 2.0, 5))
    assert primitive.texture_map.shape[0] == 1, (
        f"a fade re-expanded the static map to {primitive.texture_map.shape[0]} "
        f"frames -- the premultiply is back"
    )
    op = primitive.texture_opacity
    assert op is not None
    assert op.numel() == 5
    assert float(op.amax()) > 0.5, "texture_opacity does not carry the fade"
    assert float(op[-1]) < 0.01, "texture_opacity does not carry the fade"
    assert surface._texture_window_collapsed, (
        "the batch sizer will keep pricing this fade per frame"
    )
    assert primitive.texture_u8_ok, "u8 provenance was dropped on the way through"
    # The map's own coverage is untouched (the fade rides the scalars).
    assert torch.equal(
        primitive.texture_map[0, ..., 4],
        _u8_texture(4, 4)[..., 4],
    )


def test_kill_switch_restores_the_premultiplied_map():
    previous_op = rt_settings.TEXTURE_OPACITY_IN_KERNEL
    rt_settings.set_texture_opacity_in_kernel(False)
    try:
        scene = SceneManager.reset()
        surface = Surface(
            color_texture=_u8_texture(4, 4), grid_height=4, grid_width=4
        ).spawn()
        with Sync(run_time=1):
            surface.opacity = 0.0

        primitive = _materialize(scene, surface, torch.linspace(0.5, 2.0, 5))
        assert primitive.texture_opacity is None
        assert primitive.texture_map.shape[0] == 5, (
            "the legacy arm must premultiply per frame (byte-identical restore)"
        )
        assert float(primitive.texture_map[-1, ..., 4].amax()) < 0.01
    finally:
        rt_settings.set_texture_opacity_in_kernel(previous_op)


def test_estimator_prices_a_fade_at_the_collapsed_window():
    scene = SceneManager.reset()
    surface = Surface(
        color_texture=_u8_texture(16, 16), grid_height=4, grid_width=4
    ).spawn()
    with Sync(run_time=1):
        surface.opacity = 0.0
    _materialize(scene, surface, torch.linspace(0.0, 1.0, 5))

    surface._memory_per_timestep_cache = None
    collapsed_estimate = surface._color_texture_bytes_per_timestep()
    # The legacy arm prices the same fade with the premultiply copy back and
    # the collapse voided; the flag is read off the previous build, so mimic
    # that build having run dense.
    surface._texture_window_collapsed = False
    previous_op = rt_settings.TEXTURE_OPACITY_IN_KERNEL
    rt_settings.set_texture_opacity_in_kernel(False)
    try:
        dense_estimate = surface._color_texture_bytes_per_timestep()
    finally:
        rt_settings.set_texture_opacity_in_kernel(previous_op)
    assert collapsed_estimate * 2 <= dense_estimate, (
        f"fade priced at {collapsed_estimate} vs {dense_estimate} -- windows "
        f"will not lengthen"
    )


def test_faded_out_frames_still_leave_the_bvh():
    """The premultiplied map's per-frame alpha used to gate visibility; with
    the opacity unbaked, _pack_frame_visibility must fold the scalars back in
    or a faded-out textured mob stays in every frame's BVH.
    """
    p = object.__new__(RayTracedTrianglePrimitive)
    tex = torch.zeros(1, 2, 2, 5)
    tex[..., 4] = 1.0
    p._rt_texture_map = tex
    p._rt_tex_opacity = torch.tensor([1.0, 0.0])
    # Corner colours fully transparent: the texture alone supplies coverage,
    # exactly the ImageMob sticker case.
    colors = torch.zeros(1, 1, 3, 5)
    lo = torch.zeros(2, 1, 3)
    hi = torch.ones(2, 1, 3)
    p._pack_frame_visibility(lo, hi, colors, "test")
    frame_visible = (p._rt_frame_hi >= p._rt_frame_lo).all(-1)
    assert bool(frame_visible[0, 0]), "opaque frame was culled"
    assert not bool(frame_visible[1, 0]), "faded-out frame stayed in the BVH"


def test_u8_lane_packing_round_trips():
    """The host packing the merge writes is exactly what the kernel's
    bit_cast + shifts read back: little-endian r|g<<8|b<<16|a<<24, one texel
    per f32 lane.
    """
    tex = _u8_texture(7, 5)
    q = (
        torch.round(tex[..., (0, 1, 2, 4)] * 255.0)
        .clamp_(0.0, 255.0)
        .to(torch.uint8)
        .reshape(-1, 4)
        .contiguous()
    )
    packed = q.view(torch.int32).view(torch.float32).reshape(-1)
    bits = packed.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    unpacked = torch.stack([(bits >> shift) & 0xFF for shift in (0, 8, 16, 24)], -1).to(
        torch.uint8
    )
    assert torch.equal(unpacked, q)
    # And the scattered LUT reproduces the f32 path's decode to within one
    # ulp, exactly for every byte with a single in-map bit pattern. Exact
    # everywhere is unattainable in principle on CPU: torch's SIMD body and
    # scalar tail can decode the SAME byte to bit patterns one ulp apart
    # inside one tensor (measured: byte 82 of a 105-element decode), so the
    # f32 arm itself stores two patterns for one byte and no table can match
    # both. The scatter copies the arm's own bits, which is the closest
    # achievable -- a table decoded from arange(256)/255 can differ on EVERY
    # byte instead.
    direct = srgb_to_linear(tex[..., :3].contiguous())
    bytes_rgb = torch.round(tex[..., :3] * 255.0).long()
    lut = torch.zeros(256)
    lut.scatter_(0, bytes_rgb.reshape(-1), direct.reshape(-1))
    via_lut = lut[bytes_rgb]
    ulp = (
        (direct.view(torch.int32).to(torch.int64) - via_lut.view(torch.int32))
        .abs()
        .max()
    )
    assert int(ulp) <= 1, f"LUT decode drifted {int(ulp)} ulp from the f32 arm"
    single_pattern = torch.tensor(
        [
            direct.reshape(-1)[bytes_rgb.reshape(-1) == byte].unique().numel() <= 1
            for byte in range(256)
        ]
    )
    exact = via_lut.view(torch.int32) == direct.view(torch.int32)
    assert bool(exact[single_pattern[bytes_rgb]].all()), (
        "LUT decode differs on a byte the f32 arm itself decodes consistently"
    )


def test_wf_textured_forces_the_legacy_premultiply():
    # The setter rejects enabling the removed legacy renderer outright, so
    # poke the module global the way only old pickled configs could.
    rt_settings.wf_textured = True
    try:
        assert not rt_settings.texture_opacity_in_kernel_active(), (
            "WF_TEXTURED's legacy bank builder consumes premultiplied maps"
        )
    finally:
        rt_settings.wf_textured = False


def test_u8_flip_is_byte_identical_end_to_end(tmp_path):
    """One small textured frame, rendered under both storage layouts."""
    import cv2

    def frame(u8, name):
        previous_u8 = rt_settings.TEXTURE_U8_STORAGE
        rt_settings.set_texture_u8_storage(u8)
        try:
            SceneManager.reset()
            ImageMob(_u8_texture(16, 16)[..., :4].contiguous()).spawn()
            result = Scene.save_frame(str(tmp_path / name), overwrite=True)
        finally:
            rt_settings.set_texture_u8_storage(previous_u8)
        assert result.rendered
        return torch.from_numpy(cv2.imread(str(result.output_path)))

    fa = frame(True, "u8_on.png")
    fb = frame(False, "u8_off.png")
    assert torch.equal(fa, fb), (
        f"u8 flip moved {int((fa != fb).sum())} channel values "
        f"(max {int((fa.int() - fb.int()).abs().max())})"
    )
