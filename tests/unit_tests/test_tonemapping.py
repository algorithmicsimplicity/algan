"""Guards on what the tonemap does to a colour.

These call the post stage directly rather than rendering a scene: the tonemap
is the last thing to touch a frame, so handing ``_finalize_on_device`` a
synthetic linear-HDR buffer exercises exactly the code under test and nothing
else. ``benchmarks/_tonemap_transfer_probe.py`` is the same technique with a
wider ramp and a printed table.

Both shipping implementations are covered wherever it is cheap to do so -- the
standalone Taichi kernel (``post_tonemap_kernel=True``, the default) and the
torch pipeline it replaced -- because the defects these guard against were
present in *both*, which is precisely why an agreement-between-implementations
check did not catch them. See ``TONEMAP_FINDINGS.md``.

Deliberately not marked ``fast``: nothing outside this module can break them.
"""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.post_processing.post_process import _finalize_on_device
from algan.rendering.raytracing import settings as rt_settings
from algan.settings._startup import render_device
from algan.utils.memory_utils import ManualMemory

# Both implementations, as (id, use_taichi_kernel).
IMPLEMENTATIONS = [("kernel", True), ("torch", False)]

# Several guards below run in both colour working spaces, because the post
# stage's job differs between them: under the linear space it applies the sRGB
# OETF at the byte write, so it is an *encoder*, and under the display-referred
# space it is a passthrough. The statements they make about a colour surviving
# the pipeline stay true in both -- they are just written in different spaces.
#
# Safe to flip in-process, unlike the shading kernels: the post stage takes
# ``linear_color`` as a ``ti.template()`` argument, so Taichi specialises on it
# instead of baking it in at first compile.


def _srgb_to_linear(c):
    """The sRGB EOTF, from the specification.

    Written out rather than imported from ``algan.utils.color_space`` so these
    guards measure the renderer against the standard instead of against its own
    transcription of it -- the lesson of the AgX matrix that was transposed in
    both implementations and agreed with itself perfectly.
    """
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c):
    """The sRGB OETF, from the specification."""
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _authored(byte_value, linear):
    """The value the post stage receives for an authored byte.

    Upstream of the post stage the renderer has already decoded authored colour
    into the working space, so under the linear space the post stage is handed
    linear light and hands back the byte it started from.
    """
    v = byte_value / 255.0
    return _srgb_to_linear(v) if linear else v


def _expect_byte(working_value, linear):
    """The byte a working-space value should land on, with no curve."""
    v = min(max(working_value, 0.0), 1.0)
    return round(255 * (_linear_to_srgb(v) if linear else v))


@pytest.fixture
def color_space(request):
    """Select the working space for one test, restoring it afterwards."""
    enabled = request.param
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(enabled)
    try:
        yield enabled
    finally:
        rt_settings.set_linear_color_space(previous)


def encode(values, *, tonemapping, exposure=1.0, method="neutral", kernel=True):
    """Push a list of neutral linear-HDR values through the post stage.

    Returns one ``(r, g, b)`` tuple of uint8 per input value -- what that value
    would land on in the encoded frame.
    """
    frame = torch.zeros(
        (1, 1, len(values), 4), dtype=torch.float32, device=render_device()
    )
    for i, v in enumerate(values):
        frame[0, 0, i, :3] = v

    memory = ManualMemory(0.0, device=render_device(), managed=False, num_bytes=1 << 22)
    was_kernel = rt_settings.post_tonemap_kernel
    rt_settings.set_post_tonemap_kernel(kernel)
    try:
        out = _finalize_on_device(
            frame,
            4,
            memory,
            tonemap_enabled=True,
            tonemapping=tonemapping,
            tonemap_method=method,
            exposure=exposure,
        )
        return [
            tuple(int(x) for x in out[0, 0, i, :3].tolist()) for i in range(len(values))
        ]
    finally:
        rt_settings.set_post_tonemap_kernel(was_kernel)


def test_tonemapping_is_off_by_default():
    # An authored colour should land on the pixel it names. The curve cannot do
    # that -- it is the identity nowhere except at 0 -- so the default is off.
    assert SETTINGS.raytracing.tonemapping is False


@pytest.mark.parametrize(
    "color_space", [True, False], indirect=True, ids=["linear", "display"]
)
@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_off_is_exact_passthrough_of_authored_bytes(impl, kernel, color_space):
    """An authored colour lands on the pixel it names.

    The intent is unchanged by the working space; only the value handed to the
    post stage is. Under the linear space the renderer decodes authored colour
    upstream, so the post stage receives linear light and its OETF returns the
    authored byte -- decode-then-encode with no arithmetic between is the
    identity, which is the whole reason flat 2-D content is unaffected by the
    linear working space.
    """
    authored = [0, 1, 17, 64, 128, 191, 254, 255]
    got = encode(
        [_authored(k, color_space) for k in authored],
        tonemapping=False,
        kernel=kernel,
    )
    assert [c[0] for c in got] == authored


@pytest.mark.parametrize(
    "color_space", [True, False], indirect=True, ids=["linear", "display"]
)
@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_exposure_applies_with_tonemapping_off(impl, kernel, color_space):
    # Regression: exposure used to be dropped on the floor whenever the curve
    # was off, so every exposure produced identical bytes. It is the documented
    # brightness control and -- with tonemapping off by default -- the only one.
    #
    # Expectations are computed from the rule rather than tabulated, so the
    # test states what the post stage owes (multiply in the working space, then
    # encode) instead of a set of bytes that has to be re-derived per space.
    values = [0.25, 0.5, 0.75]
    at_half = encode(values, tonemapping=False, exposure=0.5, kernel=kernel)
    at_one = encode(values, tonemapping=False, exposure=1.0, kernel=kernel)
    at_two = encode(values, tonemapping=False, exposure=2.0, kernel=kernel)

    for exposure, got in ((1.0, at_one), (0.5, at_half), (2.0, at_two)):
        want = [_expect_byte(v * exposure, color_space) for v in values]
        assert [c[0] for c in got] == pytest.approx(want, abs=1), f"exposure={exposure}"

    assert at_half != at_one
    assert at_two != at_one


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_exposure_of_one_is_exact(impl, kernel):
    # The default must move no pixel, or flipping the default would have
    # silently re-baselined every scene.
    values = [k / 255.0 for k in (0, 33, 128, 200, 255)]
    assert encode(values, tonemapping=False, exposure=1.0, kernel=kernel) == encode(
        values, tonemapping=False, kernel=kernel
    )


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_agx_maps_neutral_to_neutral(impl, kernel):
    """A tonemap may darken a grey; it may not tint one.

    This is the guard for the transposed Rec.2020 -> Rec.709 matrix, which gave
    every neutral a fixed +52% red / -56% green and rendered authored grey
    ``(128, 128, 128)`` as magenta ``(255, 77, 180)``. The invariant is that
    the two spaces share a white point, so the conversion maps white to white.
    """
    for value, out in zip(
        (0.25, 0.5, 0.75, 1.0),
        encode([0.25, 0.5, 0.75, 1.0], tonemapping=True, method="agx", kernel=kernel),
    ):
        assert max(out) - min(out) <= 1, f"agx tinted neutral {value}: {out}"


def test_agx_implementations_agree():
    """Useful, but on its own it proves nothing about correctness.

    This test passed throughout the period when both implementations carried
    the transposed matrix: two wrong implementations agree perfectly. It is
    here to catch one of them drifting from the other, and
    ``test_agx_maps_neutral_to_neutral`` above is what checks either is right.
    """
    values = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    taichi = encode(values, tonemapping=True, method="agx", kernel=True)
    torch_ = encode(values, tonemapping=True, method="agx", kernel=False)
    assert taichi == torch_


def test_neutral_curve_transfer_is_unchanged():
    """Pin the shipped neutral curve, so a change to it has to be deliberate.

    These are Khronos PBR Neutral's own numbers. They are recorded here not
    because they are desirable -- ``1.0 -> 222`` is why the default is off --
    but so that editing the curve shows up as an edit to this list.
    """
    values = [0.0, 0.08, 0.25, 0.5, 0.76, 1.0, 2.0, 4.0]
    expected = [0, 10, 54, 117, 184, 222, 245, 251]
    previous = rt_settings.linear_color_space
    # Pinned in the display-referred space, where the byte *is* the curve's
    # output: that keeps this a pin on the curve itself rather than on the
    # curve composed with the OETF. What the linear space does to these same
    # numbers is the subject of the test below.
    rt_settings.set_linear_color_space(False)
    try:
        for kernel in (True, False):
            got = [c[0] for c in encode(values, tonemapping=True, kernel=kernel)]
            assert got == expected, f"kernel={kernel}"
    finally:
        rt_settings.set_linear_color_space(previous)


def test_neutral_curve_is_encoded_after_the_curve_under_the_linear_space():
    """The OETF runs last, after the curve -- three.js's order.

    Composing the pinned curve outputs above with the sRGB OETF must reproduce
    what the linear arm actually emits. That is what makes this an ordering
    check: applying the transfer function *before* the curve, or in place of
    it, would not land on these bytes.
    """
    values = [0.0, 0.08, 0.25, 0.5, 0.76, 1.0, 2.0, 4.0]
    curve_out = [0, 10, 54, 117, 184, 222, 245, 251]
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(True)
    try:
        got = [c[0] for c in encode(values, tonemapping=True)]
    finally:
        rt_settings.set_linear_color_space(previous)

    # +-1 because the pinned column is already quantised, so re-deriving the
    # curve's output from it carries up to half a byte of rounding into the
    # (non-linear) encode.
    want = [round(255 * _linear_to_srgb(c / 255.0)) for c in curve_out]
    assert got == pytest.approx(want, abs=1)


def test_curve_moves_values_that_were_already_in_range():
    """The finding that prompted the default flip, kept as a live measurement.

    Everything here is inside the display range before the curve runs, so an
    ideal HDR-only tonemap would leave it alone. None of it survives untouched
    except black -- which is why "only tonemap HDR values" is not something the
    curve can be tuned into doing.
    """
    values = [0.08, 0.25, 0.5, 0.75, 1.0]
    # Compared within one working space, so this measures the curve rather
    # than the difference between the two spaces.
    white_cost = {True: 15, False: 33}
    previous = rt_settings.linear_color_space
    try:
        for linear in (True, False):
            rt_settings.set_linear_color_space(linear)
            on = [c[0] for c in encode(values, tonemapping=True)]
            off = [c[0] for c in encode(values, tonemapping=False)]
            assert all(a < b for a, b in zip(on, off)), (linear, on, off)
            # White is the worst of them either way. The linear space does not
            # rescue the curve, it only changes the size of the bill: an
            # authored white still cannot render 255 with the curve on, because
            # Khronos Neutral reserves headroom by mapping linear 1.0 to 0.869
            # and the OETF cannot put that back.
            assert off[-1] - on[-1] == white_cost[linear], linear
    finally:
        rt_settings.set_linear_color_space(previous)


def _lit_rgb(values):
    """Run ``values`` through the torch shaders' output bound."""
    from algan.rendering.shaders.material_shaders import _recombine

    rgb = torch.tensor(values, dtype=torch.float32).reshape(-1, 3)
    glow = torch.zeros((rgb.shape[0], 1), dtype=torch.float32)
    return _recombine(rgb, glow)[..., :3]


def test_lit_colour_below_one_is_untouched():
    # The bound must be the identity in range, or enabling it would have
    # re-baselined every scene rather than only the pixels that were clipping.
    values = [[0.0, 0.0, 0.0], [0.25, 0.5, 0.75], [1.0, 1.0, 1.0], [1.0, 0.0, 0.5]]
    out = _lit_rgb(values)
    assert torch.equal(out, torch.tensor(values, dtype=torch.float32))


def test_over_range_lit_colour_keeps_its_hue():
    """A clamp truncates channels independently and slides colour to white.

    Scaling by the peak is what keeps a lit orange face orange instead of
    turning it flat yellow-white. The guard is the ratio between channels.

    Display-referred only. Under the linear working space the bound is off by
    design -- see the test below.
    """
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(False)
    try:
        out = _lit_rgb([[2.0, 1.0, 0.4]])[0]
    finally:
        rt_settings.set_linear_color_space(previous)
    assert float(out.max()) == pytest.approx(1.0)
    # Ratios preserved: 2.0 : 1.0 : 0.4 -> 1.0 : 0.5 : 0.2
    assert out.tolist() == pytest.approx([1.0, 0.5, 0.2])
    # What the old per-channel clamp would have produced, for contrast.
    assert out.tolist() != pytest.approx([1.0, 1.0, 0.4])


def test_linear_space_does_not_bound_the_lit_colour():
    """Under the linear space the shader hands over its radiance untouched.

    The peak bound exists to keep an over-range colour's hue when the encoder
    is a bare clamp. In linear light the range belongs to the display
    transform at the end of the pipeline -- the tonemap, or the clamp the OETF
    is applied through -- and bounding here instead would make lights stop
    adding, which is the thing the working space exists to fix. Values above
    1.0 pass through as radiance.
    """
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(True)
    try:
        out = _lit_rgb([[2.0, 1.0, 0.4]])[0]
        floored = _lit_rgb([[-0.5, 0.25, 0.5]])[0]
    finally:
        rt_settings.set_linear_color_space(previous)
    assert out.tolist() == pytest.approx([2.0, 1.0, 0.4])
    # The negative floor is *not* part of the bound and must survive: it is
    # what stops a negative reaching the encoder's pow, which would be NaN.
    assert floored.tolist() == pytest.approx([0.0, 0.25, 0.5])


def test_negative_lit_colour_is_floored_not_flipped():
    assert _lit_rgb([[-0.5, 0.25, 0.5]])[0].tolist() == pytest.approx([0.0, 0.25, 0.5])


def test_black_does_not_divide_by_zero():
    out = _lit_rgb([[0.0, 0.0, 0.0]])[0]
    assert torch.isfinite(out).all()
    assert out.tolist() == [0.0, 0.0, 0.0]


def _lambert(albedo=1.0, light_intensity=1.0, ambient=1.0):
    """One white light head-on at ``n.l == 1``, through the torch Lambert shader."""
    from algan.rendering.shaders.material_shaders import lambert_shader

    t = torch.tensor
    return lambert_shader(
        None,
        t([[0.0, 0.0, 0.0]]),  # vertex at the origin
        t([[0.0, 0.0, 1.0]]),  # normal along +z
        t([[albedo, albedo, albedo, 0.0]]),
        t([[0.0, 0.0, 5.0]]),  # camera
        t([[0.0, 0.0, 1.0]]),  # light directly above -> n.l == 1
        t([[1.0, 1.0, 1.0, 1.0]]),
        light_intensity,
        ambient,
        emissive=torch.zeros(1, 3),
        emissive_intensity=1.0,
    )[0, :3]


def test_fully_lit_surface_reflects_its_albedo_and_no_more():
    """Energy conservation: reflected <= incident.

    The ambient fill used to be added on top of a full direct term, so a
    fully lit surface reflected ``albedo * 1.1`` -- more light than arrived.
    A mid-grey now comes back as exactly its albedo. (White cannot show this:
    1.1 and 1.0 both bound to 1.0, which is why the test uses 0.5.)

    Display-referred only. Normalising the illumination budget is what makes
    lights stop adding, and the linear working space removes it deliberately --
    see the test below.
    """
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(False)
    try:
        assert _lambert(albedo=0.5).tolist() == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)
    finally:
        rt_settings.set_linear_color_space(previous)


def test_linear_space_lets_light_accumulate():
    """Lights add, which is the point of the working space.

    In linear light two lamps really do deliver twice the radiance, and the
    display transform at the end of the pipeline is what decides where that
    lands on a pixel. The budget normalisation made the surface reflect its
    albedo and no more however much light arrived, so a second lamp changed
    nothing -- measured on a real render, totals of 1.2, 1.5 and 1.8 all
    produced the same byte.

    Ambient sits on top of the direct term here rather than sharing a budget
    with it, so a fully lit mid-grey reflects more than its albedo. That is
    over-exposure, and it is the author's to fix with intensities or exposure,
    not something to normalise away behind their back.
    """
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(True)
    try:
        lit = float(_lambert(albedo=0.5)[0])
        brighter = float(_lambert(albedo=0.5, light_intensity=2.0)[0])
    finally:
        rt_settings.set_linear_color_space(previous)

    assert lit > 0.5, "ambient must not be absorbed into a shared budget"
    assert brighter > lit, "more light must make the surface brighter"


def test_under_lit_surface_is_not_scaled():
    """Below unit incident light the budget is inert, so dim rigs are untouched.

    Half a light's worth of illumination on a mid-grey: 0.5*0.1 ambient plus
    0.5*0.5 direct. Normalising here would darken a scene that was never over
    range in the first place.

    Display-referred: the budget only exists there, and so does the 0.1 ambient
    coefficient this arithmetic assumes -- in linear light the same fill is
    0.01, because the units changed (ambient_strength_linear).
    """
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(False)
    try:
        assert _lambert(albedo=0.5, light_intensity=0.5).tolist() == pytest.approx(
            [0.3, 0.3, 0.3], abs=1e-6
        )
    finally:
        rt_settings.set_linear_color_space(previous)


def test_budget_counts_radiance_not_light_count():
    """Three lights at 1/3 intensity must cost the same budget as one at 1.

    Weighting the budget by geometry alone penalised a rig for how many lights
    it used rather than how much light it emitted, which visibly over-darkened
    dim multi-light scenes.

    Display-referred, for the same two reasons as the test above.
    """
    assert rt_settings.ambient_strength == 0.1
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(False)
    try:
        one_bright = _lambert(albedo=0.5, light_intensity=0.9)
        # Same total emitted radiance, split three ways, is the same
        # illumination.
        assert float(one_bright.max()) == pytest.approx(0.5 * (0.1 + 0.9), abs=1e-6)
    finally:
        rt_settings.set_linear_color_space(previous)


def test_energy_scale_is_identity_below_unity():
    """Display-referred: below unity the budget divisor changes nothing.

    Above it the divisor is what stopped gamma-space light sums running away.
    """
    from algan.rendering.shaders.material_shaders import _energy_scale

    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(False)
    try:
        for w in (0.0, 0.25, 0.5, 1.0):
            assert float(_energy_scale(torch.tensor([w]))) == pytest.approx(1.0)
        assert float(_energy_scale(torch.tensor([2.0]))) == pytest.approx(0.5)
        assert float(_energy_scale(torch.tensor([4.0]))) == pytest.approx(0.25)
    finally:
        rt_settings.set_linear_color_space(previous)


def test_energy_scale_is_identity_everywhere_under_the_linear_space():
    """No budget at all in linear light -- the divisor is 1.0 for any weight.

    This is the mechanism behind ``test_linear_space_lets_light_accumulate``:
    the scale being unconditionally 1.0 is what makes N lights deliver N
    lights' worth instead of one rig's worth.
    """
    from algan.rendering.shaders.material_shaders import _energy_scale

    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(True)
    try:
        for w in (0.0, 0.25, 1.0, 2.0, 4.0, 100.0):
            assert float(_energy_scale(torch.tensor([w]))) == pytest.approx(1.0), w
    finally:
        rt_settings.set_linear_color_space(previous)
