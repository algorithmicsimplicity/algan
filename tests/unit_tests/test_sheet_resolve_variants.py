"""Which ``sheet_resolve_shade`` variant each launch compiles.

The resolve is the renderer's largest kernel, and its cold compile is paid once
per distinct set of template values. Most of that cost is the five
continuation-tap fans, each unrolling ``sec_aa`` inlined copies of a jittered
surface sample, a normal fetch and a pool spawn. Two launches compiled them for
nothing (``findings/raster_bvh_verdicts.md`` S1):

* the shadow event pass (mode 1), which spawns nothing and reads nothing a fan
  writes after the ``break`` that follows it;
* the shading launches of a batch with no reflector, glass or pane, where every
  fan sits behind a branch that cannot be taken.

These tests pin both: the kernel's fans unroll ``fan_taps``, which is zero in
mode 1, and the host hands a non-reflective batch's shading launches
``sec_aa`` 1 while the event pass keeps the live count its ``event_dp``
footprint is sized and computed from. That the output is unchanged rests on
the branches being unreachable, and was checked by rendering before and after.

``vis_lights``, the light slots of the per-sheet visibility payload, is the
same kind of waste (S2): only mode 2 reads the payload, yet every launch
specialised on the batch's light bucket, so each bucket compiled its own
identical mode-0 and mode-1 resolves.

The same recorder shows which scenes reach the resolve at all, and with what:
glass with the refraction template and the nested-IOR stack, an environment
map through the background prefill, a transparent background like any other.
Nothing else in the CI or pre-PR groups took any of the three down the sheet
route, and a batch that falls off it renders with no anti-aliasing at
PREVIEW, silently (test_sheet_route_vetoes.py).

Not marked fast: a change elsewhere cannot break these without changing the
resolve or its launch.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from PIL import Image

from algan import (
    BLUE,
    GLASS,
    GREEN,
    LEFT,
    MIRROR,
    OUT,
    RED,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    Cube,
    DirectionalLight,
    MeshLambertMaterial,
    Off,
    PointLight,
    Scene,
    Sphere,
    Square,
    cosine_color,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import sheet_resolve_taichi
from algan.rendering.raytracing.shading_taichi import shadow_vis_slots
from algan.settings import SETTINGS

PARAMS = sheet_resolve_taichi._SHEET_RESOLVE_SHADE_PARAMS
PREFILL = inspect.signature(sheet_resolve_taichi.env_background_prefill)

#: An 8-bit equirectangular map, height by width, and the intensity it is set at.
ENV_MAP = torch.full((16, 32, 3), 128, dtype=torch.uint8)
ENV_INTENSITY = 0.75


def _fan_loops(tree):
    """Every ``for s in ti.static(range(X))`` loop whose body takes a jittered
    surface sample -- the continuation fans -- with its ``X``.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and ast.unparse(it.func) == "ti.static"
            and isinstance(it.args[0], ast.Call)
            and ast.unparse(it.args[0].func) == "range"
        ):
            continue
        calls = {ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)}
        if "_jittered_surface_sample" in calls:
            found.append(ast.unparse(it.args[0].args[0]))
    return found


def test_mode_1_unrolls_no_continuation_fan():
    """The event pass compiled all five fans at the full tap count.

    Each fan is gated ``mode != 1`` only around its spawns, and the frontend
    builds the rest regardless (``advanced_optimization`` is off), so a
    shadowed batch's event pass cost about as much to compile as its shading
    pass. The fans now unroll ``fan_taps``, a compile-time count that is the
    live ``sec_aa`` in modes 0 and 2 and zero in mode 1.
    """
    source = inspect.getsource(sheet_resolve_taichi.sheet_resolve_shade_arena)
    tree = ast.parse(source.lstrip())
    fans = _fan_loops(tree)
    assert len(fans) == 5, f"expected the five continuation fans, found {fans}"
    assert set(fans) == {"fan_taps"}, (
        f"a fan unrolls {sorted(set(fans) - {'fan_taps'})} instead of fan_taps, "
        "so the event pass compiles it"
    )
    assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and [ast.unparse(t) for t in n.targets] == ["fan_taps"]
    ]
    assert len(assigns) == 1
    call = assigns[0].value
    assert ast.unparse(call.func) == "ti.static"
    expr = compile(ast.Expression(call.args[0]), "<fan_taps>", "eval")
    for sec_aa in (1, 2, 4, 8):
        assert eval(expr, {}, {"sec_aa": sec_aa, "mode": 1}) == 0
        for mode in (0, 2):
            assert eval(expr, {}, {"sec_aa": sec_aa, "mode": mode}) == sec_aa


def _install_recorder(monkeypatch, launches, prefills):
    """Record each resolve launch's arguments by name instead of launching,
    and each environment prefill's.

    Both launch sites import their kernel from its module at call time, so
    patching the module attribute intercepts it. Not calling through keeps the
    test from compiling either kernel; the frame is then empty, which nothing
    here looks at.
    """

    def record(*args):
        launches.append(dict(zip(PARAMS, args)))

    def record_prefill(*args):
        prefills.append(PREFILL.bind(*args).arguments)

    monkeypatch.setattr(sheet_resolve_taichi, "sheet_resolve_shade", record)
    monkeypatch.setattr(sheet_resolve_taichi, "env_background_prefill", record_prefill)


def _render(tmp_path, name, build, setup, shadows):
    """Render one frame; return the image it wrote."""
    SETTINGS.raytracing.set(shadows=shadows)
    with Scene() as scene:
        if setup is not None:
            setup(scene)
        with Off():
            # Two lights, so the visibility payload's light bucket is not 1.
            DirectionalLight(location=RIGHT * 4 + UP * 5 + OUT * 4, color=WHITE).spawn(
                animate=False
            )
            PointLight(location=LEFT * 3 + UP * 2 + OUT * 3, color=WHITE).spawn(
                animate=False
            )
            for mob in build():
                mob.spawn(animate=False)
        result = scene.save_frame(str(tmp_path / name), video_settings=SMOKE_TEST)
    return Path(result.output_path)


def _plain():
    return [
        Cube(size=1.2).set_material(MeshLambertMaterial(color=GREEN)).move(LEFT),
        Square(size=1.0, color=BLUE).move(RIGHT),
    ]


def _mirror():
    return [
        Cube(size=1.2).set_material(MIRROR).move(LEFT),
        Cube(size=0.8).set_material(MeshLambertMaterial(color=RED)).move(RIGHT),
    ]


def _custom():
    """A user fragment pipeline, which is handed the visibility payload raw."""
    return [
        Cube(size=1.2).set_fragment_shader(cosine_color).move(LEFT),
        Square(size=1.0, color=BLUE).move(RIGHT),
    ]


def _glass():
    return [
        Sphere(radius=0.6).set_material(GLASS).move(LEFT),
        Cube(size=0.8).set_material(MeshLambertMaterial(color=RED)).move(RIGHT),
    ]


def _env_map(scene):
    scene.set_environment_map(ENV_MAP.clone(), intensity=ENV_INTENSITY)


def _transparent(scene):
    scene.set_background((0.1, 0.2, 0.3, 0.5))


#: (scene, its mobs, what it sets on the Scene, shadows)
RENDERS = (
    ("plain", _plain, None, False),
    ("plain", _plain, None, True),
    ("mirror", _mirror, None, False),
    ("mirror", _mirror, None, True),
    ("custom", _custom, None, False),
    ("glass", _glass, None, False),
    ("glass", _glass, None, True),
    ("env map", _plain, _env_map, False),
    ("env map", _plain, _env_map, True),
    ("transparent", _plain, _transparent, False),
)


class Render(NamedTuple):
    """What one render handed the resolve and the prefill, and what it wrote."""

    launches: dict
    prefills: list
    image: Path


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    """Each of the one-frame renders above, by scene and shadows.

    Module-scoped so the renders happen once. ``pytest.MonkeyPatch`` is used
    directly because the ``monkeypatch`` fixture is function-scoped.
    """
    tmp_path = tmp_path_factory.mktemp("resolve_variants")
    snapshot = SETTINGS.snapshot()
    seen = {}
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            for name, build, setup, shadows in RENDERS:
                got, prefilled = [], []
                _install_recorder(monkeypatch, got, prefilled)
                stem = f"{name.replace(' ', '_')}_{int(shadows)}"
                image = _render(tmp_path, stem, build, setup, shadows)
                by_mode = {int(a["mode"]): a for a in got}
                seen[name, shadows] = Render(by_mode, prefilled, image)
    finally:
        SETTINGS.restore(snapshot)
    return seen


@pytest.fixture(scope="module")
def launches(renders):
    """Each render's resolve launches, their arguments by name, by mode."""
    return {key: render.launches for key, render in renders.items()}


@pytest.fixture(scope="module")
def prefills(renders):
    """Each render's environment prefills, their arguments by name."""
    return {key: render.prefills for key, render in renders.items()}


def test_the_renders_reached_every_resolve_mode(launches):
    assert rt_settings.effective_analytic_aa_secondary_samples() > 1, (
        "continuation fans are off at this setting, so nothing here is tested"
    )
    for name, _build, _setup, shadows in RENDERS:
        want = {1, 2} if shadows else {0}
        assert set(launches[name, shadows]) == want, launches[name, shadows].keys()
    for got in launches.values():
        for launch in got.values():
            assert shadow_vis_slots(launch["num_lights"]) > 1, (
                "one light bucket: the vis_lights tests below compare nothing"
            )


def test_a_batch_with_nothing_to_reflect_compiles_its_shading_without_fans(
    launches,
):
    """A Lambert cube and a flat square: no fan's branch can be taken.

    The shading launches took the live tap count anyway, so the built-in 2-D
    variant compiled ~45 s where ~17 s does the same work. The event pass keeps
    the live count: its ``event_dp`` footprint is sized from it, and
    raster_shadow_trace reads it.
    """
    live = rt_settings.effective_analytic_aa_secondary_samples()
    assert launches["plain", False][0]["sec_aa"] == 1
    assert launches["plain", True][2]["sec_aa"] == 1
    assert launches["plain", True][1]["sec_aa"] == live


def test_a_batch_with_a_mirror_keeps_its_fans(launches):
    live = rt_settings.effective_analytic_aa_secondary_samples()
    assert launches["mirror", False][0]["sec_aa"] == live
    assert launches["mirror", True][1]["sec_aa"] == live
    assert launches["mirror", True][2]["sec_aa"] == live


def test_only_the_shading_pass_of_a_shadowed_batch_specialises_on_lights(
    launches,
):
    """Modes 0 and 1 never read the visibility payload, so they take one slot.

    Mode 0 shades with ``shadows`` 0, which compiles every built-in stage's
    read out (``_light_vis``); mode 1 shades nothing. Before, both specialised
    on the batch's light bucket, so a one-light and a two-light scene compiled
    two identical copies of each -- the built-in ``3d`` and ``3d-default-light``
    resolve specs differed in nothing else.
    """
    for name in ("plain", "mirror", "glass", "env map"):
        assert launches[name, False][0]["vis_lights"] == 1
        assert launches[name, True][1]["vis_lights"] == 1
        mode2 = launches[name, True][2]
        assert mode2["vis_lights"] == shadow_vis_slots(mode2["num_lights"])


def test_a_user_pipeline_keeps_its_light_slots_without_shadows(launches):
    """A custom fragment pipeline is handed ``vis`` itself and may index it
    by light whatever ``shadows`` says, so its batch keeps the bucket.
    """
    mode0 = launches["custom", False][0]
    assert mode0["frag_pipelines"], "the custom scene injected no pipeline"
    assert mode0["vis_lights"] == shadow_vis_slots(mode0["num_lights"])


def test_a_batch_with_glass_refracts_on_the_sheet_route(launches):
    """Glass takes the resolve, not the supersampled fallback, with the
    refraction template and the nested-IOR stack in every launch, and its
    shading launches keep their fans.
    """
    live = rt_settings.effective_analytic_aa_secondary_samples()
    stack = 1 if rt_settings.nested_ior_mode() != 0 else 0
    for shadows in (False, True):
        for mode, launch in launches["glass", shadows].items():
            assert launch["refraction"] == 1, mode
            assert launch["ior_stack"] == stack, mode
            if mode != 1:
                assert launch["sec_aa"] == live, mode


def test_an_environment_map_is_prefilled_as_the_background_sheet(launches, prefills):
    """Background-as-final-sheet (DESIGN_sheet_resolve.md §4.5): an
    env-mapped batch prefills the frame buffer with the map, once per chunk,
    and the composite multiplies each covered pixel's leftover weight by it.
    Empty pixels get no resolve launch at all, so without the prefill they
    show the flat background; and the prefill must read the map from the
    texture buffer the resolve reads, at the size and intensity it was set at.
    No other scene prefills.
    """
    for (name, shadows), calls in prefills.items():
        assert len(calls) == (1 if name == "env map" else 0), (name, shadows)
    height, width, _ = ENV_MAP.shape
    for shadows in (False, True):
        (call,) = prefills["env map", shadows]
        assert (call["env_w"], call["env_h"]) == (width, height)
        assert call["env_intensity"] == pytest.approx(ENV_INTENSITY)
        for launch in launches["env map", shadows].values():
            assert launch["textures"] is call["textures"]


def test_a_transparent_background_keeps_the_route(renders):
    """A transparent background composites from the sheets' leftover weight
    like an opaque one, so it takes the resolve -- as this render did, whose
    frame really was transparent.
    """
    render = renders["transparent", False]
    assert set(render.launches) == {0}
    with Image.open(render.image) as image:
        assert image.mode == "RGBA", "the background came out opaque"
