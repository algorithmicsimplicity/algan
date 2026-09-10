"""Settings the kernels fold in at compile time still take effect.

A handful of renderer settings do not reach a kernel as an argument: they sit
behind a ``ti.static`` gate, so tracing the kernel turns the value into code.
``_energy_scale``'s illumination budget, ``_run_frag_pipeline``'s peak bound,
``_stage_manim``'s encode/decode pair and the ambient fill are all chosen that
way by ``linear_color_space``, and the two triangle/shadow switches in
``raytrace_kernels_taichi`` the same way.

Nothing keys a specialization on them. That is the hazard: a specialization is
cached under its ``ti.template()`` arguments, and these are not among them, so
a second render that changed one *reuses the first render's kernels* and gets
the first render's picture with no error and no warning. It cost six counts of
255 on an authored floor -- a render came out darker only because a scene
earlier in the process had compiled the shared specialization under the other
working space.

``rt_settings.KERNEL_COMPILED_IN_FIELDS`` is the list of those settings and
``taichi_runtime`` rebuilds the program when one of them moves between render
jobs. These tests are the two halves of that: the rebuild actually happens and
actually re-bakes, and the list has not drifted from the gates in the kernels.
"""

# No ``from __future__ import annotations`` here: the probe below is a real
# ``@ti.kernel`` and the compiler needs its annotation as the runtime object
# ``ti.types.ndarray()``, not as a string (``pyproject.toml`` exempts
# ``tests/*`` from the required-import rule for exactly this).

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from algan import (
    BLACK,
    OUT,
    WHITE,
    MeshLambertMaterial,
    Off,
    PointLight,
    Prism,
    Scene,
    SceneManager,
    VideoSettings,
)
from algan.rendering.raytracing.settings import KERNEL_COMPILED_IN_FIELDS
from algan.settings import SETTINGS

_REPOSITORY_ROOT = Path(__file__).parents[2]

#: Small: the claim is "the same bytes twice", and the kernels are the same
#: kernels at any resolution.
_SETTINGS = VideoSettings((48, 48), 30, supersampling=1)


def render_probe_frame(path):
    """One deterministic frame of a lit slab, as an (H, W, 3) int array.

    Lit rather than unlit on purpose: the gates in the list are all on the way
    to a *shaded* pixel and none of them touches flat unlit content.

    Module scope rather than a local helper because the reference arm renders
    it from its own process (:data:`_REFERENCE_PROBE`), and the two arms have
    to be the same scene at the same settings down to the pixel.
    """
    SceneManager.instance().reset()
    Scene.set_background(BLACK)
    with Off():
        Scene.clear_lights()
        PointLight(location=OUT * 4.0, color=WHITE, intensity=1.0).spawn(animate=False)
        slab = Prism(width=3.0, height=3.0, depth=0.2).set_material(
            MeshLambertMaterial(color=(0.8, 0.4, 0.2))
        )
        slab.spawn(animate=False)
    Scene.save_frame(str(path), _SETTINGS)
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert frame is not None, f"the probe render produced no file at {path}"
    return frame[..., 2::-1].astype(np.int32)


#: The display-referred arm rendered by a process that has compiled nothing
#: else -- the only reference this defect can be measured against.
#:
#: Within one process there is nothing to compare to. The kernels bake the
#: working space of whichever render reaches them first and (without the fix)
#: never change again, so every frame that process produces is consistent with
#: every other one; "the same twice" is exactly what the bug delivers. What is
#: wrong is the frame itself, and only a process that baked the other way can
#: say so.
_REFERENCE_PROBE = """
import sys

sys.path.insert(0, sys.argv[2])

from algan.settings import SETTINGS

import test_compiled_in_settings as probe

SETTINGS.raytracing.set(linear_color_space=False)
probe.render_probe_frame(sys.argv[1])
print("reference frame written")
"""


def _reference_frame(tmp_path):
    """``render_probe_frame`` display-referred, as a first render in a process."""
    out = tmp_path / "reference.png"
    environment = dict(os.environ)
    # A warm daemon is another process that has already rendered something,
    # which is the very condition under test.
    environment["ALGAN_USE_DAEMON"] = "0"
    environment["ALGAN_AUTO_DAEMON"] = "0"
    # The compiler's inspector needs real source, so a file rather than -c
    # (tests/unit_tests/test_rgb_shadow_payload.py hits the same wall).
    with tempfile.NamedTemporaryFile(
        "w", suffix="_working_space_probe.py", delete=False, dir=tmp_path
    ) as handle:
        handle.write(_REFERENCE_PROBE)
        script = handle.name
    result = subprocess.run(
        [sys.executable, script, str(out), str(Path(__file__).parent)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    frame = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert frame is not None, f"the reference render produced no file at {out}"
    return frame[..., 2::-1].astype(np.int32)


def test_a_render_does_not_inherit_the_working_space_of_the_one_before_it(tmp_path):
    """The user-visible claim: a frame is what its own settings say it is.

    Render the linear arm first, so the kernels bake it, then the
    display-referred arm -- and require the second frame to be the frame a
    process that had rendered nothing else produces. Before the rebuild it was
    the linear arm's shading wearing the display-referred pipeline's host side:
    a slab six counts of 255 out, with nothing said about it.
    """
    reference = _reference_frame(tmp_path)
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.set(linear_color_space=True)
        linear = render_probe_frame(tmp_path / "linear.png")
        SETTINGS.raytracing.set(linear_color_space=False)
        after = render_probe_frame(tmp_path / "after.png")
    finally:
        SceneManager.instance().reset()
        SETTINGS.restore(snapshot)

    assert np.abs(reference - linear).max() > 2, (
        "the two working spaces rendered this slab identically, so it cannot "
        "see the gates and the comparison below proves nothing"
    )
    assert np.array_equal(reference, after), (
        "the display-referred render came out up to "
        f"{int(np.abs(reference - after).max())} of 255 away from the same "
        "render in a fresh process, because the linear render before it left "
        "its own working space compiled into the kernels"
    )


def test_ensure_taichi_for_render_rebuilds_when_a_compiled_in_setting_moves():
    """The mechanism under the render test, without a render.

    Pinned separately because the render test cannot tell a rebuild from any
    other reason the two frames might agree, and because this is what a future
    entry in ``KERNEL_COMPILED_IN_FIELDS`` inherits.
    """
    from algan.rendering import taichi_runtime
    from algan.taichi_compat import kernel_specializations, program, submodule, ti

    taichi_runtime.ensure_taichi_for_render()

    @ti.kernel
    def _touch(out: ti.types.ndarray()):
        out[0] += 1.0

    _touch(torch.zeros(1, dtype=torch.float32))
    assert taichi_runtime._COMPILED_IN_SETTINGS is not None, (
        "materializing a kernel did not record what its gates folded in"
    )
    live = program()
    assert live is not None
    assert taichi_runtime.ensure_taichi_for_render() is False
    assert program() is live, "an unchanged setting must not cost a rebuild"

    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.set(
            linear_color_space=not SETTINGS.raytracing.linear_color_space
        )
        assert taichi_runtime.ensure_taichi_for_render() is True
        assert program() is not live
        runtime = submodule("lang.impl").get_runtime()
        assert all(not kernel_specializations(kernel) for kernel in runtime.kernels), (
            "the rebuild left a specialization alive, so it still bakes the "
            "value the render moved away from"
        )
    finally:
        SETTINGS.restore(snapshot)
        taichi_runtime.ensure_taichi_for_render()


# ---------------------------------------------------------------------------
# The list of record has not drifted from the gates
# ---------------------------------------------------------------------------

#: Roots a settings read can start from. Three are the storage module under
#: the names ``algan/rendering`` binds it to; ``SETTINGS`` is the facade, whose
#: sections are attributes on the way to the field, so walking to the OUTERMOST
#: attribute reads both ``rt_settings.x`` and ``SETTINGS.raytracing.x`` with
#: one rule (a chain that stops at a section yields ``raytracing``, which is
#: not a field name and drops out).
_SETTINGS_ROOTS = frozenset({"rt_settings", "settings", "_rts", "SETTINGS"})


def _kernel_sources():
    for path in sorted((_REPOSITORY_ROOT / "algan" / "rendering").rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def _functions_by_name(tree):
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(node):
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def _settings_root(node):
    """The ``Name`` an attribute chain starts from, or ``None``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _settings_reads(node, functions, seen):
    """Every settings attribute ``node`` reads, following local calls."""
    names = set()
    for attribute in ast.walk(node):
        # Section names come through too (``SETTINGS.raytracing.x`` yields
        # both ``raytracing`` and ``x``). Harmless: the caller keeps only the
        # names that are fields, and a section is not one.
        if isinstance(attribute, ast.Attribute) and (
            _settings_root(attribute) in _SETTINGS_ROOTS
        ):
            names.add(attribute.attr)
    for called in _called_names(node):
        callee = functions.get(called)
        if callee is None or called in seen:
            continue
        seen.add(called)
        names |= _settings_reads(callee, functions, seen)
    return names


def _static_gate_reads(tree):
    """Settings read while a ``ti.static`` argument is evaluated."""
    functions = _functions_by_name(tree)
    names = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "static"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ti"
        ):
            continue
        for argument in node.args:
            names |= _settings_reads(argument, functions, set())
    return names


def test_every_static_gate_over_a_setting_is_declared():
    """A ``ti.static`` gate over a mutable setting must be in the list.

    A new gate is easy to add and impossible to see the consequence of: it
    compiles, it renders, and it is wrong only in the second render of a
    process that changed the setting. So the gates are read out of the source
    rather than trusted to a reviewer.

    Import-frozen settings are exempt -- ``max_shadow_lights`` and friends are
    refused by ``SETTINGS.raytracing.set`` outright, so no write can reach a
    kernel that already baked one.
    """
    from algan.settings.raytracing_settings import (
        _IMPORT_FROZEN_FIELDS,
        _field_names,
    )

    fields = _field_names()
    undeclared = {}
    for path, tree in _kernel_sources():
        for name in _static_gate_reads(tree):
            if name not in fields or name in _IMPORT_FROZEN_FIELDS:
                continue
            if name in KERNEL_COMPILED_IN_FIELDS:
                continue
            undeclared.setdefault(name, set()).add(
                str(path.relative_to(_REPOSITORY_ROOT))
            )
    assert not undeclared, (
        "these settings are folded into kernels behind ti.static but are not "
        "in rt_settings.KERNEL_COMPILED_IN_FIELDS, so a render that changes "
        "one reuses kernels baking the old value: "
        + ", ".join(
            f"{name} ({', '.join(sorted(where))})"
            for name, where in sorted(undeclared.items())
        )
    )


@pytest.mark.parametrize("field", KERNEL_COMPILED_IN_FIELDS)
def test_every_declared_field_is_a_writable_setting(field):
    """The other direction: a name here that no longer exists, or that became
    import-frozen, is a stale entry costing every render a comparison and
    every reader a wrong explanation.
    """
    from algan.settings.raytracing_settings import (
        _IMPORT_FROZEN_FIELDS,
        _field_names,
    )

    assert field in _field_names(), f"{field} is not a renderer setting"
    assert field not in _IMPORT_FROZEN_FIELDS, (
        f"{field} cannot be written after import, so nothing can move it "
        "between renders and it does not belong in this list"
    )
