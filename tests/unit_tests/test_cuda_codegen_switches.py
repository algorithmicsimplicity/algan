"""The two CUDA codegen switches Quadrants patches 0005 and 0006 provide.

Both patches ship in the pinned ``algan-quadrants`` wheel and both stay off
unless asked for, so what these tests hold is mostly the *off* arm: nothing
about a default render's kernels, or the keys they are cached under, may move.

* ``SETTINGS.raytracing.experimental.megakernel_max_reg`` (0005, a per-kernel
  PTX ``.maxnreg``) accepts only what PTX does, reaches the four megakernels
  only on Quadrants and CUDA, and traces nothing at 0 -- checked by compiling
  the same gate on the CPU arch and comparing compiler keys;
* ``ALGAN_READONLY_NDARRAY_LDG`` (0006, ``ld.global.nc`` for read-only
  ndarray arguments) becomes an ``init`` keyword on CUDA only, and is not
  passed at all when unset.

Neither can be *seen* working here: both act in CUDA codegen. The GPU checks
are an A/B render pair per switch, compared byte for byte and timed.
"""

# No ``from __future__ import annotations``: the probe kernels need their
# annotations as runtime objects.

import ast
import inspect
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import algan.taichi_compat
from algan.errors import AlganConfigurationError, AlganWarning
from algan.settings import SETTINGS
from algan.taichi_compat import BACKEND, program, ti

quadrants_only = pytest.mark.skipif(
    BACKEND != "quadrants", reason="the switches drive Quadrants-only patches"
)

_RAYTRACING = Path(__file__).parents[2] / "algan" / "rendering" / "raytracing"

#: The kernels the register cap applies to, by the module that defines them.
_MEGAKERNELS = {
    "sheet_resolve_taichi.py": ("sheet_resolve_shade_arena",),
    "wavefront_kernels_taichi.py": (
        "wavefront_traverse_events_arena",
        "wavefront_shade_arena",
    ),
    "path_tracer_taichi.py": ("pt_shade_arena",),
}


# --- 0005: the register cap ---------------------------------------------------


def test_the_register_cap_takes_only_what_ptx_accepts(monkeypatch):
    monkeypatch.setattr(algan.taichi_compat, "BACKEND", "quadrants")
    experimental = SETTINGS.raytracing.experimental
    snapshot = SETTINGS.snapshot()
    try:
        assert SETTINGS.raytracing.megakernel_max_reg == 0
        for value in (16, 128, 255, 0):
            experimental.set(megakernel_max_reg=value)
            assert SETTINGS.raytracing.megakernel_max_reg == value
        # Below 16 or above 255 would fail inside ptxas at the first launch.
        for value in (8, 15, 256, -1, True, 64.0):
            with pytest.raises(AlganConfigurationError):
                experimental.set(megakernel_max_reg=value)
        assert SETTINGS.raytracing.megakernel_max_reg == 0
        with pytest.raises(AlganConfigurationError, match="experimental"):
            SETTINGS.raytracing.set(megakernel_max_reg=64)

        monkeypatch.setattr(algan.taichi_compat, "BACKEND", "taichi")
        with pytest.raises(AlganConfigurationError, match="0005"):
            experimental.set(megakernel_max_reg=64)
        experimental.set(megakernel_max_reg=0)
    finally:
        SETTINGS.restore(snapshot)


def test_an_unusable_environment_seed_warns_and_leaves_the_cap_off(monkeypatch):
    from algan.rendering.raytracing import settings as rt_settings

    monkeypatch.setattr(algan.taichi_compat, "BACKEND", "quadrants")
    assert rt_settings._seeded_max_reg(0) == 0
    assert rt_settings._seeded_max_reg(96) == 96
    with pytest.warns(AlganWarning, match="ALGAN_MEGAKERNEL_MAX_REG"):
        assert rt_settings._seeded_max_reg(300) == 0


def test_the_cap_reaches_kernels_only_on_quadrants_cuda(monkeypatch):
    """Another arch ignores ``max_reg`` but keys on it; Taichi has no such option."""
    from algan.rendering.raytracing import raytrace_kernels_taichi as kernels
    from algan.rendering.raytracing import settings as rt_settings
    from algan.taichi_compat import submodule

    impl = submodule("lang.impl")

    def on_arch(arch):
        monkeypatch.setattr(impl, "current_cfg", lambda: SimpleNamespace(arch=arch))

    monkeypatch.setattr(rt_settings, "megakernel_max_reg", 96)
    monkeypatch.setattr(algan.taichi_compat, "BACKEND", "quadrants")
    on_arch(ti.cuda)
    assert kernels.megakernel_register_cap() == 96
    on_arch(ti.x64)
    assert kernels.megakernel_register_cap() == 0
    on_arch(ti.cuda)
    monkeypatch.setattr(algan.taichi_compat, "BACKEND", "taichi")
    assert kernels.megakernel_register_cap() == 0
    monkeypatch.setattr(algan.taichi_compat, "BACKEND", "quadrants")
    monkeypatch.setattr(rt_settings, "megakernel_max_reg", 0)
    assert kernels.megakernel_register_cap() == 0


@quadrants_only
def test_the_gated_megakernels_keep_their_source_keys():
    """The source key walks everything a kernel references, the accessor included.

    Its first version reached the arch through ``taichi_runtime``, whose
    helpers end in ``importlib``: the walk gave up on all four megakernels, and
    each would have paid its whole frontend in every process.
    """
    import importlib

    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        megakernel_register_cap,
    )
    from algan.rendering.taichi_runtime import ensure_taichi_for_render
    from algan.utils import taichi_source_key as sk

    ensure_taichi_for_render()
    functions = [megakernel_register_cap]
    for module, names in _MEGAKERNELS.items():
        loaded = importlib.import_module(
            f"algan.rendering.raytracing.{module.removesuffix('.py')}"
        )
        functions += [getattr(loaded, name)._primal.func for name in names]
    for function in functions:
        sk._hash_function(function, sk._KeyContext(BACKEND), 0)


def test_the_cap_is_compiled_in():
    """A change between renders has to rebuild the program to reach a kernel."""
    from algan.rendering.raytracing.settings import KERNEL_COMPILED_IN_FIELDS

    assert "megakernel_max_reg" in KERNEL_COMPILED_IN_FIELDS


_GATE = (
    "if ti.static(megakernel_register_cap()):\n"
    "    ti.loop_config(max_reg=ti.static(megakernel_register_cap()))"
)


@pytest.mark.parametrize(
    ("module", "kernel"),
    [(module, kernel) for module, names in _MEGAKERNELS.items() for kernel in names],
)
def test_each_megakernel_gates_its_main_loop(module, kernel):
    """The gate sits directly before the kernel's first top-level ``for``.

    ``loop_config`` decorates the next loop traced, so anywhere else the cap
    would land on some other loop, or on none.
    """
    tree = ast.parse((_RAYTRACING / module).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel
    )
    loop = next(i for i, node in enumerate(function.body) if isinstance(node, ast.For))
    assert loop > 0
    assert ast.unparse(function.body[loop - 1]) == _GATE


@quadrants_only
def test_the_installed_compiler_has_both_options():
    """A wheel without 0005 or 0006 would reject the call or the keyword."""
    assert "max_reg" in inspect.signature(ti.loop_config).parameters
    from algan.rendering import taichi_runtime

    taichi_runtime.ensure_taichi_for_render()
    assert hasattr(program().config(), "readonly_ndarray_ldg")


@quadrants_only
def test_the_gate_lands_on_the_loop_and_traces_nothing_when_off(monkeypatch):
    """Compiled on whatever arch the suite runs, compared by compiler key.

    The gated form must compile exactly as a bare ``loop_config`` before the
    loop -- the cap reached the loop through the static branch -- and the gate
    at 0 exactly as the uncapped kernel, so turning the switch off leaves every
    megakernel's artifact what it was. The salt is a template value new to
    each run, so every kernel goes through the compiler and reports its key.
    """
    from algan.rendering import kernel_precompile, taichi_runtime

    taichi_runtime.ensure_taichi_for_render()
    keys = []
    note = kernel_precompile.note_materialized
    monkeypatch.setattr(
        kernel_precompile,
        "note_materialized",
        lambda token, **kwargs: keys.append(kwargs.get("cache_key"))
        or note(token, **kwargs),
    )
    on, off = 64, 0

    @ti.kernel
    def _plain(out: ti.types.ndarray(), salt: ti.template()):
        for i in range(out.shape[0]):
            out[i] += salt

    @ti.kernel
    def _direct(out: ti.types.ndarray(), salt: ti.template()):
        ti.loop_config(max_reg=64)
        for i in range(out.shape[0]):
            out[i] += salt

    @ti.kernel
    def _gated_on(out: ti.types.ndarray(), salt: ti.template()):
        if ti.static(on):
            ti.loop_config(max_reg=ti.static(on))
        for i in range(out.shape[0]):
            out[i] += salt

    @ti.kernel
    def _gated_off(out: ti.types.ndarray(), salt: ti.template()):
        if ti.static(off):
            ti.loop_config(max_reg=ti.static(off))
        for i in range(out.shape[0]):
            out[i] += salt

    salt = time.time_ns() % 1_000_000_007
    found = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for kernel in (_plain, _direct, _gated_on, _gated_off):
            keys.clear()
            kernel(torch.zeros(4), salt)
            found[kernel] = keys[-1]
    assert found[_gated_on] == found[_direct] != found[_plain]
    assert found[_gated_off] == found[_plain]


# --- 0006: read-only ndarray loads --------------------------------------------


@pytest.mark.parametrize(
    ("device", "flag", "passed"),
    [
        ("cuda", "1", True),
        ("cuda", None, False),
        ("cuda", "0", False),
        ("cpu", "1", False),
    ],
)
def test_the_readonly_ldg_flag_is_passed_only_on_cuda_when_asked(
    monkeypatch, device, flag, passed
):
    """Unset, the keyword is absent, so the off arm's init is the old one.

    On the CPU it is withheld even when set: that codegen ignores it, and the
    flag is in the compiler's config key on every arch.
    """
    from algan.rendering import taichi_runtime

    monkeypatch.setattr(taichi_runtime, "render_device", lambda: torch.device(device))
    if flag is None:
        monkeypatch.delenv("ALGAN_READONLY_NDARRAY_LDG", raising=False)
    else:
        monkeypatch.setenv("ALGAN_READONLY_NDARRAY_LDG", flag)
    kwargs = taichi_runtime.taichi_init_kwargs()
    if passed:
        assert kwargs["readonly_ndarray_ldg"] is True
    else:
        assert "readonly_ndarray_ldg" not in kwargs
