"""The source key's two belts key what can reach a kernel, and nothing else.

``_environment_fingerprint`` and ``_settings_fingerprint`` back up the bytecode
walk (see the ``taichi_source_key`` module docstring). Keyed too broadly they
cost no correctness, only misses: every index lookup of a run that set a
host-only setting, or an ``ALGAN_`` variable to its own default, paid the
whole Python frontend again. These tests pin both halves of the trade -- what
is left out does not change the key, and what is left out is provably out of
a kernel's reach -- by reading the package source.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import algan
from algan import SETTINGS
from algan.environment import _LIVE_VARIABLES
from algan.settings import raytracing_settings
from algan.utils import taichi_source_key as sk

PACKAGE = Path(algan.__file__).parent


def _settings_key():
    from algan.taichi_compat import BACKEND

    ctx = sk._KeyContext(BACKEND)
    sk._settings_fingerprint(ctx)
    return list(ctx.out)


def _is_kernel_definition(node):
    """A ``def`` decorated ``@ti.kernel`` / ``@ti.func`` (or the ``qd.`` spelling)."""
    if not isinstance(node, ast.FunctionDef):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and target.attr in ("kernel", "func", "real_func")
            and isinstance(target.value, ast.Name)
            and target.value.id in ("ti", "qd")
        ):
            return True
    return False


def _sources():
    for path in PACKAGE.rglob("*.py"):
        if "external_libraries" in path.parts:
            continue
        yield path, path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ALGAN_PT_SEED", "0"),  # a renderer-setting seed, at its default
        ("ALGAN_RENDER_DEVICE", "cpu"),  # seeds computing.render_device
        ("ALGAN_TORCH_COMPILE", "0"),  # torch, never a Taichi kernel
    ],
)
def test_a_variable_that_cannot_change_a_kernel_leaves_the_key_alone(
    monkeypatch, name, value
):
    monkeypatch.delenv(name, raising=False)
    before = sk._environment_fingerprint()
    monkeypatch.setenv(name, value)
    assert sk._environment_fingerprint() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rendering_memory_fraction", 0.41),
        ("cpu_render_memory_budget", 123 * 2**20),
        ("max_animation_batch_size", 17),
    ],
)
def test_a_host_only_computing_setting_leaves_the_key_alone(field, value):
    before = _settings_key()
    with SETTINGS.computing.override(**{field: value}):
        assert _settings_key() == before


def test_a_kernel_selecting_computing_setting_is_still_keyed():
    before = _settings_key()
    flipped = SETTINGS.computing.mps_friendly is not True
    with SETTINGS.computing.override(mps_friendly=flipped):
        assert _settings_key() != before


def test_a_live_kernel_variable_is_still_keyed(monkeypatch):
    # Read at render time, not seeded into a setting: only the env belt sees it.
    keyable = sorted(
        name
        for name in _LIVE_VARIABLES
        if name.startswith("ALGAN_")
        and name not in sk._ENV_NOT_IN_KEY
        and name not in sk._settings_seed_variables()
    )
    assert keyable, "no live variable left in the key to test with"
    name = keyable[0]
    monkeypatch.delenv(name, raising=False)
    before = sk._environment_fingerprint()
    monkeypatch.setenv(name, "1")
    assert sk._environment_fingerprint() != before


def test_the_derived_seeds_are_raytracing_fields_set_from_the_environment():
    seeds = sk._settings_seed_variables()
    # A handful of well-known ones, so a regex that stops matching fails here
    # rather than silently putting every seed back in the key.
    assert {"ALGAN_PT_SEED", "ALGAN_KBUF", "ALGAN_BVH_REFIT"} <= seeds
    # Private module constants are not fields and are not hashed by value.
    from algan.environment import ALGAN_ENVIRONMENT_VARIABLES

    assert seeds <= ALGAN_ENVIRONMENT_VARIABLES


def test_a_seed_variable_is_read_nowhere_but_its_seed():
    """Dropping a seed is sound only if the field's value is all it decides.

    So every read of one must sit at the top level of a settings storage
    module. A second read inside a function -- a live override at render time
    -- would let the variable change a kernel without the settings value
    changing, and needs the variable back in the key.
    """
    seeds = sk._settings_seed_variables()
    raytracing_settings._resolve()
    storage = {
        Path(module.__file__).resolve()
        for module in raytracing_settings._FIELD_MODULE.values()
    }
    allowed_mentions = {
        (PACKAGE / "environment.py").resolve(),  # the declaration
        (PACKAGE / "settings" / "raytracing_settings.py").resolve(),  # messages
    }
    offenders = []
    for path, source in _sources():
        resolved = path.resolve()
        if resolved in allowed_mentions:
            continue
        tree = ast.parse(source)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        ]
        in_function = {
            id(inner) for function in functions for inner in ast.walk(function)
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and node.value in seeds):
                continue
            if resolved not in storage or id(node) in in_function:
                offenders.append(
                    f"{path.relative_to(PACKAGE)}:{node.lineno} {node.value}"
                )
    assert not offenders, (
        "these seed variables are also read outside their module-level seed; "
        "keep them in the key (see _settings_seed_variables):\n" + "\n".join(offenders)
    )


def test_host_only_computing_fields_are_real_and_never_reach_a_kernel_module():
    fields = set(SETTINGS.computing.to_dict())
    assert fields >= sk._HOST_ONLY_COMPUTING_FIELDS
    # The settings that do select kernels stay in.
    assert {"render_device", "mps_friendly"} <= fields - sk._HOST_ONLY_COMPUTING_FIELDS

    offenders = []
    kernel_modules = 0
    for path, source in _sources():
        tree = ast.parse(source)
        if not any(map(_is_kernel_definition, ast.walk(tree))):
            continue
        kernel_modules += 1
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for field in sorted(sk._HOST_ONLY_COMPUTING_FIELDS & names):
            offenders.append(f"{path.relative_to(PACKAGE)}: {field}")
    # So a change in how kernels are spelled cannot turn this into a no-op.
    assert kernel_modules >= 20
    assert not offenders, (
        "a kernel module reads a computing field the source key leaves out; "
        "drop it from _HOST_ONLY_COMPUTING_FIELDS:\n" + "\n".join(offenders)
    )
