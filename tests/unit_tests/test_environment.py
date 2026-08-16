from __future__ import annotations

import ast
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from algan.environment import (
    ALGAN_ENVIRONMENT_VARIABLES,
    env_flag,
    env_float,
    env_int,
    env_is_set,
    env_overrides,
    env_str,
    startup_environment_variables,
    unknown_algan_environment_variables,
    warn_for_unknown_algan_environment_variables,
)
from algan.errors import AlganConfigurationError, AlganWarning


def _own_nodes(statement):
    """Every node belonging to ``statement`` itself, not to statements inside it."""
    stack = [statement]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, ast.stmt)
        )


def _is_environ_access(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr in ("environ", "getenv")
    )


def _is_environ_subscript_write(statement):
    targets = getattr(statement, "targets", None) or [getattr(statement, "target", None)]
    return any(
        isinstance(target, ast.Subscript) and _is_environ_access(target.value)
        for target in targets
        if target is not None
    )


#: Names reached through one of these are accounted for, even when the same
#: statement also touches ``os.environ`` (``dict(os.environ, **env_overrides(...))``).
_ACCESSORS = frozenset(
    {"env_flag", "env_float", "env_int", "env_is_set", "env_overrides", "env_str"}
)


def _algan_names(nodes):
    accounted = set()
    for node in nodes:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _ACCESSORS
        ):
            accounted.update(id(child) for child in ast.walk(node))
    literals = {
        node.value
        for node in nodes
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("ALGAN_")
        and id(node) not in accounted
    }
    keywords = {
        node.arg
        for node in nodes
        if isinstance(node, ast.keyword)
        and (node.arg or "").startswith("ALGAN_")
        and id(node) not in accounted
    }
    return sorted(literals | keywords)


def test_package_reaches_algan_variables_only_through_the_accessors():
    """The one rule that keeps the registry in algan/environment.py honest.

    Every ``ALGAN_`` read goes through an ``env_*`` accessor, which rejects a
    name that is not declared -- so a knob added without a declaration fails
    loudly on its first read instead of becoming a variable users can misspell
    without warning. This test is what stops the package from routing around
    that, and it needs no maintenance as variables come and go.

    Writes through ``os.environ[...]`` are allowed: :mod:`algan.daemon` has to
    set ``ALGAN_DAEMON_CHILD`` before importing algan at all, and a misspelled
    write surfaces as an unknown-variable warning in the child process.
    """
    source_root = Path(__file__).parents[2] / "algan"
    violations = []
    for source_path in source_root.rglob("*.py"):
        if (
            "external_libraries" in source_path.parts
            or source_path.name == "environment.py"
        ):
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for statement in ast.walk(tree):
            if not isinstance(statement, ast.stmt):
                continue
            if _is_environ_subscript_write(statement):
                continue
            nodes = list(_own_nodes(statement))
            if not any(_is_environ_access(node) for node in nodes):
                continue
            for name in _algan_names(nodes):
                violations.append(
                    f"{source_path.relative_to(source_root.parent)}:"
                    f"{statement.lineno}: {name}"
                )

    assert not violations, (
        "read these through algan.environment's env_* accessors instead of os:\n"
        + "\n".join(violations)
    )


def test_undeclared_names_are_rejected_by_every_accessor():
    for read in (
        lambda: env_str("ALGAN_NOT_DECLARED"),
        lambda: env_flag("ALGAN_NOT_DECLARED", False),
        lambda: env_int("ALGAN_NOT_DECLARED", 1),
        lambda: env_float("ALGAN_NOT_DECLARED", 1.0),
        lambda: env_is_set("ALGAN_NOT_DECLARED"),
        lambda: env_overrides(ALGAN_NOT_DECLARED="1"),
    ):
        with pytest.raises(AlganConfigurationError, match="not a declared"):
            read()


def test_startup_variables_are_declared_and_ordered():
    startup = startup_environment_variables()
    assert startup[:2] == ("ALGAN_ANIMATION_DEVICE", "ALGAN_RENDER_DEVICE")
    assert "TI_OFFLINE_CACHE_FILE_PATH" in startup
    algan_startup = {name for name in startup if name.startswith("ALGAN_")}
    assert algan_startup <= ALGAN_ENVIRONMENT_VARIABLES


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("TRUE", True), (" yes ", True), ("0", False), ("off", False)],
)
def test_env_flag_parses_both_spellings(monkeypatch, value, expected):
    monkeypatch.setenv("ALGAN_HYBRID_RASTER", value)
    assert env_flag("ALGAN_HYBRID_RASTER", not expected) is expected


def test_env_accessors_fall_back_to_the_default(monkeypatch):
    monkeypatch.delenv("ALGAN_KBUF", raising=False)
    assert env_int("ALGAN_KBUF", 4) == 4
    assert not env_is_set("ALGAN_KBUF")

    monkeypatch.setenv("ALGAN_KBUF", "")
    assert env_int("ALGAN_KBUF", 4) == 4
    assert env_is_set("ALGAN_KBUF")

    monkeypatch.setenv("ALGAN_KBUF", "8")
    assert env_int("ALGAN_KBUF", 4) == 8


@pytest.mark.parametrize(
    ("accessor", "default"),
    [(env_flag, False), (env_int, 4), (env_float, 1.5)],
)
def test_unusable_values_warn_and_keep_the_default(monkeypatch, accessor, default):
    monkeypatch.setenv("ALGAN_KBUF", "nonsense")
    with pytest.warns(AlganWarning, match="ALGAN_KBUF='nonsense' is not"):
        assert accessor("ALGAN_KBUF", default) == default


def test_env_str_returns_the_value_unstripped(monkeypatch):
    monkeypatch.setenv("ALGAN_BVH_BUILD", " morton ")
    assert env_str("ALGAN_BVH_BUILD", "split") == " morton "
    monkeypatch.delenv("ALGAN_BVH_BUILD")
    assert env_str("ALGAN_BVH_BUILD", "split") == "split"
    assert env_str("ALGAN_BVH_BUILD") == ""


def test_unknown_algan_environment_variables_are_sorted_and_warned():
    environ = {
        "PATH": "ignored",
        "ALGAN_RENDER_DEVICE": "cpu",
        "ALGAN_Z_UNKNOWN": "1",
        "ALGAN_A_UNKNOWN": "1",
    }

    assert unknown_algan_environment_variables(environ) == (
        "ALGAN_A_UNKNOWN",
        "ALGAN_Z_UNKNOWN",
    )
    with pytest.warns(
        AlganWarning,
        match="Unknown Algan environment variables: ALGAN_A_UNKNOWN, ALGAN_Z_UNKNOWN",
    ):
        warn_for_unknown_algan_environment_variables(environ)


def test_all_registered_algan_environment_variables_are_accepted():
    environ = dict.fromkeys(ALGAN_ENVIRONMENT_VARIABLES, "unused")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_for_unknown_algan_environment_variables(environ)
    assert not caught


def test_harness_variables_are_accepted():
    """Set by the test/bench harnesses, so exporting one must not warn."""
    assert not unknown_algan_environment_variables(
        {
            "ALGAN_RUN_DOC_RENDERS": "1",
            "ALGAN_RUN_FULL_RENDERS": "1",
            "ALGAN_UPDATE_FAST_BASELINE": "1",
            "ALGAN_UPDATE_FULL_RENDER_BASELINES": "1",
        }
    )


@pytest.mark.slow
def test_import_algan_warns_for_an_unknown_environment_variable(tmp_path):
    environ = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("ALGAN_")
    }
    environ.update(
        {
            "ALGAN_ANIMATION_DEVICE": "cpu",
            "ALGAN_HOME": str(tmp_path / "algan_home"),
            "ALGAN_RENDER_DEVIC": "cpu",
            "ALGAN_RENDER_DEVICE": "cpu",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import algan"],
        capture_output=True,
        check=False,
        cwd=os.fspath(Path(__file__).parents[2]),
        env=environ,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Unknown Algan environment variable: ALGAN_RENDER_DEVIC" in completed.stderr
