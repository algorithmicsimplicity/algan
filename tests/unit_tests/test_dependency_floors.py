"""The declared ``torch`` floor admits only versions that have what Algan calls.

A floor is what a resolver keeps when the user already has something
installed: ``pip install algan`` beside a conda or CUDA-matched torch 2.2
keeps that torch, because it satisfies the requirement. ``torch>=2.0.0`` let
it, and the first transform then raised ``AttributeError: module 'torch' has
no attribute 'get_default_device'``, which torch 2.3 added. CI installs the
lock's newer torch, so nothing there can see a floor that is too low; this
reads the floor and the source instead.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]

#: ``torch.<name>`` attributes Algan reaches for, and the release that added
#: each. Extend it when the package starts calling something newer.
TORCH_INTRODUCED = {"get_default_device": Version("2.3")}


def _torch_floor() -> Version:
    # By hand rather than with tomllib, which Python 3.10, the floor CI tests,
    # does not have: the [project] dependencies array, one string per line.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"(?ms)^\[project\]$.*?^dependencies = \[$(.*?)^\]$", text)
    assert block, "pyproject.toml has no [project] dependencies array"
    for line in re.findall(r'^\s*"([^"]+)",?\s*$', block.group(1), re.M):
        requirement = Requirement(line)
        if requirement.name == "torch" and requirement.marker is None:
            floors = [
                Version(s.version) for s in requirement.specifier if s.operator == ">="
            ]
            assert floors, f"torch has no >= floor in pyproject.toml: {line!r}"
            return max(floors)
    raise AssertionError("pyproject.toml declares no torch dependency")


def _torch_attributes_used() -> set[str]:
    used = set()
    for path in (ROOT / "algan").rglob("*.py"):
        if "external_libraries" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        # Parse only what can match: the AST of the whole package is seconds.
        if not any(f"torch.{name}" in source for name in TORCH_INTRODUCED):
            continue
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
                and node.attr in TORCH_INTRODUCED
            ):
                used.add(node.attr)
    return used


def test_the_torch_floor_has_every_function_algan_calls():
    floor = _torch_floor()
    used = _torch_attributes_used()
    assert "get_default_device" in used, "the scan no longer finds a known call"
    too_new = {
        name: str(TORCH_INTRODUCED[name])
        for name in used
        if TORCH_INTRODUCED[name] > floor
    }
    assert not too_new, (
        f"pyproject.toml lets torch {floor} install, but algan/ calls {too_new} "
        f"(name: first torch release with it). Raise the floor in "
        f"[project] dependencies and run `uv lock`."
    )
