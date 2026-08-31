"""``IN``/``OUT`` belong to the script, so the library must not read them.

Algan carries no compatibility aliases for its own API -- ``CLAUDE.md`` says so,
and one name per thing is the rule everywhere else. These two are the deliberate
exception, for a reason this module makes permanent: ``in`` and ``out`` are
ordinary enough words that a script will want them, and a name the library
depends on is a poor thing to leave in the way. ``INWARD``/``OUTWARD`` are what
Algan's own source says; ``IN``/``OUT`` are the same objects under the names a
script is free to keep, rebind, or ignore.

The walk below is over the AST rather than the text, so a comment saying "OUT"
and a Taichi kernel's ``# [n] i32 OUT`` output marker are not mistaken for
reads of the constant.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import algan
from algan.constants import spatial

PACKAGE_ROOT = Path(algan.__file__).resolve().parent

#: The one module allowed to say ``IN``/``OUT``: the one that defines them.
_ALLOWED = {PACKAGE_ROOT / "constants" / "spatial.py"}

pytestmark = pytest.mark.fast


def test_the_short_names_are_the_long_ones():
    assert spatial.IN is spatial.INWARD
    assert spatial.OUT is spatial.OUTWARD
    assert torch.equal(spatial.OUTWARD, -spatial.INWARD)
    # +z runs towards the viewer -- Manim's, Three.js's and glTF's convention --
    # so OUTWARD is +z and the world basis (RIGHT, UP, OUTWARD) is right-handed.
    assert torch.equal(spatial.OUTWARD, torch.tensor((0.0, 0.0, 1.0)))
    assert torch.equal(torch.linalg.cross(spatial.RIGHT, spatial.UP), spatial.OUTWARD)
    # A Mob's default orientation faces the way the camera looks, into the
    # scene, so its forward axis is INWARD rather than OUTWARD.
    assert torch.equal(spatial.DEFAULT_BASIS[2], spatial.INWARD)


def test_all_four_names_are_star_exported():
    for name in ("IN", "OUT", "INWARD", "OUTWARD"):
        assert name in algan.__all__, f"{name} is missing from algan.__all__"
        assert getattr(algan, name) is getattr(spatial, name)


def _modules():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "external_libraries" in path.parts or path in _ALLOWED:
            continue
        yield path


def test_no_algan_module_reads_in_or_out():
    """A shadowed ``OUT`` in a script must not be able to reach the library.

    It cannot today -- every Algan module binds its own copy at import -- but
    that is a property of how the imports happen to be written, and this makes
    it a property of the names instead.
    """
    offenders = []
    for path in _modules():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover -- would fail the build anyway
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ("IN", "OUT"):
                offenders.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} reads {node.id}"
                )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in ("IN", "OUT"):
                        offenders.append(
                            f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} "
                            f"imports {alias.name}"
                        )

    assert not offenders, (
        "Algan's own source must say INWARD/OUTWARD, so that a script is free "
        "to bind IN and OUT to whatever it likes:\n  " + "\n  ".join(offenders)
    )
