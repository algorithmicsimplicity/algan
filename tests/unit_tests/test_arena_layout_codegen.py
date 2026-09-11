"""Generated ABI regions and explicit kernel interfaces cannot drift silently."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = runpy.run_path(str(ROOT / "scripts/generate_arena_bindings.py"))


def test_arena_regions_are_generated_from_the_canonical_layout():
    expected = GENERATOR["generate_sources"](ROOT)
    assert len(expected) == 4  # five kernels, with two in the wavefront module
    for path, source in expected.items():
        assert path.read_text(encoding="utf-8") == source, (
            f"{path}: run python scripts/generate_arena_bindings.py"
        )


@pytest.mark.parametrize(
    "fields",
    [
        ("a", "a"),
        (("a", "f32", 0),),
        (("a", "f32", True),),
        (("a", "bad", 2),),
        (("a", "f32"),),
        ((),),
        ("bad name",),
    ],
)
def test_layout_rejects_invalid_fields(fields):
    with pytest.raises(ValueError):
        GENERATOR["_validated_fields"](fields)


def test_layout_preserves_call_order_and_derives_bound_order():
    names, spec = GENERATOR["_validated_fields"](
        ("n", ("coords", "f32", 2), "gate", ("ids", "i32", 1))
    )
    assert names == ["n", "coords", "gate", "ids"]
    assert spec == [("coords", "f32", 2), ("ids", "i32", 1)]


def test_generator_rejects_missing_and_duplicate_regions():
    replace = GENERATOR["_replace_region"]
    with pytest.raises(ValueError, match="exactly one"):
        replace("", "example", "layout", "")
    region = "# BEGIN GENERATED example layout\n# END GENERATED example layout\n"
    with pytest.raises(ValueError, match="exactly one"):
        replace(region + region, "example", "layout", "")
