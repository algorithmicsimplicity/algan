"""Pin ``algan.__all__`` so a change to the public surface is a reviewable diff.

``from algan import *`` is the documented entry point, so the export list *is*
the public API. It is assembled by rules in ``algan/__init__.py`` rather than
written out by hand, which is what keeps it honest -- and also what lets it
move without anyone noticing: adding a module-level helper to an exported
module is enough to publish a new name.

This test holds the roster in a checked-in file. It fails with the added and
removed names spelled out, so a rename lands in the diff of
``public_api_snapshot.txt`` next to the code that caused it, and an accidental
export is caught at the commit that introduces it rather than at release.

Refresh it deliberately with ``ALGAN_UPDATE_API_SNAPSHOT=1`` and read the
result before committing -- the point is to notice, not to rubber-stamp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import algan
from algan.environment import env_flag

SNAPSHOT_PATH = Path(__file__).with_name("public_api_snapshot.txt")


def _current_exports() -> list[str]:
    return sorted(algan.__all__)


def _read_snapshot() -> list[str]:
    text = SNAPSHOT_PATH.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _write_snapshot(names: list[str]) -> None:
    SNAPSHOT_PATH.write_text("\n".join(names) + "\n", encoding="utf-8")


@pytest.mark.fast
def test_public_api_surface_matches_snapshot():
    current = _current_exports()

    if env_flag("ALGAN_UPDATE_API_SNAPSHOT", False):
        _write_snapshot(current)
        pytest.skip(
            f"Rewrote {SNAPSHOT_PATH.name} with {len(current)} names. "
            "Review the diff before committing."
        )

    if not SNAPSHOT_PATH.exists():
        _write_snapshot(current)
        pytest.fail(
            f"No API snapshot existed; wrote one with {len(current)} names. "
            "Review it and commit it."
        )

    expected = _read_snapshot()
    added = sorted(set(current) - set(expected))
    removed = sorted(set(expected) - set(current))

    if not added and not removed:
        return

    report = ["algan.__all__ has changed."]
    if added:
        report.append(f"  Newly exported ({len(added)}): {', '.join(added)}")
    if removed:
        report.append(f"  No longer exported ({len(removed)}): {', '.join(removed)}")
    report.append(
        "  If deliberate, re-run with ALGAN_UPDATE_API_SNAPSHOT=1 and commit "
        "the updated snapshot alongside the change."
    )
    pytest.fail("\n".join(report))


@pytest.mark.fast
def test_every_exported_name_resolves():
    """A name in ``__all__`` that does not resolve breaks ``import *`` outright."""
    missing = [name for name in algan.__all__ if not hasattr(algan, name)]
    assert not missing, f"__all__ names that do not resolve on the package: {missing}"


@pytest.mark.fast
def test_no_underscored_or_duplicate_exports():
    """The assembly rules should never emit a private name, or the same name twice."""
    private = [name for name in algan.__all__ if name.startswith("_")]
    assert not private, f"private names in __all__: {private}"

    seen = set()
    duplicates = sorted({n for n in algan.__all__ if n in seen or seen.add(n)})
    assert not duplicates, f"names exported more than once: {duplicates}"
