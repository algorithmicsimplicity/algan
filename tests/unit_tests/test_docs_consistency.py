"""Facts that the documentation and the agent skill state about the repository.

``test_doc_examples.py`` checks the code in the documentation. This module
checks the few things around that code which drift without anything failing:
which modules the reference manual documents, and the commit and version the
agent skill says it describes.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import algan

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / ".claude" / "skills" / "algan-video"
REFERENCE_INDEX = REPO_ROOT / "docs" / "source" / "reference_index"


def _reference_modules() -> set[str]:
    """Every module the reference index hands to autosummary.

    An entry is ``~module`` under an optional ``.. currentmodule::``, which is
    how ``reference_index/*.rst`` writes them; autosummary gives each one a
    page, and the module template lists the classes defined in it.
    """
    modules = set()
    for page in REFERENCE_INDEX.glob("*.rst"):
        current = ""
        for line in page.read_text(encoding="utf-8").splitlines():
            directive = re.match(r"\.\.\s+currentmodule::\s*(\S+)", line)
            if directive:
                current = directive.group(1)
                continue
            entry = re.match(r"^\s+~([A-Za-z_][\w.]*)\s*$", line)
            if entry:
                name = entry.group(1)
                if current and not name.startswith("algan"):
                    name = f"{current}.{name}"
                modules.add(name)
    return modules


def test_every_public_class_has_a_reference_page():
    """Each class in ``algan.__all__`` lives in a module the reference documents.

    The 87 root adapters (``Arc``, ``Axes``, ``SVGMob``, ...) once had no page,
    because ``algan.mobs.manim_adapters`` was missing from the index: every
    link to them landed on the ``mn.`` class of the same name, whose
    signature is in radians, and five of them had no page at all.
    """
    documented = _reference_modules()
    missing: dict[str, list[str]] = {}
    for name in algan.__all__:
        obj = getattr(algan, name, None)
        if inspect.isclass(obj) and obj.__module__ not in documented:
            missing.setdefault(obj.__module__, []).append(name)
    assert not missing, (
        "public classes defined in modules that docs/source/reference_index/ "
        f"does not list: {missing}. Add the module to the autosummary in the "
        "matching reference_index page."
    )


def _skill_metadata(field: str) -> str:
    header = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    match = re.search(rf'^\s*{re.escape(field)}:\s*"([^"]*)"', header, re.M)
    assert match, f"SKILL.md's header has no {field!r} field"
    return match.group(1)


def test_skill_pins_one_commit():
    """The skill names one commit, in its header and in every source link.

    They named three different commits once, so an agent resolving API drift
    followed links into a tree two months older than the text it was reading.
    """
    commit = _skill_metadata("algan-source-commit")
    assert re.fullmatch(r"[0-9a-f]{40}", commit), (
        f"algan-source-commit should be one full commit hash, got {commit!r}"
    )
    sources = (SKILL_ROOT / "references" / "api-sources.md").read_text(encoding="utf-8")
    named = set(re.findall(r"\b[0-9a-f]{40}\b", sources))
    assert named == {commit}, (
        f"references/api-sources.md names commits {sorted(named)}, but SKILL.md's "
        f"algan-source-commit is {commit}. When the skill is re-checked against "
        "the source, update both, and the links, to the same commit."
    )


def test_skill_version_is_the_package_version():
    """``algan-version`` in the skill is the version in ``pyproject.toml``.

    A version bump fails this until the skill has been re-checked against the
    source: update ``algan-version``, ``algan-source-commit``, ``source-checked``
    and the links in ``references/api-sources.md`` together.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    assert match, "pyproject.toml has no version"
    assert _skill_metadata("algan-version") == match.group(1), (
        f"SKILL.md says algan-version {_skill_metadata('algan-version')!r}, "
        f"pyproject.toml says {match.group(1)!r}"
    )
