"""Fast, render-free tests for the Manim parity manifest helpers."""

from __future__ import annotations

import re
import types

import pytest

from algan.mobs import manim_parity as mp


def _full_namespace():
    return {name: object() for name in mp.MANIM_MOBJECT_NAMES}


def test_version_string_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", mp.MANIM_COMMUNITY_VERSION)


def test_mobject_names_are_unique_and_identifiers():
    names = mp.MANIM_MOBJECT_NAMES
    assert len(names) == len(set(names))
    assert all(n.isidentifier() for n in names)


@pytest.mark.parametrize(
    "table",
    [
        "MANIM_OPENGL_MOBJECT_NAMES",
        "MANIM_PRIVATE_MOBJECT_NAMES",
        "MANIM_EXTERNAL_TOOL_MOBJECT_NAMES",
        "MANIM_PANGO_MOBJECT_NAMES",
        "MANIM_UNVENDORED_MOBJECT_NAMES",
    ],
)
def test_sub_tables_are_unique_subsets_of_the_inventory(table):
    values = getattr(mp, table)
    assert isinstance(values, tuple)
    assert len(values) == len(set(values))
    assert set(values) <= set(mp.MANIM_MOBJECT_NAMES)


def test_opengl_table_is_exactly_the_opengl_prefixed_names():
    expected = {n for n in mp.MANIM_MOBJECT_NAMES if n.startswith("OpenGL")}
    assert set(mp.MANIM_OPENGL_MOBJECT_NAMES) == expected


def test_private_table_is_exactly_the_underscore_names():
    expected = {n for n in mp.MANIM_MOBJECT_NAMES if n.startswith("_")}
    assert set(mp.MANIM_PRIVATE_MOBJECT_NAMES) == expected


def test_pango_names_are_external_tool_names():
    assert set(mp.MANIM_PANGO_MOBJECT_NAMES) <= set(
        mp.MANIM_EXTERNAL_TOOL_MOBJECT_NAMES
    )


def test_all_lists_existing_public_names():
    assert len(mp.__all__) == len(set(mp.__all__))
    for name in mp.__all__:
        assert hasattr(mp, name)


def test_full_namespace_has_nothing_missing():
    ns = _full_namespace()
    assert mp.missing_manim_mobjects(ns) == ()
    assert mp.validate_manim_mobject_parity(ns) is None


def test_empty_namespace_reports_everything_but_pango():
    missing = mp.missing_manim_mobjects({})
    expected = tuple(
        n for n in mp.MANIM_MOBJECT_NAMES if n not in mp.MANIM_PANGO_MOBJECT_NAMES
    )
    assert missing == expected


def test_missing_names_preserve_manifest_order():
    ns = _full_namespace()
    for name in ("Square", "Arc", "VGroup"):
        del ns[name]
    assert mp.missing_manim_mobjects(ns) == ("Arc", "Square", "VGroup")


def test_extra_names_in_namespace_are_ignored():
    ns = _full_namespace()
    ns["NotAManimThing"] = 1
    assert mp.missing_manim_mobjects(ns) == ()


def test_values_do_not_matter_only_keys():
    ns = dict.fromkeys(mp.MANIM_MOBJECT_NAMES)
    assert mp.missing_manim_mobjects(ns) == ()


def test_all_pango_names_absent_together_is_exempt():
    ns = _full_namespace()
    for name in mp.MANIM_PANGO_MOBJECT_NAMES:
        del ns[name]
    assert mp.missing_manim_mobjects(ns) == ()
    mp.validate_manim_mobject_parity(ns)


@pytest.mark.parametrize("name", mp.MANIM_PANGO_MOBJECT_NAMES)
def test_single_pango_name_missing_is_still_a_gap(name):
    ns = _full_namespace()
    del ns[name]
    assert mp.missing_manim_mobjects(ns) == (name,)


def test_two_of_three_pango_names_missing_are_reported():
    ns = _full_namespace()
    a, b, _ = mp.MANIM_PANGO_MOBJECT_NAMES
    del ns[a], ns[b]
    assert mp.missing_manim_mobjects(ns) == (a, b)


def test_pango_exemption_does_not_hide_other_gaps():
    ns = _full_namespace()
    for name in (*mp.MANIM_PANGO_MOBJECT_NAMES, "Circle"):
        del ns[name]
    assert mp.missing_manim_mobjects(ns) == ("Circle",)


def test_accepts_any_mapping():
    ns = types.MappingProxyType(_full_namespace())
    assert mp.missing_manim_mobjects(ns) == ()


def test_returns_tuple():
    assert isinstance(mp.missing_manim_mobjects({}), tuple)
    assert isinstance(mp.missing_manim_mobjects(_full_namespace()), tuple)


def test_validate_raises_with_count_version_and_names():
    ns = _full_namespace()
    del ns["Dot"], ns["Torus"]
    with pytest.raises(RuntimeError) as info:
        mp.validate_manim_mobject_parity(ns)
    msg = str(info.value)
    assert msg == (f"Missing 2 Manim {mp.MANIM_COMMUNITY_VERSION} Mobjects: Dot, Torus")


def test_validate_on_empty_namespace_counts_non_pango_names():
    with pytest.raises(RuntimeError, match=r"^Missing (\d+) ") as info:
        mp.validate_manim_mobject_parity({})
    count = int(re.match(r"Missing (\d+)", str(info.value)).group(1))
    assert count == len(mp.MANIM_MOBJECT_NAMES) - len(mp.MANIM_PANGO_MOBJECT_NAMES)


def test_real_algan_manim_namespace_passes():
    import algan.manim as mn

    assert mn.missing_manim_mobjects(vars(mn)) == ()
    mn.validate_manim_mobject_parity(vars(mn))
    assert mn.missing_manim_mobjects is mp.missing_manim_mobjects
