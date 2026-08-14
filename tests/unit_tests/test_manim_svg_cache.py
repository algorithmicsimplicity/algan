"""Round-trip guarantees for the persistent Manim SVG/Tex geometry cache.

The cache replaces Manim's ``SVGMobject.init_svg_mobject``, so a cache hit has
to reproduce everything the running Manim reads back off the mobject -- not just
the glyph tree. Manim 0.21 also reads ``id_to_vgroup_dict``
(``MathTex._break_up_by_substrings`` indexes it to split a formula into its
``tex_strings``), which a hit used to leave empty: the first ``Tex`` on a machine
succeeded and every later one died on ``KeyError: 'root'``.
"""

from __future__ import annotations

import manim as mn
import pytest
import torch

import algan
from algan.settings import SETTINGS
from algan.utils import manim_svg_cache as svg_cache


@pytest.fixture
def isolated_svg_cache(tmp_path):
    """Point the cache at an empty directory and clear the in-process memo."""
    SETTINGS.paths.set(cache_directory=str(tmp_path))
    saved = dict(svg_cache._MEM_CACHE)
    svg_cache._MEM_CACHE.clear()
    try:
        yield tmp_path
    finally:
        svg_cache._MEM_CACHE.clear()
        svg_cache._MEM_CACHE.update(saved)


def _container_with_groups():
    """A stand-in SVGMobject: three glyphs and a group map referencing them.

    Mirrors what ``get_mobjects_from`` produces -- the VGroups hold the very
    glyph instances that are also the container's submobjects.
    """
    glyphs = [mn.VMobject() for _ in range(3)]
    for i, glyph in enumerate(glyphs):
        glyph.set_points_as_corners(
            [[0.0, 0.0, 0.0], [float(i + 1), 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
    container = mn.VMobject()
    container.add(*glyphs)
    container.id_to_vgroup_dict = {
        "root": mn.VGroup(*glyphs),
        "unique000": mn.VGroup(glyphs[0], glyphs[2]),
        "empty": mn.VGroup(),
    }
    return container, glyphs


def test_rebuild_restores_the_group_map_onto_the_rebuilt_glyphs():
    source, _ = _container_with_groups()

    nodes, groups = svg_cache._parse_recipe(svg_cache._extract(source))
    target = mn.VMobject()
    target.id_to_vgroup_dict = {}
    svg_cache._rebuild(target, nodes, groups)

    assert set(target.id_to_vgroup_dict) == {"root", "unique000", "empty"}
    assert len(target.id_to_vgroup_dict["empty"].submobjects) == 0
    # The map must hand out the *rebuilt* glyphs: MathTex re-parents whatever it
    # finds there, so members aliasing the source (or a throwaway copy) would
    # graft foreign mobjects into the scene.
    assert [id(m) for m in target.id_to_vgroup_dict["root"].submobjects] == [
        id(m) for m in target.submobjects
    ]
    assert [id(m) for m in target.id_to_vgroup_dict["unique000"].submobjects] == [
        id(target.submobjects[0]),
        id(target.submobjects[2]),
    ]


def test_recipes_round_trip_without_a_group_map():
    """Manim < 0.21 has no ``id_to_vgroup_dict``; the recipe just omits it."""
    source, _ = _container_with_groups()
    del source.id_to_vgroup_dict

    nodes, groups = svg_cache._parse_recipe(svg_cache._extract(source))
    assert groups is None

    target = mn.VMobject()
    svg_cache._rebuild(target, nodes, groups)

    assert len(target.submobjects) == 3
    assert not hasattr(target, "id_to_vgroup_dict")


def test_untagged_recipes_are_reparsed_when_the_group_map_is_needed():
    """A pre-group entry from an older Algan must not be replayed.

    Replaying one leaves ``id_to_vgroup_dict`` empty, which is exactly the
    failure this cache shipped with; the entry has to be detected and dropped.
    """
    source, _ = _container_with_groups()
    legacy = tuple(svg_cache._extract_node(sm) for sm in source.submobjects)

    nodes, groups = svg_cache._parse_recipe(legacy)
    assert groups is None
    assert len(nodes) == 3

    tagged = svg_cache._extract(source)
    assert tagged[0] == svg_cache._RECIPE_TAG
    assert svg_cache._parse_recipe(tagged)[1] is not None


def test_tex_survives_a_cache_hit(isolated_svg_cache):
    """The end-to-end regression: the second Tex is the one that used to die."""
    cold = algan.Tex("x^2", "+ 1")
    assert isolated_svg_cache.joinpath("manim_svg").exists()

    warm = algan.Tex("x^2", "+ 1")

    # Segment sizes come straight from the MathTex substring split, so equality
    # here means the cache hit reproduced the group map the split relies on.
    assert torch.equal(warm.num_mobs_per_segment, cold.num_mobs_per_segment)
    assert warm.num_mobs_per_segment.sum() > 0
