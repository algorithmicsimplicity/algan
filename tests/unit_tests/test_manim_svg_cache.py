"""Round-trip guarantees for the persistent Manim SVG/Tex geometry cache.

The cache replaces Manim's ``SVGMobject.init_svg_mobject``, so a cache hit has
to reproduce everything the running Manim reads back off the mobject -- not just
the glyph tree. Manim 0.21 also reads ``id_to_vgroup_dict``
(``MathTex._break_up_by_substrings`` indexes it to split a formula into its
``tex_strings``), which a hit used to leave empty: the first ``Tex`` on a machine
succeeded and every later one died on ``KeyError: 'root'``.
"""

from __future__ import annotations

import pathlib

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


_SQUARE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
    'width="100" height="100">'
    '<path d="M10 10 L90 10 L90 90 L10 90 Z" fill="#3366cc"/>'
    "</svg>"
)
_THREE_SHAPE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
    'width="100" height="100">'
    '<path d="M50 10 L90 90 L10 90 Z" fill="#cc3366"/>'
    '<path d="M20 20 L40 20 L40 40 L20 40 Z" fill="#33cc66"/>'
    '<circle cx="70" cy="30" r="15" fill="#cccc33"/>'
    "</svg>"
)


def test_editing_a_user_svg_is_not_served_from_the_cache(isolated_svg_cache, tmp_path):
    """A user's own SVG is keyed on its bytes, not its name.

    The key used to be the file *basename*, which is content-addressed for
    manim's Tex output (``tex_hash(source) + .svg``) but says nothing about a
    file the user drew. Editing ``logo.svg`` and re-running therefore replayed
    the previous drawing -- silently, and across processes, because the cache
    persists to disk.
    """
    svg_file = tmp_path / "logo.svg"

    svg_file.write_text(_SQUARE_SVG)
    before = len(mn.SVGMobject(svg_file).submobjects)

    svg_file.write_text(_THREE_SHAPE_SVG)
    after = len(mn.SVGMobject(svg_file).submobjects)

    assert before == 1
    assert after == 3


def test_two_different_svgs_sharing_a_basename_do_not_collide(
    isolated_svg_cache, tmp_path
):
    first = tmp_path / "a" / "logo.svg"
    second = tmp_path / "b" / "logo.svg"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(_SQUARE_SVG)
    second.write_text(_THREE_SHAPE_SVG)

    assert len(mn.SVGMobject(first).submobjects) == 1
    assert len(mn.SVGMobject(second).submobjects) == 3


def test_a_missing_svg_raises_rather_than_replaying_a_stale_entry(
    isolated_svg_cache, tmp_path, monkeypatch
):
    """The cache is consulted before manim resolves the path.

    So a basename key let a deleted file keep "loading" from whatever was
    cached under that name, instead of manim reporting it missing.
    """
    svg_file = tmp_path / "logo.svg"
    svg_file.write_text(_SQUARE_SVG)
    monkeypatch.chdir(tmp_path)
    assert len(mn.SVGMobject("logo.svg").submobjects) == 1

    svg_file.unlink()

    with pytest.raises(OSError):
        mn.SVGMobject("logo.svg")


def test_manim_generated_svgs_stay_keyed_on_their_content_addressed_basename():
    """Tex output must not be keyed on the SVG bytes.

    dvisvgm's output carries its version and emits in its own order, so hashing
    it would key the cache to the machine that produced it -- exactly the
    cross-machine sharing this cache exists for. The basename is already
    ``tex_hash(source)``, so it stays the identity.
    """
    tex_dir = pathlib.Path(mn.config.get_dir("tex_dir"))
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_svg = tex_dir / "0123456789abcdef.svg"
    tex_svg.write_text(_SQUARE_SVG)
    try:
        assert svg_cache._svg_content_id(tex_svg) == "0123456789abcdef.svg"
    finally:
        tex_svg.unlink()

    # Pango text output is named by ``_text2hash`` and lives in ``text_dir``,
    # so it takes the same fast path.
    text_dir = pathlib.Path(mn.config.get_dir("text_dir"))
    text_dir.mkdir(parents=True, exist_ok=True)
    text_svg = text_dir / "fedcba9876543210.svg"
    text_svg.write_text(_SQUARE_SVG)
    try:
        assert svg_cache._svg_content_id(text_svg) == "fedcba9876543210.svg"
    finally:
        text_svg.unlink()

    user_svg = tex_dir.parent / "logo.svg"
    user_svg.write_text(_SQUARE_SVG)
    try:
        assert svg_cache._svg_content_id(user_svg).startswith("logo.svg:")
    finally:
        user_svg.unlink()
