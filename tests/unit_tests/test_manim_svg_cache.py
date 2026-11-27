"""Round-trip guarantees for the persistent Manim SVG/Tex geometry cache.

The cache replaces Manim's ``SVGMobject.init_svg_mobject``, so a cache hit has
to reproduce everything the running Manim reads back off the mobject -- not just
the glyph tree. Manim 0.21 also reads ``id_to_vgroup_dict``
(``MathTex._break_up_by_substrings`` indexes it to split a formula into its
``tex_strings``), which a hit used to leave empty: the first ``Tex`` on a machine
succeeded and every later one died on ``KeyError: 'root'``.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

import manim as mn
import numpy as np
import pytest
import torch

import algan
from algan.mobs import text as text_module
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


@pytest.mark.parametrize(
    "options",
    [
        {"text": "affine η 0.25", "font_size": 24},
        {
            "text": "affine\n0.25",
            "font_size": 32,
            "disable_ligatures": True,
            "color_map": {"affine": algan.RED},
            "weight": "BOLD",
            "stroke_width": 1,
        },
        {"text": "gradient", "gradient": (algan.BLUE, algan.YELLOW)},
    ],
)
def test_native_pango_text_matches_uncached_on_cold_memory_and_disk_hits(
    isolated_svg_cache, monkeypatch, options
):
    if not hasattr(mn, "Text"):
        pytest.skip("Pango is unavailable")

    def snapshot():
        # Tex keeps each source's outlines in memory above this cache; clear
        # it so every stage reaches the SVG layer under test.
        text_module._TEX_GLYPH_MEMO.clear()
        with algan.Scene():
            text = algan.Text(**options)
            values = [len(text.character_mobs)]
            for glyph in text.character_mobs:
                values.extend(
                    t.detach().clone()
                    for t in (
                        glyph.control_points.location,
                        glyph.color,
                        glyph.stroke_color,
                        glyph.stroke_width,
                        glyph.opacity,
                    )
                )
            # Mutation of one instance must not poison the reusable recipe.
            with algan.Off():
                text.move(algan.RIGHT).scale(2)
                text.color = algan.GREEN
            return values

    original = mn.SVGMobject.init_svg_mobject
    with monkeypatch.context() as patch:
        patch.setattr(
            mn.SVGMobject,
            "init_svg_mobject",
            lambda self, use_svg_cache: original(self, False),
        )
        expected = snapshot()
    for stage in ("cold", "memory", "disk"):
        if stage == "disk":
            svg_cache._MEM_CACHE.clear()
        actual = snapshot()
        assert svg_cache._MEM_CACHE, "Native Text must use the SVG recipe cache"
        assert actual[0] == expected[0]
        for got, want in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(got, want, atol=0, rtol=0, msg=stage)


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


def test_manimpango_version_changes_generated_text_cache_namespace(monkeypatch):
    import manimpango

    monkeypatch.setattr(manimpango, "__version__", "emoji-backend-v1")
    first = svg_cache._manimpango_cache_version()
    monkeypatch.setattr(manimpango, "__version__", "emoji-backend-v2")
    second = svg_cache._manimpango_cache_version()

    assert first != second


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


def test_pre_emoji_recipe_tag_is_reparsed():
    """v2 recipes can predate SVG ``<image>`` support and must be discarded."""
    source, _ = _container_with_groups()
    current = svg_cache._extract(source)
    old = ("algan-manim-svg-recipe-v2", current[1], current[2])

    nodes, groups = svg_cache._parse_recipe(old)

    assert groups is None
    assert nodes == old
    assert svg_cache._RECIPE_TAG == "algan-manim-svg-recipe-v4"


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


def test_pango_backend_upgrade_regenerates_svg_and_emoji_image(monkeypatch, tmp_path):
    """A ManimPango upgrade must not replay SVG/geometry from the old backend.

    Color emoji are raster glyphs in Cairo's SVG output. Algan's vendored
    ``SVGMobject`` turns the resulting ``<image>`` into an ``ImageMobject``,
    which :class:`algan.Text` then converts to an ``ImageMob``. This simulates
    upgrading from a backend that dropped that image to one that emits it while
    keeping the authored text identical.
    """
    import manimpango

    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8f4LhPwMD"
        "AwMTAxQAACuXAsoUVo9TAAAAAElFTkSuQmCC"
    )
    old_cache = SETTINGS.paths.cache_directory
    saved_memo = dict(svg_cache._MEM_CACHE)
    svg_cache._MEM_CACHE.clear()
    calls = []

    def fake_text2svg(*args, **kwargs):
        del kwargs
        output = pathlib.Path(args[4])
        output.parent.mkdir(parents=True, exist_ok=True)
        version = manimpango.__version__
        calls.append(version)
        image = ""
        if version == "emoji-fixed":
            image = (
                '<image x="18" y="0" width="12" height="12" '
                f'href="data:image/png;base64,{png}"/>'
            )
        output.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="16" '
            'viewBox="0 0 48 16">'
            '<path d="M0 12 L5 0 L10 12 Z" fill="#fff"/>'
            f"{image}"
            '<path d="M36 0 L46 0 L46 12 L36 12 Z" fill="#fff"/>'
            "</svg>"
        )
        return str(output)

    try:
        SETTINGS.paths.set(cache_directory=str(tmp_path))
        monkeypatch.setattr(manimpango, "text2svg", fake_text2svg)

        monkeypatch.setattr(manimpango, "__version__", "emoji-broken")
        broken = algan.Text("A😀B")
        assert broken.image_mobs == []

        monkeypatch.setattr(manimpango, "__version__", "emoji-fixed")
        fixed = algan.Text("A😀B")
        assert len(fixed.image_mobs) == 1
        assert calls == ["emoji-broken", "emoji-fixed"]
    finally:
        SETTINGS.paths.set(cache_directory=old_cache)
        svg_cache._MEM_CACHE.clear()
        svg_cache._MEM_CACHE.update(saved_memo)


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


def test_manim_generated_svgs_stay_content_addressed_without_hashing_svg_bytes():
    """Generated SVG output must not be keyed on the SVG bytes.

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

    # Pango text output is named by ``_text2hash`` too, but that hash omits the
    # backend version. Keep the fast basename path while namespacing it by the
    # ManimPango version so renderer upgrades cannot replay stale geometry.
    text_dir = pathlib.Path(mn.config.get_dir("text_dir"))
    text_dir.mkdir(parents=True, exist_ok=True)
    text_svg = text_dir / "fedcba9876543210.svg"
    text_svg.write_text(_SQUARE_SVG)
    try:
        assert svg_cache._svg_content_id(text_svg) == (
            f"pango-{svg_cache._manimpango_cache_version()}:fedcba9876543210.svg"
        )
    finally:
        text_svg.unlink()

    user_svg = tex_dir.parent / "logo.svg"
    user_svg.write_text(_SQUARE_SVG)
    try:
        assert svg_cache._svg_content_id(user_svg).startswith("logo.svg:")
    finally:
        user_svg.unlink()


def test_a_cache_hit_reproduces_glyph_geometry_bit_for_bit(isolated_svg_cache):
    """A replayed recipe is the same bits as the parse it stands in for.

    ``test_tex_survives_a_cache_hit`` proves the *structure* survives; this is
    the half that a render can see. Glyph outlines meet the rasterizer through
    analytic-AA coverage, so a few ULPs of control-point position move whole
    channel values wherever a coverage sample flips side of an edge -- which is
    how the sibling tessellation cache's origin bug showed up as an 18-channel
    cold-vs-warm split in ``text_and_media``. Nothing may separate the cold
    parse from the warm replay, in-process memo or disk.
    """

    def glyph_points(mob):
        family = list(mob.family_members_with_points())
        assert family, "the fixture produced no glyphs"
        return np.concatenate([np.asarray(part.points) for part in family])

    cold = glyph_points(mn.MathTex(r"\frac{a^2}{b}+\sqrt{x}"))
    assert svg_cache._MEM_CACHE, "the cold parse was not cached"

    warm_from_memo = glyph_points(mn.MathTex(r"\frac{a^2}{b}+\sqrt{x}"))

    svg_cache._MEM_CACHE.clear()
    warm_from_disk = glyph_points(mn.MathTex(r"\frac{a^2}{b}+\sqrt{x}"))

    for label, warm in (("memo", warm_from_memo), ("disk", warm_from_disk)):
        assert warm.shape == cold.shape, f"{label} replay changed the glyph count"
        assert np.array_equal(warm, cold), (
            f"the {label} replay differs from the cold parse by "
            f"{np.abs(warm - cold).max():.3e}"
        )


def test_a_replayed_svg_logs_what_its_parse_logged(isolated_svg_cache):
    """A replay stands in for the parse, so it says what the parse said.

    Manim cannot import an SVG ``<text>`` element and logs that as it parses.
    A replayed recipe is not parsed, so once an SVG was cached -- in memory,
    and on disk for every later run -- its text was left out without a word.
    """
    path = isolated_svg_cache / "label.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        'viewBox="0 0 100 100">'
        '<rect x="10" y="10" width="80" height="50" fill="#3080ff"/>'
        '<text x="20" y="85" font-size="16">Label</text>'
        "</svg>"
    )

    class Recorder(logging.Handler):
        def __init__(self):
            super().__init__()
            self.said = []

        def emit(self, record):
            self.said.append((record.levelno, record.getMessage()))

    def logged_by_import():
        recorder = Recorder()
        mn.logger.addHandler(recorder)
        try:
            mn.SVGMobject(str(path))
        finally:
            mn.logger.removeHandler(recorder)
        return recorder.said

    parsed = logged_by_import()
    assert [message for _, message in parsed] == [
        "SVG text is not imported, so 'Label' is left out. To keep it, convert "
        "the text to paths in the SVG (Inkscape: Path > Object to Path; "
        "Illustrator: Type > Create Outlines)."
    ]
    assert logged_by_import() == parsed, "the replay from memory said nothing"
    svg_cache._MEM_CACHE.clear()
    assert logged_by_import() == parsed, "the replay from disk said nothing"


def test_the_in_process_recipe_memo_is_bounded(monkeypatch):
    # It kept every recipe a process ever parsed: a label built per frame grew
    # it without limit. Past its size the oldest half goes.
    monkeypatch.setattr(svg_cache, "_MEM_CACHE", {})
    monkeypatch.setattr(svg_cache, "_MEM_CACHE_SIZE", 8)
    for index in range(20):
        svg_cache._memoize(f"key{index}", ("recipe", index))
        assert len(svg_cache._MEM_CACHE) <= 8
    assert "key19" in svg_cache._MEM_CACHE
    assert "key0" not in svg_cache._MEM_CACHE


class _RunsCodeWhenUnpickled:
    marker = None

    def __reduce__(self):
        return (pathlib.Path.touch, (pathlib.Path(self.marker),))


def test_a_cache_file_is_never_unpickled(isolated_svg_cache, tmp_path):
    # The disk layer used torch.load(weights_only=False), so anyone who could
    # write into the cache directory ran code in the next process to typeset
    # the matching string.
    marker = tmp_path / "ran"
    payload = _RunsCodeWhenUnpickled()
    payload.marker = os.fspath(marker)
    svg_cache._cache_dir().mkdir(parents=True, exist_ok=True)
    for path in (
        svg_cache._disk_path("planted"),
        svg_cache._legacy_disk_path("planted"),
        svg_cache._cache_dir() / "planted.pt",
    ):
        torch.save(payload, path)
        assert svg_cache._load_disk("planted") is None
        path.unlink()
    assert not marker.exists()


@pytest.mark.parametrize(
    "spelling", ["|O", "<U4", "[('a', '<f8')]", "(2,)<f8", "<M8[s]"]
)
def test_a_recipe_file_may_carry_only_plain_number_arrays(isolated_svg_cache, spelling):
    # np.frombuffer over an object dtype would read raw bytes as pointers.
    path = svg_cache._disk_path("planted")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {"tree": ["ndarray", 0], "arrays": [[spelling, [2], "C", 0, 16]]}
    ).encode()
    start = len(svg_cache._DISK_MAGIC) + 8 + len(header)
    path.write_bytes(
        svg_cache._DISK_MAGIC
        + len(header).to_bytes(8, "little")
        + header
        + b"\x00" * (-start % svg_cache._ARRAY_ALIGNMENT)
        + b"\x01" * 16
    )
    assert svg_cache._load_disk("planted") is None


def test_a_data1_entry_is_carried_over_with_the_same_arrays(isolated_svg_cache):
    # A cache filled before data2 must not be re-parsed glyph by glyph.
    text = mn.Text("carried over", use_svg_cache=True)
    [path] = sorted(svg_cache._cache_dir().glob(f"*{svg_cache._DISK_SUFFIX}"))
    key = path.name.split(".")[0]
    recipe = svg_cache._read_disk_file(path)
    path.unlink()

    # The data1 encoding: the same tree, every array a tensor.
    def as_data1(value):
        if type(value) is list and value and value[0] == "ndarray":
            return ["ndarray", torch.from_numpy(arrays[value[1]])]
        if type(value) is list:
            return [as_data1(item) for item in value]
        return value

    arrays = []
    tree = svg_cache._encode(recipe, arrays)
    torch.save(as_data1(tree), svg_cache._legacy_disk_path(key))

    svg_cache._MEM_CACHE.clear()
    again = mn.Text("carried over", use_svg_cache=True)
    assert path.exists()  # saved again in the current format
    for before, after in zip(
        text.family_members_with_points(), again.family_members_with_points()
    ):
        assert type(before) is type(after)
        assert before.points.tobytes() == after.points.tobytes()
        assert before.fill_rgbas.tobytes() == after.fill_rgbas.tobytes()


@pytest.mark.parametrize(
    "node",
    [
        ["object", "os", "system", ["dict", []]],
        ["object", "builtins", "object", ["dict", []]],
        ["object", "manim.utils.file_ops", "Path", ["dict", []]],
        ["enum", "subprocess", "Popen", 0],
    ],
)
def test_a_recipe_naming_a_class_outside_vendored_manim_is_refused(node):
    with pytest.raises(svg_cache._NotPlainData):
        svg_cache._decode(node)


def test_recipes_are_saved_as_plain_data_and_rebuild_the_same_glyphs(
    isolated_svg_cache,
):
    text = mn.Text("plain data", use_svg_cache=True)
    [path] = sorted(svg_cache._cache_dir().glob(f"*{svg_cache._DISK_SUFFIX}"))
    assert path.name.endswith(f".{svg_cache._DISK_FORMAT}{svg_cache._DISK_SUFFIX}")
    # JSON and raw array bytes: no pickle anywhere in the file.
    assert path.read_bytes().startswith(svg_cache._DISK_MAGIC)
    assert b"\x80\x02" not in path.read_bytes()[: len(svg_cache._DISK_MAGIC) + 8]

    svg_cache._MEM_CACHE.clear()
    again = mn.Text("plain data", use_svg_cache=True)
    for before, after in zip(
        text.family_members_with_points(), again.family_members_with_points()
    ):
        assert type(before) is type(after)
        assert before.points.tobytes() == after.points.tobytes()
        assert before.fill_rgbas.tobytes() == after.fill_rgbas.tobytes()


def test_two_saves_of_one_key_write_through_different_temporary_files(
    isolated_svg_cache, monkeypatch
):
    # The temporary name was ``{key}.pt.{pid}.tmp``, shared by every thread of
    # a process saving the same key.
    written = []
    original_write = svg_cache._write_disk_file

    def watching_write(path, recipe):
        written.append(os.fspath(path))
        return original_write(path, recipe)

    monkeypatch.setattr(svg_cache, "_write_disk_file", watching_write)
    svg_cache._save_disk("samekey", (svg_cache._RECIPE_TAG, (), None, ()))
    svg_cache._save_disk("samekey", (svg_cache._RECIPE_TAG, (), None, ()))

    assert len(set(written)) == 2
    assert os.fspath(svg_cache._disk_path("samekey")) not in written
    assert svg_cache._load_disk("samekey") == (svg_cache._RECIPE_TAG, (), None, ())
