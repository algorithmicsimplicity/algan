"""Fast tests for the pure helpers in ``algan.utils.docbuild``."""

from __future__ import annotations

import os
import textwrap
import time

import pytest

pytest.importorskip("docutils")

from algan.utils.docbuild import algan_directive, module_parsing  # noqa: E402
from algan.utils.docbuild.autoaliasattr_directive import smart_replace  # noqa: E402

pytestmark = pytest.mark.fast


@pytest.fixture
def parse(tmp_path, monkeypatch):
    """Parse synthetic modules written under ``tmp_path``."""
    monkeypatch.setattr(module_parsing, "MANIM_ROOT", tmp_path)
    monkeypatch.setattr(module_parsing, "ALIAS_DOCS_DICT", {})
    monkeypatch.setattr(module_parsing, "DATA_DICT", {})
    monkeypatch.setattr(module_parsing, "TYPEVAR_DICT", {})

    def _parse(files: dict[str, str]):
        for rel, src in files.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(src), encoding="utf-8")
        return module_parsing.parse_module_attributes()

    return _parse


# --- smart_replace ---------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "alias", "sub", "expected"),
    [
        ("Point", "Point", "X", "X"),
        ("Point | Point3D", "Point", "X", "X | Point3D"),
        ("list[Point]", "Point", "X", "list[X]"),
        ("MyPoint", "Point", "X", "MyPoint"),
        ("Point_2", "Point", "X", "Point_2"),
        ("_Point", "Point", "X", "_Point"),
        ("Point, Point", "Point", "X", "X, X"),
        ("", "Point", "X", ""),
        ("int", "Point", "X", "int"),
        ("PointPoint Point", "Point", "Q", "PointPoint Q"),
        ("a.Point", "Point", "LongerSubstitution", "a.LongerSubstitution"),
    ],
)
def test_smart_replace(base, alias, sub, expected):
    assert smart_replace(base, alias, sub) == expected


def test_smart_replace_substitution_containing_alias_not_rereplaced():
    assert smart_replace("A | A", "A", "A_A") == "A_A | A_A"


# --- process_name_list -----------------------------------------------------


def test_process_name_list_basic():
    result = algan_directive.process_name_list("Foo Bar", "class")
    assert result == [":class:`~.Foo`", ":class:`~.Bar`"]


def test_process_name_list_extra_whitespace_and_empty():
    assert algan_directive.process_name_list("  a\n b  ", "func") == [
        ":func:`~.a`",
        ":func:`~.b`",
    ]
    assert algan_directive.process_name_list("", "meth") == []


# --- _next_occurrence ------------------------------------------------------


def test_next_occurrence_counts_per_name(monkeypatch):
    monkeypatch.setattr(algan_directive, "_class_occurrences", {})
    assert algan_directive._next_occurrence("A") == 1
    assert algan_directive._next_occurrence("A") == 2
    assert algan_directive._next_occurrence("B") == 1
    assert algan_directive._next_occurrence("A") == 3


# --- _find_video -----------------------------------------------------------


def test_find_video_prefers_named_output(tmp_path):
    (tmp_path / "Scene.mp4").write_bytes(b"")
    (tmp_path / "other.mp4").write_bytes(b"")
    assert algan_directive._find_video(tmp_path, "Scene", 0.0) == tmp_path / "Scene.mp4"


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".webm"])
def test_find_video_named_any_suffix(tmp_path, suffix):
    (tmp_path / f"S{suffix}").write_bytes(b"")
    assert algan_directive._find_video(tmp_path, "S", time.time() + 100) == (
        tmp_path / f"S{suffix}"
    )


def test_find_video_mp4_preferred_over_webm(tmp_path):
    (tmp_path / "S.webm").write_bytes(b"")
    (tmp_path / "S.mp4").write_bytes(b"")
    assert algan_directive._find_video(tmp_path, "S", 0.0).suffix == ".mp4"


def test_find_video_falls_back_to_single_new_video(tmp_path):
    old = tmp_path / "old.mp4"
    old.write_bytes(b"")
    os.utime(old, (1000, 1000))
    new = tmp_path / "my_video.mov"
    new.write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert algan_directive._find_video(tmp_path, "Scene", 2000.0) == new


def test_find_video_ignores_directories_with_video_suffix(tmp_path):
    (tmp_path / "dir.mp4").mkdir()
    new = tmp_path / "v.webm"
    new.write_bytes(b"")
    assert algan_directive._find_video(tmp_path, "Scene", 0.0) == new


def test_find_video_several_new_videos_raises(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="a.mp4, b.mp4"):
        algan_directive._find_video(tmp_path, "Scene", 0.0)


def test_find_video_none_raises(tmp_path):
    old = tmp_path / "old.mp4"
    old.write_bytes(b"")
    os.utime(old, (1000, 1000))
    with pytest.raises(FileNotFoundError, match="did not create a video"):
        algan_directive._find_video(tmp_path, "Scene", 2000.0)


# --- _write_rendering_stats ------------------------------------------------


def test_write_rendering_stats(tmp_path, monkeypatch):
    csv_path = tmp_path / "times.csv"
    monkeypatch.setattr(algan_directive, "rendering_times_file_path", csv_path)
    algan_directive._write_rendering_stats("MyScene", 1.23456, "reference/algan.foo")
    algan_directive._write_rendering_stats("Other", 2.0, "tutorials/intro")
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["foo,MyScene,1.235", "tutorials/intro,Other,2.000"]


# --- parse_module_attributes -----------------------------------------------


def test_parse_type_alias_with_category_and_doc(parse):
    aliases, data, typevars = parse(
        {
            "pkg/typing.py": '''
            from typing import TypeAlias
            """[CATEGORY] Points"""
            Point: TypeAlias = "np.ndarray"
            """A point."""
            Vec: TypeAlias = npt.NDArray
            ''',
        }
    )
    assert aliases == {
        "pkg.typing": {
            "Points": {
                "Point": {"definition": "'np.ndarray'", "doc": "A point."},
                "Vec": {"definition": "NDArray"},
            }
        }
    }
    assert data == {}
    assert typevars == {}


def test_parse_alias_without_category_goes_to_empty_key(parse):
    aliases, _, _ = parse({"m.py": "X: TypeAlias = int\n"})
    assert aliases == {"m": {"": {"X": {"definition": "int"}}}}


def test_parse_union_converted_to_bar(parse):
    aliases, _, _ = parse({"m.py": "U: TypeAlias = Union[int, str, None]\n"})
    assert aliases["m"][""]["U"]["definition"] == "int | str | None"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: single-member Union[int] has no .elts, parse raises AttributeError",
)
def test_parse_single_member_union(parse):
    aliases, _, _ = parse({"m.py": "U: TypeAlias = Union[int]\n"})
    assert aliases["m"][""]["U"]["definition"] == "int"


@pytest.mark.parametrize("test", ["TYPE_CHECKING", "typing.TYPE_CHECKING"])
def test_parse_alias_under_type_checking(parse, test):
    aliases, _, _ = parse({"m.py": f"if {test}:\n    A: TypeAlias = float\n"})
    assert aliases == {"m": {"": {"A": {"definition": "float"}}}}


def test_parse_alias_under_other_if_ignored(parse):
    aliases, _, _ = parse({"m.py": "if DEBUG:\n    A: TypeAlias = float\n"})
    assert aliases == {}


def test_parse_typevar_escapes_underscores(parse):
    _, _, typevars = parse({"m.py": 'T_co = TypeVar("T_co", covariant=True)\n'})
    assert typevars == {"m": {"T_co": r"TypeVar('T\_co', covariant=True)"}}


def test_parse_documented_annotated_attribute_is_data(parse):
    _, data, _ = parse(
        {
            "m.py": '''
            DOCUMENTED: int = 1
            """Docs."""
            UNDOCUMENTED: int = 2
            PLAIN = 3
            """Plain assignments are not collected."""
            ''',
        }
    )
    assert data == {"m": ["DOCUMENTED"]}


def test_parse_module_name_from_package_init(parse):
    _, data, _ = parse({"a/b/__init__.py": 'X: int = 1\n"""d"""\n'})
    assert data == {"a.b.__init__": ["X"]}


def test_parse_skips_external_libraries(parse):
    aliases, _, _ = parse(
        {
            "external_libraries/manim/typing.py": "X: TypeAlias = int\n",
            "keep.py": "Y: TypeAlias = int\n",
        }
    )
    assert list(aliases) == ["keep"]


def test_parse_is_cached(parse, tmp_path):
    first = parse({"m.py": "X: TypeAlias = int\n"})
    (tmp_path / "n.py").write_text("Y: TypeAlias = int\n", encoding="utf-8")
    second = module_parsing.parse_module_attributes()
    assert second[0] is first[0]
    assert "n" not in second[0]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: data_name is not reset after its docstring, so a second "
    "string literal records the attribute twice",
)
def test_parse_attribute_followed_by_two_strings_recorded_once(parse):
    _, data, _ = parse({"m.py": 'X: int = 1\n"""doc"""\n"""unrelated"""\n'})
    assert data == {"m": ["X"]}
