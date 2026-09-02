"""Contracts the authoring surface owes a script that gets something wrong.

Every case here is one a user reaches by writing ordinary scene code and
guessing: ``if mob:``, a runtime that came out negative, Manim's ``run_time``,
``mob.rotate = 90``, ``save_video("renders/")``. Each used to succeed quietly,
or die several frames down in a function the user never typed.

These are feature tests for the guards themselves, so they are deliberately
outside the fast suite -- they fail when the guards are worked on, not when
something elsewhere moves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from algan.animation_timeline.animation_contexts import Lag, Off, Seq, Sync
from algan.errors import AlganConfigurationError, HierarchyError
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Square
from algan.mobs.text import Text
from algan.scene import Scene
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS
from algan.utils.algan_utils import (
    _check_container_is_supported,
    _resolve_output_destination,
)


@pytest.fixture(autouse=True)
def reset_global_authoring_state():
    SceneManager.reset()
    yield
    SceneManager.reset()


# --------------------------------------------------------------------------
# A Mob is a thing, not a count
# --------------------------------------------------------------------------


def test_a_mob_is_truthy_however_many_batch_members_it_has():
    """``if mob:`` must not depend on whether the Mob happens to be batched.

    ``__len__`` reports 0 for an unbatched Mob, and without ``__bool__`` that
    made ``bool(Square())`` False while ``bool(Text("hi"))`` was True -- so
    ``if mob:`` skipped a perfectly good square and ``mob or fallback``
    returned the fallback.
    """
    square = Square()
    text = Text("hi")

    assert bool(square) is True
    assert bool(text) is True
    assert len(square) == 0
    assert (square or "fallback") is square


# --------------------------------------------------------------------------
# Time only moves forward
# --------------------------------------------------------------------------


def test_scene_wait_refuses_a_negative_number_of_seconds():
    """``scene.wait(target - now)`` coming out negative is a realistic bug.

    Left alone it rewound the scene clock and the render silently came out
    two frames long.
    """
    with pytest.raises(AlganConfigurationError, match=r"time=-5"):
        Scene.wait(-5)


@pytest.mark.parametrize(
    "make_context",
    [
        pytest.param(lambda: Seq(runtime=-1), id="Seq-runtime"),
        pytest.param(lambda: Sync(runtime=-0.5), id="Sync-runtime"),
        pytest.param(lambda: Lag(0.5, runtime=-2), id="Lag-runtime"),
        pytest.param(lambda: Seq(runtime_per_part=-1), id="runtime_per_part"),
    ],
)
def test_a_context_refuses_a_negative_runtime(make_context):
    with pytest.raises(AlganConfigurationError, match=r"negative"):
        make_context()


def test_zero_is_still_a_legal_runtime():
    """``Off`` is a zero-length block, so the guard is ``< 0``, not ``<= 0``."""
    with Seq(runtime=0):
        pass
    with Off():
        pass
    Scene.wait(0)


# --------------------------------------------------------------------------
# Timing belongs to the context, not to the call
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda mob: mob.rotate(90, runtime=2), id="rotate"),
        pytest.param(lambda mob: mob.scale(2, runtime=2), id="scale"),
        pytest.param(lambda mob: mob.set_scale(2, runtime=2), id="set_scale"),
        pytest.param(lambda mob: mob.spawn(runtime=2), id="spawn"),
        pytest.param(lambda mob: mob.despawn(runtime=2), id="despawn"),
        pytest.param(lambda mob: mob.become(Square(), runtime=2), id="become"),
    ],
)
def test_runtime_on_a_method_names_the_context_to_wrap_the_call_in(call):
    """The guard used to reach only methods with a ``**kwargs`` passthrough.

    ``mob.move(RIGHT, runtime=2)`` got the helpful message; ``mob.rotate(90,
    runtime=2)`` and ``mob.scale(2, runtime=2)`` got a bare
    ``TypeError: ... unexpected keyword argument``.
    """
    mob = Square()
    with pytest.raises(AlganConfigurationError, match=r"with Seq\(runtime=2\)"):
        call(mob)


def test_the_equalize_runtimes_arm_of_the_guard_actually_fires():
    """It was keyed ``equialize_runtimes``, so it never matched anything."""
    mob = Square()
    with pytest.raises(AlganConfigurationError, match=r"with Sync\(equalize_runtimes"):
        mob.rotate(90, equalize_runtimes=True)


def test_a_zero_runtime_on_a_method_points_at_Off():
    mob = Square()
    with pytest.raises(AlganConfigurationError, match=r"with Off\(\)"):
        mob.rotate(90, runtime=0)


# --------------------------------------------------------------------------
# Names Algan does not have
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "replacement"),
    [
        ({"duration": 1}, "runtime"),
        ({"run_time": 1}, "runtime"),
        ({"rate_func": None}, "easing"),
    ],
)
def test_a_legacy_name_on_a_context_names_the_replacement(kwargs, replacement):
    """``Sync(duration=1)`` used to be an unexpected-keyword TypeError."""
    with pytest.raises(AlganConfigurationError, match=replacement):
        Sync(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "replacement"),
    [
        ({"duration": 1}, "runtime"),
        ({"run_time": 1}, "runtime"),
        ({"rate_func": None}, "easing"),
    ],
)
def test_a_legacy_name_on_a_mob_method_names_the_replacement(kwargs, replacement):
    """``sq.move(RIGHT, duration=1)`` used to die inside ``set_location``."""
    from algan.constants.spatial import RIGHT

    mob = Square()
    with pytest.raises(AlganConfigurationError, match=replacement):
        mob.move(RIGHT, **kwargs)
    with pytest.raises(AlganConfigurationError, match=replacement):
        mob.rotate(90, **kwargs)


@pytest.mark.parametrize("name", ["duration", "run_time", "rate_func"])
def test_a_legacy_name_on_scene_wait_names_the_replacement(name):
    with pytest.raises(AlganConfigurationError):
        Scene.wait(**{name: 1})


def test_an_unknown_keyword_on_scene_wait_is_still_a_plain_type_error():
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'seconds'"):
        Scene.wait(seconds=1)


# --------------------------------------------------------------------------
# Verbs are methods; hierarchy links are written at both ends
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("move", "RIGHT"),
        ("move_to", "ORIGIN"),
        ("rotate", 90),
        ("scale", 2),
        ("become", None),
        ("spawn", True),
        ("despawn", True),
    ],
)
def test_assigning_to_a_verb_method_raises_instead_of_shadowing_it(name, value):
    """``mob.rotate = 90`` read like an attribute write and silently was one."""
    mob = Square()
    with pytest.raises(AttributeError, match=rf"'{name}' is a method"):
        setattr(mob, name, value)
    # The method still reads back as a bound method afterwards.
    assert callable(getattr(mob, name))


def test_children_cannot_be_rebound_but_the_hierarchy_methods_still_work():
    """``p.children = []`` left every ex-child still naming ``p`` as a parent."""
    child = Square()
    parent = Group(child)

    with pytest.raises(HierarchyError, match=r"replace_children"):
        parent.children = []
    with pytest.raises(HierarchyError, match=r"add_parent"):
        child.parents = []

    assert parent.children == [child]
    assert child.parents == [parent]

    parent.remove_child(child)
    assert parent.children == []
    assert child.parents == []


def test_replace_children_is_the_supported_way_to_swap_the_whole_set():
    first, second = Square(), Square()
    parent = Group(first)
    parent.replace_children([second])

    assert parent.children == [second]
    assert first.parents == []
    assert second.parents == [parent]


# --------------------------------------------------------------------------
# Group's link_children is a public parameter with a public name
# --------------------------------------------------------------------------


def test_group_link_children_has_no_leading_underscore():
    import inspect

    parameters = inspect.signature(Group.__init__).parameters
    assert "link_children" in parameters
    assert "_link_children" not in parameters


def test_group_link_children_false_makes_an_unlinked_view():
    members = [Square(), Square()]
    view = Group(*members, link_children=False)

    assert view.children == members
    assert all(member.parents == [] for member in members)

    owning = Group(*members)
    assert all(member.parents == [owning] for member in members)


def test_a_group_slice_is_still_an_unlinked_view():
    """The internal caller of the renamed parameter, exercised end to end."""
    members = [Square() for _ in range(3)]
    group = Group(*members)
    sliced = group[1:]

    assert sliced.children == members[1:]
    assert all(member.parents == [group] for member in members[1:])


# --------------------------------------------------------------------------
# Where the file lands
# --------------------------------------------------------------------------


@pytest.fixture
def isolated_output(tmp_path):
    """Point Algan's output defaults at ``tmp_path`` and restore afterwards."""
    snapshot = SETTINGS.snapshot()
    SETTINGS.paths.set(
        output_root=str(tmp_path),
        output_directory="algan_outputs",
        output_filename="named_after_the_script",
    )
    try:
        yield tmp_path
    finally:
        SETTINGS.restore(snapshot)


def test_every_resolved_output_path_is_absolute(isolated_output, monkeypatch):
    """``RenderResult.output_path`` is the answer to "where did it go?"."""
    monkeypatch.chdir(isolated_output)
    for requested in ("bare", "nope/deeper/a.mp4", "./here.mp4", None):
        resolved = _resolve_output_destination(requested, ".mp4")
        assert resolved.is_absolute(), requested


def test_a_bare_name_lands_in_the_output_directory(isolated_output):
    resolved = _resolve_output_destination("intro", ".mp4")
    assert resolved == isolated_output / "algan_outputs" / "intro.mp4"


def test_an_existing_directory_is_a_directory_to_write_into(
    isolated_output, monkeypatch
):
    """The rule ``algan render -o`` already applied; ``save_video`` did not."""
    monkeypatch.chdir(isolated_output)
    (isolated_output / "renders").mkdir()

    resolved = _resolve_output_destination("renders", ".mp4")
    assert resolved == isolated_output / "renders" / "named_after_the_script.mp4"


def test_a_trailing_separator_is_a_directory_to_write_into(
    isolated_output, monkeypatch
):
    """``save_video("adir2/")`` used to drop the directory and write adir2.mp4."""
    monkeypatch.chdir(isolated_output)

    resolved = _resolve_output_destination("adir2/", ".mp4")
    assert resolved == isolated_output / "adir2" / "named_after_the_script.mp4"
    assert resolved.parent.is_dir()


def test_a_named_file_keeps_its_own_name_even_with_no_suffix(
    isolated_output, monkeypatch
):
    """The CLI also treats "no suffix" as a directory; the resolver must not.

    ``save_video("intro")`` names a file called ``intro``, which is the most
    common call in the documentation.
    """
    monkeypatch.chdir(isolated_output)
    resolved = _resolve_output_destination("out/intro", ".mp4")
    assert resolved == isolated_output / "out" / "intro.mp4"


def test_a_placeholder_main_file_resolves_like_no_script_at_all(monkeypatch):
    """Piping a script into Python sets ``__file__`` to ``<stdin>``.

    Taking a stem from that produced ``algan_outputs/<stdin>.mp4`` -- a name
    Windows refuses outright.
    """
    import sys

    from algan.settings.path_settings import _main_script_path, output_filename_for

    main = sys.modules["__main__"]
    for placeholder in ("<stdin>", "<string>", "<ipython-input-3-abcdef>"):
        monkeypatch.setattr(main, "__file__", placeholder, raising=False)
        assert _main_script_path() is None
        assert output_filename_for(_main_script_path()) == "algan_render_output"


@pytest.mark.parametrize("suffix", [".xyz", ".txt", ".mp3"])
def test_an_unwritable_video_container_is_rejected_before_the_render(suffix):
    """It used to cost a whole render, then surface as a missing temp file."""
    with pytest.raises(AlganConfigurationError, match=r"video container"):
        _check_container_is_supported(Path(f"weird{suffix}"))


def test_an_unwritable_still_format_is_rejected_before_the_render():
    with pytest.raises(AlganConfigurationError, match=r"still-image format"):
        _check_container_is_supported(Path("frame.mp4"), still=True)


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".webm", ".MP4"])
def test_the_containers_algan_writes_are_accepted(suffix):
    _check_container_is_supported(Path(f"video{suffix}"))


def test_save_video_rejects_a_bad_container_without_rendering(
    isolated_output, monkeypatch
):
    """The check has to run before anything is rendered, not after."""
    scene = SceneManager.instance().current_scene
    Square(scene=scene).spawn()

    def fail(*_args, **_kwargs):
        raise AssertionError("the render started despite an unwritable container")

    monkeypatch.setattr(scene, "_render_to_video", fail)

    with pytest.raises(AlganConfigurationError, match=r"video container"):
        scene.save_video("broken.xyz")


def test_audio_is_not_the_only_thing_a_wav_extension_could_mean():
    """A ``.wav`` target is a container mistake, not an audio-track request."""
    with pytest.raises(AlganConfigurationError, match=r"video container"):
        _check_container_is_supported(Path("narration.wav"))
