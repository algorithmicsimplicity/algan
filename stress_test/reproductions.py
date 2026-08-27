"""Minimal reproductions for the findings in ``STRESS_TEST_FINDINGS.md``.

Run it with the project venv::

    uv run python stress_test/reproductions.py           # every check
    uv run python stress_test/reproductions.py --list     # just the ids
    uv run python stress_test/reproductions.py F1 F3      # selected checks

Each check prints ``REPRO <id> <BUG|FIXED> -- <one line>``. ``BUG`` means the
behaviour described in the report still reproduces; ``FIXED`` means it does
not, so the check is the regression test for that finding.

Every check renders at ``SMOKE_TEST`` (32x32, 2 fps) so the whole file runs in
a couple of minutes on CPU.
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import cv2  # noqa: E402
import torch  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.animation_timeline import animation_contexts as _contexts  # noqa: E402

CHECKS = {}


def check(ident, title):
    def deco(fn):
        CHECKS[ident] = (title, fn)
        return fn

    return deco


def _video_seconds(path):
    cap = cv2.VideoCapture(str(path))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames / fps if fps else 0.0


def _raised(fn):
    """Return the exception ``fn`` raises, or None."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


def _tmp(name):
    return os.path.join(OUT_DIR, name)


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_repro_out")


# --------------------------------------------------------------------------
# F1 -- re-entering a context object leaks the animation-manager override
# --------------------------------------------------------------------------
@check("F1", "re-entering an AnimationContext makes every later Sync sequential")
def f1():
    def sync_seconds():
        with Scene(video_settings=SMOKE_TEST):
            square = Square(color=BLUE).spawn(animate=False)
            with Sync():
                square.move(RIGHT * 0.1)
                square.move(RIGHT * 0.1)
                square.move(RIGHT * 0.1)
            Scene.save_video(_tmp("f1.mp4"), SMOKE_TEST)
        return _video_seconds(_tmp("f1.mp4"))

    before = sync_seconds()

    with Scene():
        square = Square().spawn()
        context = Sync()
        with context:  # noqa: SIM117 -- the nesting is the bug being reproduced
            with context:  # same object, entered while already entered
                square.move(RIGHT)

    after = sync_seconds()
    leaked = _contexts._ANIMATION_MANAGER_OVERRIDE.get(None) is not None
    bug = leaked or after != before
    return bug, (
        f"Sync of 3 lasts {before:.2f}s before the re-entry and {after:.2f}s after "
        f"(override leaked: {leaked})"
    )


# --------------------------------------------------------------------------
# F2 -- creating a Mob inside an updater crashes the render
# --------------------------------------------------------------------------
@check("F2", "constructing a Mob inside an updater crashes the render")
def f2():
    def build():
        with Scene(video_settings=SMOKE_TEST):
            square = Square(color=BLUE).spawn()
            square.add_updater(lambda mob, t: Circle(radius=0.1, color=RED))
            Scene.wait(1)
            Scene.save_frame(_tmp("f2.png"))

    exc = _raised(build)
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "renders"


@check("F2b", "DecimalNumber.set_value() inside an updater crashes the render")
def f2b():
    def build():
        with Scene(video_settings=SMOKE_TEST):
            number = DecimalNumber(0).spawn()
            tracker = ValueTracker(0).spawn()
            number.add_updater(lambda mob, t: mob.set_value(tracker.get_value()))
            tracker.set_value(5)
            Scene.save_frame(_tmp("f2b.png"))

    exc = _raised(build)
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "renders"


# --------------------------------------------------------------------------
# F3 -- save_video ignores the Scene's own video settings
# --------------------------------------------------------------------------
@check("F3", "save_video() ignores Scene(video_settings=...); save_frame() honours it")
def f3():
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(color=BLUE).spawn()
        frame = Scene.save_frame(_tmp("f3.png"))
        video = Scene.save_video(_tmp("f3.mp4"))
        wanted = scene.video_settings.resolution

    png = cv2.imread(str(frame.output_path), cv2.IMREAD_UNCHANGED)
    cap = cv2.VideoCapture(str(video.output_path))
    got = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return got != tuple(wanted), (
        f"Scene asked for {tuple(wanted)}; save_frame wrote {png.shape[1]}x{png.shape[0]}, "
        f"save_video wrote {got[0]}x{got[1]}"
    )


# --------------------------------------------------------------------------
# F4 -- the README quickstart does not run
# --------------------------------------------------------------------------
@check("F4", "README.md quickstart calls three names that do not exist")
def f4():
    with Scene(video_settings=SMOKE_TEST):
        sphere = Sphere(color=BLUE, radius=1.2)
        broken = []
        if _raised(lambda: sphere.spawn(duration=1.0)):
            broken.append("spawn(duration=)")
        if _raised(lambda: Sync(duration=2.0)):
            broken.append("Sync(duration=)")
        if _raised(lambda: sphere.shift([2, 0, 0])):
            broken.append("Mob.shift()")
    return bool(broken), "unsupported in the README example: " + ", ".join(broken)


# --------------------------------------------------------------------------
# F5 -- colour arguments only accept Color
# --------------------------------------------------------------------------
@check("F5", "Mob colour rejects hex strings, hex ints and RGB tuples")
def f5():
    with Scene(video_settings=SMOKE_TEST):
        rejected = []
        for label, call in (
            ('Square(color="#ff0000")', lambda: Square(color="#ff0000")),
            ('Square(color="red")', lambda: Square(color="red")),
            ("Square(color=0xFF0000)", lambda: Square(color=0xFF0000)),
            ("Square(color=(1, 0, 0))", lambda: Square(color=(1, 0, 0))),
            (
                'mob.color = "#ff0000"',
                lambda: setattr(Square().spawn(), "color", "#ff0000"),
            ),
        ):
            exc = _raised(call)
            if exc is not None:
                rejected.append(f"{label} -> {type(exc).__name__}")
    return bool(rejected), "; ".join(rejected) or "all accepted"


# --------------------------------------------------------------------------
# F6 -- spawn() after despawn() is a silent no-op
# --------------------------------------------------------------------------
@check("F6", "spawn() after despawn() silently does nothing")
def f6():
    with Scene(video_settings=SMOKE_TEST):
        square = Square(color=BLUE).spawn()
        Scene.wait(1)
        square.despawn()
        Scene.wait(1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            square.spawn()
        Scene.wait(1)
        result = Scene.save_frame(_tmp("f6.png"))

    image = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
    return image.max() == 0 and not caught, (
        f"after re-spawn the frame's brightest pixel is {image.max()}, "
        f"warnings emitted: {len(caught)}"
    )


# --------------------------------------------------------------------------
# F7 -- a surface that tessellates to nothing crashes the renderer
# --------------------------------------------------------------------------
@check("F7", "a zero-triangle Surface crashes the renderer with a tensor error")
def f7():
    failures = []
    for label, build in (
        ("Sphere(radius=0)", lambda: Sphere(radius=0, color=BLUE)),
        ("Sphere(radius=1e-9)", lambda: Sphere(radius=1e-9, color=BLUE)),
        ("Cylinder(radius=0)", lambda: Cylinder(radius=0, color=BLUE)),
    ):

        def render(build=build):
            with Scene(video_settings=SMOKE_TEST):
                build().spawn()
                Scene.save_frame(_tmp("f7.png"))

        exc = _raised(render)
        if exc is not None:
            failures.append(f"{label} -> {type(exc).__name__}")
    return bool(failures), "; ".join(failures) or "all render"


# --------------------------------------------------------------------------
# F8 -- set_material does not validate its argument
# --------------------------------------------------------------------------
@check("F8", "set_material() reports an internal AttributeError for a non-Material")
def f8():
    with Scene(video_settings=SMOKE_TEST):
        messages = []
        for label, value in (
            ("GOLD (a Color)", GOLD),
            ("None", None),
            ('"gold"', "gold"),
        ):
            exc = _raised(lambda value=value: Sphere().set_material(value))
            if isinstance(exc, AttributeError):
                messages.append(f"{label} -> {exc}")
    return bool(messages), "; ".join(messages) or "validated"


# --------------------------------------------------------------------------
# F9 -- empty / whitespace Text
# --------------------------------------------------------------------------
@check("F9", 'Text("") fails on spawn with a raw torch.cat error')
def f9():
    with Scene(video_settings=SMOKE_TEST):
        exc = _raised(lambda: Text("").spawn())
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "spawns"


# --------------------------------------------------------------------------
# F10 -- Seq/Sync reject lag_ratio with a message naming Lag
# --------------------------------------------------------------------------
@check("F10", "Seq(lag_ratio=...) raises a TypeError naming an internal class")
def f10():
    exc = _raised(lambda: Seq(lag_ratio=0.5))
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "accepted"


# --------------------------------------------------------------------------
# F11 -- an unusable codec surfaces as a missing temp file
# --------------------------------------------------------------------------
@check("F11", "save_video(codec=...) reports a missing temp file, not a bad codec")
def f11():
    def render():
        with Scene(video_settings=SMOKE_TEST):
            Square(color=BLUE).spawn()
            Scene.save_video(_tmp("f11.mp4"), SMOKE_TEST, codec="notacodec")

    exc = _raised(render)
    bug = isinstance(exc, FileNotFoundError)
    return bug, f"{type(exc).__name__}: {exc}" if exc else "encoded"


# --------------------------------------------------------------------------
# F12 -- ImageMob rejects numpy arrays
# --------------------------------------------------------------------------
@check("F12", "ImageMob(numpy_array) fails with a raw torch TypeError")
def f12():
    import numpy as np

    with Scene(video_settings=SMOKE_TEST):
        exc = _raised(lambda: ImageMob(np.zeros((8, 8, 4), np.uint8)))
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "accepted"


# --------------------------------------------------------------------------
# F13 -- the texture shape error reports a shape the caller never passed
# --------------------------------------------------------------------------
@check("F13", "ImageMob's channel-count error reports a padded shape")
def f13():
    with Scene(video_settings=SMOKE_TEST):
        exc = _raised(lambda: ImageMob(torch.zeros(8, 8, 2)))
    bug = exc is not None and "(8, 8, 2)" not in str(exc)
    return bug, f"passed [8, 8, 2], error says: {exc}"


# --------------------------------------------------------------------------
# F14 -- a bad background colour is reported as a missing file
# --------------------------------------------------------------------------
@check("F14", "set_background_color('not a color') reports a missing file")
def f14():
    def call():
        with Scene(video_settings=SMOKE_TEST):
            Scene.set_background_color("not a color")

    exc = _raised(call)
    bug = exc is not None and "No such file" in str(exc)
    return bug, f"{type(exc).__name__}: {exc}" if exc else "accepted"


# --------------------------------------------------------------------------
# F15 -- reset=True leaves a cryptic error behind
# --------------------------------------------------------------------------
@check("F15", "using a Mob after save_video(reset=True) reports an internal message")
def f15():
    def call():
        with Scene(video_settings=SMOKE_TEST):
            square = Square(color=BLUE).spawn()
            Scene.save_video(_tmp("f15.mp4"), SMOKE_TEST, reset=True)
            square.move(RIGHT)

    exc = _raised(call)
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "reusable"


# --------------------------------------------------------------------------
# F16 -- Manim method names raise a bare AttributeError
# --------------------------------------------------------------------------
@check("F16", "Manim's Mobject methods raise a bare AttributeError with no pointer")
def f16():
    with Scene(video_settings=SMOKE_TEST):
        square = Square().spawn()
        missing = [
            name
            for name in (
                "shift",
                "animate",
                "next_to",
                "to_edge",
                "arrange",
                "set_fill",
            )
            if not hasattr(square, name)
        ]
    return bool(missing), "no attribute and no hint: " + ", ".join(missing)


# --------------------------------------------------------------------------
# F17 -- self-parenting is accepted where Group rejects it
# --------------------------------------------------------------------------
@check("F17", "set_parent_to() accepts self-parents and cycles that Group rejects")
def f17():
    with Scene(video_settings=SMOKE_TEST):
        square = Square()
        self_parent = _raised(lambda: square.set_parent_to(square)) is None
        first, second = Square(), Circle()
        first.set_parent_to(second)
        cycle = _raised(lambda: second.set_parent_to(first)) is None
        group_rejects = _raised(lambda: Group(Square()).add) is None
        holder = Group(Square())
        group_rejects = _raised(lambda: holder.add(holder)) is not None
    return (self_parent or cycle) and group_rejects, (
        f"set_parent_to(self) accepted={self_parent}, 2-cycle accepted={cycle}, "
        f"Group.add(self) rejected={group_rejects}"
    )


def main(argv):
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = [a for a in argv if not a.startswith("-")]
    if "--list" in argv:
        for ident, (title, _) in CHECKS.items():
            print(f"{ident:5s} {title}")
        return 0
    bugs = 0
    for ident, (title, fn) in CHECKS.items():
        if wanted and ident not in wanted:
            continue
        try:
            is_bug, detail = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"REPRO {ident:5s} ERROR  -- the check itself failed: {exc!r}")
            continue
        bugs += bool(is_bug)
        print(f"REPRO {ident:5s} {'BUG  ' if is_bug else 'FIXED'} -- {title}")
        print(f"             {detail}")
    print(f"\n{bugs} of the checked findings still reproduce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
