"""One check per finding in ``STRESS_TEST_FINDINGS.md``.

Run it with the project venv::

    uv run python stress_test/reproductions.py           # every check
    uv run python stress_test/reproductions.py --list     # just the ids
    uv run python stress_test/reproductions.py F1 F3      # selected checks

Each check prints ``REPRO <id> <FIXED|BUG> -- <one line>``. ``FIXED`` means the
behaviour the report asked for is what happens now; ``BUG`` means the original
finding still reproduces.

Every finding also has a test in ``tests/unit_tests`` -- that is what CI runs,
and it is the authority. This file exists so the report has one command behind
it, and so the before/after of each finding can be seen in one place.

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
# F1 -- re-entering a context object leaked the animation-manager override
# --------------------------------------------------------------------------
@check("F1", "re-entering an AnimationContext is refused, and Sync still overlaps")
def f1():
    from algan.errors import ContextReuseError

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

    nested = _raised(lambda: _enter_twice_nested())
    sequential = _raised(lambda: _enter_twice_sequentially())

    after = sync_seconds()
    leaked = _contexts._ANIMATION_MANAGER_OVERRIDE.get(None) is not None
    refused = isinstance(nested, ContextReuseError) and isinstance(
        sequential, ContextReuseError
    )
    fixed = refused and not leaked and after == before
    return not fixed, (
        f"re-entry refused: {refused}; override leaked: {leaked}; "
        f"Sync of 3 lasts {before:.2f}s before and {after:.2f}s after"
    )


def _enter_twice_nested():
    with Scene():
        square = Square().spawn()
        context = Sync()
        with context:  # noqa: SIM117 -- the nesting is the point
            with context:
                square.move(RIGHT)


def _enter_twice_sequentially():
    with Scene():
        square = Square().spawn()
        context = Sync()
        with context:
            square.move(RIGHT)
        with context:
            square.move(UP)


# --------------------------------------------------------------------------
# F2 -- creating a Mob inside an updater crashed the render
# --------------------------------------------------------------------------
@check("F2", "an updater can build a Mob, and the render leaves the Scene alone")
def f2():
    def build_and_render():
        with Scene(video_settings=SMOKE_TEST) as scene:
            square = Square(color=BLUE).spawn()
            square.add_updater(lambda mob, t: Circle(radius=0.1, color=RED))
            Scene.wait(1)
            actors = len(scene.actors)
            rows = {
                name: timeline.pointer
                for name, timeline in scene.timeline_manager.attr_to_timeline.items()
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Scene.save_video(_tmp("f2.mp4"), SMOKE_TEST)
            return actors == len(scene.actors) and rows == {
                name: timeline.pointer
                for name, timeline in scene.timeline_manager.attr_to_timeline.items()
            }

    exc = _raised(lambda: _store(build_and_render))
    unchanged = _LAST.get("value")
    return exc is not None or not unchanged, (
        f"{type(exc).__name__}: {exc}"
        if exc
        else f"renders; Scene unchanged by the render: {unchanged}"
    )


_LAST = {}


def _store(fn):
    _LAST["value"] = fn()


@check("F2b", "a Mob that reshapes itself in an updater is explained, not crashed")
def f2b():
    from algan.errors import UnsupportedFeatureError

    def build():
        with Scene(video_settings=SMOKE_TEST):
            number = DecimalNumber(0).spawn()
            tracker = ValueTracker(0).spawn()
            number.add_updater(lambda mob, t: mob.set_value(tracker.get_value()))
            tracker.set_value(5)
            Scene.save_video(_tmp("f2b.mp4"), SMOKE_TEST)

    exc = _raised(build)
    explained = isinstance(exc, UnsupportedFeatureError) and "updater" in str(exc)
    return (
        not explained,
        f"{type(exc).__name__}: {str(exc)[:120] if exc else 'renders'}",
    )


@check("F2c", "NumericDisplay counts inside an updater")
def f2c():
    def build():
        with Scene(video_settings=SMOKE_TEST):
            display = NumericDisplay(0.0).spawn()
            tracker = ValueTracker(0).spawn()
            display.add_updater(lambda mob, t: mob.set_value(tracker.get_value()))
            tracker.set_value(5)
            Scene.save_video(_tmp("f2c.mp4"), SMOKE_TEST)

    exc = _raised(build)
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "renders"


@check("F2d", "set_value leaves a Manim number visible")
def f2d():
    def brightest(value):
        with Scene(video_settings=SMOKE_TEST):
            number = DecimalNumber(0.0).spawn()
            if value is not None:
                number.set_value(value)
            result = Scene.save_frame(_tmp(f"f2d_{value}.png"))
        return cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED).max()

    plain, after_set = brightest(None), brightest(3.0)
    return (
        after_set == 0,
        f"brightest pixel: {plain} plain, {after_set} after set_value",
    )


# --------------------------------------------------------------------------
# F3 -- save_video ignored the Scene's own video settings
# --------------------------------------------------------------------------
@check("F3", "save_video and save_frame agree on the Scene's video settings")
def f3():
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(color=BLUE).spawn()
        frame = Scene.save_frame(_tmp("f3.png"))
        video = Scene.save_video(_tmp("f3.mp4"))
        wanted = tuple(scene.video_settings.resolution)

    png = cv2.imread(str(frame.output_path), cv2.IMREAD_UNCHANGED)
    cap = cv2.VideoCapture(str(video.output_path))
    got = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return got != wanted, (
        f"Scene asked for {wanted}; save_frame wrote {png.shape[1]}x{png.shape[0]}, "
        f"save_video wrote {got[0]}x{got[1]}"
    )


# --------------------------------------------------------------------------
# F4 -- the README quickstart did not run
# --------------------------------------------------------------------------
@check("F4", "every name the README quickstart calls exists")
def f4():
    with Scene(video_settings=SMOKE_TEST):
        sphere = Sphere(color=BLUE, radius=1.2)
        sphere.set_material(MeshPhysicalMaterial(roughness=0.15))
        broken = []
        with Seq():
            if _raised(sphere.spawn):
                broken.append("spawn()")
            if _raised(lambda: Sync(run_time=2.0)):
                broken.append("Sync(run_time=)")
            if _raised(lambda: sphere.move([2, 0, 0])):
                broken.append("Mob.move()")
    return bool(broken), "unsupported: " + (", ".join(broken) or "nothing")


# --------------------------------------------------------------------------
# F5 -- colour arguments only accepted Color
# --------------------------------------------------------------------------
@check("F5", "hex strings, CSS names, hex ints and RGB tuples are colours")
def f5():
    rejected = []
    with Scene(video_settings=SMOKE_TEST):
        for label, call in (
            ('Square(color="#ff0000")', lambda: Square(color="#ff0000")),
            ('Square(color="red")', lambda: Square(color="red")),
            ("Square(color=0xFF0000)", lambda: Square(color=0xFF0000)),
            ("Square(color=(1, 0, 0))", lambda: Square(color=(1, 0, 0))),
            ("Sphere(color=0xFF0000)", lambda: Sphere(color=0xFF0000)),
            ('Text(color="red")', lambda: Text("x", color="red")),
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
# F6 -- spawn() after despawn() was a silent no-op
# --------------------------------------------------------------------------
@check("F6", "spawn() on a despawned Mob warns")
def f6():
    from algan.errors import DespawnedMobWarning

    with Scene(video_settings=SMOKE_TEST):
        square = Square(color=BLUE).spawn()
        square.despawn()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            square.spawn()
    warned = any(issubclass(w.category, DespawnedMobWarning) for w in caught)
    return not warned, f"warnings emitted: {[w.category.__name__ for w in caught]}"


# --------------------------------------------------------------------------
# F7 -- a zero-triangle surface crashed the renderer
# --------------------------------------------------------------------------
@check("F7", "a surface with no extent renders nothing instead of failing")
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
# F8 -- set_material did not check its argument
# --------------------------------------------------------------------------
@check("F8", "set_material names what it wants")
def f8():
    from algan.errors import AlganConfigurationError

    with Scene(video_settings=SMOKE_TEST):
        wrong = [
            _raised(lambda value=value: Sphere().set_material(value))
            for value in (GOLD, None, "gold")
        ]
        good = _raised(lambda: Sphere().set_material(GLASS))
    explained = all(isinstance(e, AlganConfigurationError) for e in wrong)
    return not (explained and good is None), (
        f"rejected with a Material message: {explained}; GLASS still accepted: "
        f"{good is None}"
    )


# --------------------------------------------------------------------------
# F9 -- empty Text
# --------------------------------------------------------------------------
@check("F9", 'Text("") spawns')
def f9():
    with Scene(video_settings=SMOKE_TEST):
        failures = [
            f"{text!r} -> {type(exc).__name__}"
            for text in ("", "   ", "\n")
            if (exc := _raised(lambda text=text: Text(text).spawn())) is not None
        ]
    return bool(failures), "; ".join(failures) or "all spawn"


# --------------------------------------------------------------------------
# F10 -- Seq/Sync rejected lag_ratio by blaming Lag
# --------------------------------------------------------------------------
@check("F10", "Seq(lag_ratio=...) explains itself without naming an internal class")
def f10():
    exc = _raised(lambda: Seq(lag_ratio=0.5))
    good = isinstance(exc, TypeError) and "Seq is Lag with lag_ratio=1" in str(exc)
    return not good, f"{type(exc).__name__}: {str(exc)[:110] if exc else 'accepted'}"


# --------------------------------------------------------------------------
# F11 -- an unusable codec surfaced as a missing temp file
# --------------------------------------------------------------------------
@check("F11", "an unusable codec is named, before the render")
def f11():
    from algan.errors import AlganConfigurationError
    from algan.utils.video_encoding import _listed_encoders

    if _listed_encoders("ffmpeg") is None:
        return False, "skipped: this FFmpeg cannot be asked for its encoder list"

    def render():
        with Scene(video_settings=SMOKE_TEST):
            Square(color=BLUE).spawn()
            Scene.save_video(_tmp("f11.mp4"), SMOKE_TEST, codec="notacodec")

    exc = _raised(render)
    good = isinstance(exc, AlganConfigurationError) and "codec" in str(exc)
    return not good, f"{type(exc).__name__}: {str(exc)[:110] if exc else 'encoded'}"


# --------------------------------------------------------------------------
# F12 / F13 -- ImageMob's inputs and its shape complaint
# --------------------------------------------------------------------------
@check("F12", "ImageMob takes a numpy array")
def f12():
    import numpy as np

    with Scene(video_settings=SMOKE_TEST):
        exc = _raised(lambda: ImageMob(np.zeros((8, 8, 4), np.uint8)))
    return exc is not None, f"{type(exc).__name__}: {exc}" if exc else "accepted"


@check("F13", "a channel-count complaint reports the shape that was passed")
def f13():
    with Scene(video_settings=SMOKE_TEST):
        exc = _raised(lambda: ImageMob(torch.zeros(8, 8, 2)))
    good = exc is not None and "(8, 8, 2)" in str(exc)
    return not good, f"passed [8, 8, 2], error says: {exc}"


# --------------------------------------------------------------------------
# F14 -- a bad background colour was reported as a missing file
# --------------------------------------------------------------------------
@check("F14", "a background string is read as a colour first")
def f14():
    from algan.errors import AlganConfigurationError

    def call(value):
        with Scene(video_settings=SMOKE_TEST):
            Scene.set_background_color(value)

    named = _raised(lambda: call("navy"))
    nonsense = _raised(lambda: call("not a color"))
    good = named is None and isinstance(nonsense, AlganConfigurationError)
    return not good, (
        f"'navy' accepted: {named is None}; 'not a color' -> "
        f"{type(nonsense).__name__ if nonsense else 'accepted'}"
    )


# --------------------------------------------------------------------------
# F15 -- reset=True left a cryptic error behind
# --------------------------------------------------------------------------
@check("F15", "using a Mob after save_video(reset=True) says why it cannot work")
def f15():
    def call():
        with Scene(video_settings=SMOKE_TEST):
            square = Square(color=BLUE).spawn()
            Scene.save_video(_tmp("f15.mp4"), SMOKE_TEST, reset=True)
            square.move(RIGHT)

    exc = _raised(call)
    good = exc is not None and "reset=True" in str(exc)
    return not good, f"{type(exc).__name__}: {str(exc)[:150] if exc else 'reusable'}"


# --------------------------------------------------------------------------
# F16 -- Manim method names raised a bare AttributeError
# --------------------------------------------------------------------------
@check("F16", "Manim's Mobject method names point at the Algan one")
def f16():
    with Scene(video_settings=SMOKE_TEST):
        square = Square().spawn()
        hints = {}
        for name in ("shift", "next_to", "to_edge", "animate"):
            exc = _raised(lambda name=name: getattr(square, name))
            hints[name] = exc is not None and "in Algan use" in str(exc)
        plain = _raised(lambda: square.wibble)
    good = (
        all(hints.values()) and plain is not None and "in Algan use" not in str(plain)
    )
    return not good, f"hinted: {hints}; an ordinary typo stays ordinary: {plain}"


# --------------------------------------------------------------------------
# F17 -- self-parenting and cycles were accepted
# --------------------------------------------------------------------------
@check("F17", "set_parent_to rejects the cycles Group rejects")
def f17():
    from algan.errors import HierarchyError

    with Scene(video_settings=SMOKE_TEST):
        square = Square()
        self_parent = _raised(lambda: square.set_parent_to(square))
        first, second = Square(), Circle()
        first.set_parent_to(second)
        cycle = _raised(lambda: second.set_parent_to(first))
        chain = _raised(lambda: Square().set_parent_to(second))
    good = (
        isinstance(self_parent, HierarchyError)
        and isinstance(cycle, HierarchyError)
        and chain is None
    )
    return not good, (
        f"self-parent rejected: {isinstance(self_parent, HierarchyError)}; "
        f"2-cycle rejected: {isinstance(cycle, HierarchyError)}; "
        f"a plain chain still allowed: {chain is None}"
    )


# --------------------------------------------------------------------------
# F18 -- updater arity
# --------------------------------------------------------------------------
@check("F18", "a one-argument updater is told it needs (mob, t)")
def f18():
    with Scene(video_settings=SMOKE_TEST):
        square = Square().spawn()
        exc = _raised(lambda: square.add_updater(lambda mob: None))
        good_one = _raised(lambda: square.add_updater(lambda mob, t: None))
    good = (
        isinstance(exc, TypeError)
        and "positional parameters" in str(exc)
        and good_one is None
    )
    return not good, f"{type(exc).__name__}: {str(exc)[:110] if exc else 'accepted'}"


# --------------------------------------------------------------------------
# F19 -- post-processing passes
# --------------------------------------------------------------------------
@check("F19", "a non-callable post-processing pass is rejected before the render")
def f19():
    from algan.errors import AlganConfigurationError

    def call():
        with Scene(video_settings=SMOKE_TEST):
            Square(color=BLUE).spawn()
            Scene.save_frame(_tmp("f19.png"), post_processes=(42,))

    exc = _raised(call)
    good = isinstance(exc, AlganConfigurationError) and "post_processes" in str(exc)
    return not good, f"{type(exc).__name__}: {str(exc)[:110] if exc else 'accepted'}"


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
