"""``algan warmup``: compile the kernels typical scenes need, before a scene needs them.

A cold render compiles its kernels serially at first launch
(``rendering/kernel_precompile.py`` explains why), so the first render on a
machine -- or the first after an update, or the first with glass or shadows --
is minutes of waiting in the middle of someone's work. ``algan warmup`` moves
that wait to a moment of the user's choosing and spreads it over the machine's
cores, in two phases:

1. **Every spec on record, in parallel.** The built-in common set
   (``kernel_specs_builtin.json``) under the current settings and under each
   variant's settings, plus every specialization this installation has
   rendered before (the manifest beside the kernel cache) that is not
   confirmed as cached for this version. One kernel per worker process,
   longest first.
2. **The variant scenes.** One small scene per variant -- 2-D shapes, 3-D
   solids under lights and PBR materials, the same with shadows, glass and
   metal -- each rendered in its own process at the current video settings
   (at 4 fps: the frame rate selects no kernel), about one per four cores,
   since a render already spreads over every core. Everything phase 1
   compiled is a cache hit here; anything it could not know about (a kernel
   the built-in list does not cover on this device) compiles now and is
   recorded, so the next update's phase 1 covers it.

The scenes render to a temporary directory that is removed afterwards. Nothing
here changes what a later render draws: it only fills the cache that render
would otherwise fill itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:
    description: str
    #: ``SETTINGS.raytracing`` fields the scene sets. They are compiled into
    #: kernels, so a spec recorded under them is only worth precompiling for a
    #: render that sets them too.
    raytracing: dict = field(default_factory=dict)


VARIANTS = {
    "2d": Variant("2-D shapes, strokes and text-like outlines"),
    "3d": Variant("3-D solids and surfaces under lights and PBR materials"),
    "shadows": Variant("the 3-D scene with shadows on", {"shadows": True}),
    "glass": Variant("glass, mirror and metal"),
}


# ---------------------------------------------------------------------------
# The scenes. Each records a second of animation on the active Scene.
# ---------------------------------------------------------------------------


def _scene_2d():
    from algan import BLUE, LEFT, RED, RIGHT, UP, YELLOW, Arrow, Circle, Square, Sync

    square = Square(color=BLUE).spawn()
    circle = Circle(color=RED).move(RIGHT * 2.5).spawn()
    arrow = Arrow(LEFT * 3, LEFT * 1.5, color=YELLOW).spawn()
    with Sync():
        square.rotate(45)
        circle.move(UP)
        arrow.color = RED


def _lights():
    from algan import (
        ORIGIN,
        OUTWARD,
        RIGHT,
        UP,
        WHITE,
        AmbientLight,
        DirectionalLight,
        Off,
        PointLight,
    )

    with Off():
        AmbientLight(color=WHITE, intensity=0.4).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 4 + UP * 5 + OUTWARD * 4, target=ORIGIN, intensity=1.0
        ).spawn(animate=False)
        PointLight(location=RIGHT * -3 + UP * 2 + OUTWARD * 1.5, intensity=1.5).spawn(
            animate=False
        )


def _scene_3d():
    from algan import (
        BLUE,
        DOWN,
        GREEN,
        LEFT,
        RED,
        RIGHT,
        UP,
        Cube,
        MeshLambertMaterial,
        MeshStandardMaterial,
        Prism,
        Sphere,
        Square,
        Sync,
    )

    _lights()
    Prism(width=10, height=0.2, depth=6).set_material(
        MeshLambertMaterial(color=GREEN)
    ).move(DOWN * 1.6).spawn()
    cube = (
        Cube()
        .set_material(MeshStandardMaterial(color=RED, roughness=0.4))
        .move(LEFT * 2)
    )
    sphere = Sphere(radius=0.8).set_material(
        MeshStandardMaterial(color=BLUE, roughness=0.3, metalness=0.2)
    )
    square = Square(color=BLUE).move(RIGHT * 2)
    cube.spawn()
    sphere.spawn()
    square.spawn()
    with Sync():
        cube.rotate(60, UP)
        sphere.move(UP * 0.5)
        square.rotate(90, UP)


def _scene_glass():
    from algan import (
        BLUE_A,
        COPPER,
        GLASS,
        LEFT,
        MIRROR,
        RIGHT,
        UP,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
        Sphere,
        Sync,
    )
    from algan.constants.color import GOLD

    _lights()
    spheres = [
        Sphere(radius=0.6).set_material(GLASS).move(LEFT * 3),
        Sphere(radius=0.6).set_material(MIRROR).move(LEFT * 1),
        Sphere(radius=0.6).set_material(COPPER).move(RIGHT * 1),
        Sphere(radius=0.6)
        .set_material(
            MeshPhysicalMaterial(
                color=BLUE_A, roughness=0.1, transmission=0.5, ior=1.45
            )
        )
        .move(RIGHT * 3),
        Sphere(radius=0.5)
        .set_material(MeshStandardMaterial(color=GOLD, metalness=1.0, roughness=0.2))
        .move(UP * 1.6),
    ]
    for sphere in spheres:
        sphere.spawn()
    with Sync():
        for sphere in spheres:
            sphere.rotate(90, UP)


_SCENES = {
    "2d": _scene_2d,
    "3d": _scene_3d,
    "shadows": _scene_3d,
    "glass": _scene_glass,
}
_WARMUP_FPS = 4


def run_variant(name, output_directory):
    """Render one variant in this process (the body of a phase-2 child)."""
    from algan import SETTINGS, Scene
    from algan.rendering import kernel_progress

    variant = VARIANTS[name]
    if variant.raytracing:
        SETTINGS.raytracing.set(**variant.raytracing)
    # A few frames are enough: the frame rate is not an input to any kernel
    # (checked: the 3d and glass scenes materialize the same specializations
    # at 15 fps and 4), and every frame past the first only costs render time.
    SETTINGS.video.set(frames_per_second=_WARMUP_FPS)
    _SCENES[name]()
    started = time.perf_counter()
    Scene.save_video(os.path.join(output_directory, f"warmup_{name}.mp4"))
    state = kernel_progress._STATE
    print(
        "REPORT "
        + json.dumps(
            {
                "variant": name,
                "seconds": time.perf_counter() - started,
                "compiled": state.compiled,
            }
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _format(seconds):
    from algan.rendering.kernel_progress import _format_seconds

    return _format_seconds(seconds)


def variant_context(context, name):
    """``context`` with a variant's settings applied."""
    overrides = VARIANTS[name].raytracing
    if not overrides:
        return context
    return {**context, "raytracing": {**context["raytracing"], **overrides}}


def known_jobs(variants):
    """Phase 1's jobs: built-ins under each variant's settings, and every recorded spec.

    Recorded specs are compiled under the context they were recorded in, which
    is only reproducible when its environment variables are this process's --
    those are read at import and a worker inherits them -- so rows from a
    context with other ``ALGAN_`` variables are left out and counted.
    """
    from algan.rendering import kernel_precompile as kp

    context = kp.current_context()
    manifest = kp.read_manifest()
    jobs = {}
    # The built-in list under each variant's settings ...
    candidates = [(variant_context(context, name), True) for name in variants]
    # ... and every recorded context this process can reproduce, for its own
    # rows only: the built-ins are compiled for the settings a variant uses,
    # not once more per custom configuration a user has rendered with.
    for recorded in manifest["contexts"].values():
        if recorded.get("env") == context["env"]:
            candidates.append((recorded, False))
    foreign = sum(
        1
        for row in manifest["entries"].values()
        if manifest["contexts"].get(row["context"], {}).get("env") != context["env"]
    )
    for candidate, include_builtin in candidates:
        for job in kp.pending_jobs(
            candidate, include_builtin=include_builtin, manifest=manifest
        ):
            jobs.setdefault(job.eid, job)
    return list(jobs.values()), foreign


def _run_pool(jobs, workers, echo):
    from algan.rendering import kernel_precompile as kp

    if not jobs:
        echo("Phase 1: every kernel on record is already cached for this version.")
        return {}
    pool = kp.PrecompilePool(jobs, workers, reason="algan warmup")
    lock = threading.Lock()
    done = [0]
    total = len(jobs)

    def listener(pool, kind, payload):
        if kind == "done":
            with lock:
                done[0] += 1
                number = done[0]
            job = payload
            if job.state == "done":
                verb = {
                    "compiled": "compiled",
                    "offline-hit": "already compiled (reindexed)",
                    "cached": "already cached",
                }.get(job.status, job.status)
                echo(
                    f"  [{number}/{total}] {job.name}: {verb} in {_format(job.seconds)}"
                )
            else:
                echo(f"  [{number}/{total}] {job.name}: not compiled ({job.reason})")
        elif kind == "fatal":
            echo(f"  a worker could not start: {payload}")

    pool.listeners.append(listener)
    echo(
        f"Phase 1: compiling {total} known kernel specialization"
        f"{'s' if total != 1 else ''} in {min(workers, total)} worker process"
        f"{'es' if min(workers, total) != 1 else ''}, longest first."
    )
    pool.start()
    try:
        pool.wait()
    except KeyboardInterrupt:
        pool.terminate()
        raise
    kp.flush_manifest()
    statuses = {}
    for job in pool.jobs:
        key = job.status if job.state == "done" else "failed"
        statuses[key] = statuses.get(key, 0) + 1
    return statuses


def _scene_concurrency(workers):
    """How many warm-up scenes to render at once.

    Not one per worker: after phase 1 a scene mostly *renders*, and a render
    already spreads over every core (torch's intra-op threads, the CPU
    kernels' own). Three scenes at once on a 4-core box took ~2 min each
    against 29-45 s alone. About one scene per four cores keeps the parallel
    win for a scene that does compile (single-threaded) without that
    oversubscription.
    """
    return max(1, min(workers, (os.cpu_count() or 1) // 4))


def _run_scenes(variants, workers, echo):
    """Phase 2: each variant scene in its own process, a few at a time."""
    workers = _scene_concurrency(workers)
    from algan.environment import env_overrides

    output = tempfile.mkdtemp(prefix="algan_warmup_")
    env = dict(os.environ)
    env.pop("ALGAN_DAEMON_CHILD", None)
    env.update(
        env_overrides(
            ALGAN_USE_DAEMON="0",
            ALGAN_AUTO_DAEMON="0",
            # Phase 1 already ran the pool; a scene's own render-start pool
            # would only compete with the other scenes for the same cores.
            ALGAN_PRECOMPILE_JOBS="0",
            ALGAN_LOG_LEVEL="WARNING",
        )
    )
    echo(
        f"Phase 2: rendering {len(variants)} warm-up scene"
        f"{'s' if len(variants) != 1 else ''}, {min(workers, len(variants))} at a time."
    )
    pending = list(variants)
    running = {}
    reports = {}
    try:
        while pending or running:
            while pending and len(running) < max(1, workers):
                name = pending.pop(0)
                code = (
                    "from algan.rendering.kernel_warmup import run_variant; "
                    f"run_variant({name!r}, {output!r})"
                )
                running[name] = (
                    subprocess.Popen(
                        [sys.executable, "-c", code],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    ),
                    time.perf_counter(),
                )
            for name, (process, started) in list(running.items()):
                if process.poll() is None:
                    continue
                stdout, stderr = process.communicate()
                del running[name]
                report = next(
                    (
                        json.loads(line[len("REPORT ") :])
                        for line in stdout.splitlines()
                        if line.startswith("REPORT ")
                    ),
                    None,
                )
                elapsed = time.perf_counter() - started
                if process.returncode != 0 or report is None:
                    tail = (stderr or stdout).strip().splitlines()[-3:]
                    echo(
                        f"  {name}: failed after {_format(elapsed)}: {' | '.join(tail)}"
                    )
                    reports[name] = None
                    continue
                reports[name] = report
                new = report["compiled"]
                echo(
                    f"  {name} ({VARIANTS[name].description}): {_format(elapsed)}"
                    + (
                        f", compiled {new} kernel{'s' if new != 1 else ''} phase 1 did not know"
                        if new
                        else ", every kernel already cached"
                    )
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        for process, _ in running.values():
            process.kill()
        raise
    finally:
        shutil.rmtree(output, ignore_errors=True)
    return reports


def warmup(*, jobs=None, variants=None, scenes=True, echo=print):
    """Run both phases; returns a process exit code."""
    from algan.rendering import kernel_precompile as kp
    from algan.settings._startup import _TAICHI_CACHE_DIRECTORY, render_device

    variants = list(variants or VARIANTS)
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        echo(
            f"Unknown variant(s): {', '.join(unknown)}. Choose from {', '.join(VARIANTS)}."
        )
        return 2
    reason = kp.skipped_reason()
    if reason is not None and "ALGAN_PRECOMPILE_JOBS" not in reason:
        echo(f"Cannot precompile here: {reason}.")
        return 1
    kp.stop_implicit_pool()
    started = time.perf_counter()
    job_list, foreign = known_jobs(variants)
    workers = (
        jobs
        if jobs is not None
        else kp.worker_budget(max(len(job_list), len(variants)))
    )
    workers = max(1, workers)
    echo(
        f"Warming the kernel cache for device {render_device()} at {_TAICHI_CACHE_DIRECTORY}, "
        f"with up to {workers} worker process{'es' if workers != 1 else ''}."
    )
    if foreign:
        echo(
            f"  ({foreign} recorded specialization{'s' if foreign != 1 else ''} "
            "used other ALGAN_ environment variables and are left for their own renders.)"
        )
    statuses = _run_pool(job_list, workers, echo)
    phase_one = time.perf_counter() - started
    failed_scenes = 0
    if scenes:
        reports = _run_scenes(variants, workers, echo)
        failed_scenes = sum(report is None for report in reports.values())
    compiled = statuses.get("compiled", 0)
    reused = statuses.get("cached", 0) + statuses.get("offline-hit", 0)
    echo(
        f"Warm-up finished in {_format(time.perf_counter() - started)} "
        f"(phase 1: {_format(phase_one)}): {compiled} kernel"
        f"{'s' if compiled != 1 else ''} compiled, {reused} already cached"
        + (f", {statuses['failed']} not compiled" if statuses.get("failed") else "")
        + (f", {failed_scenes} scene(s) failed" if failed_scenes else "")
        + ". Renders with these features now start without compiling."
    )
    return 1 if failed_scenes else 0
