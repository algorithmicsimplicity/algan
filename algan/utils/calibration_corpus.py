"""Scenes and configurations measured by :mod:`algan.utils.calibrate_memory`.

The corpus has two jobs, and they pull in different directions:

*Shape coverage* -- the trace and unit-coefficient fits need each driver varied
independently, at realistic sizes. Toy resolutions are actively harmful here:
bloom's downsample factor is ``max(int(scale_factor * height / 2160), 1)``, so
at 48 px every ``scale_factor`` collapses to 1 and the key-screening step
"proves" an axis irrelevant that in fact matters at 1080p.

*Content coverage* -- the value-dependent densities are percentiles over what
real scenes actually do, so this half wants breadth: dense text, meshes,
reflective materials, glow.
"""

from __future__ import annotations

import torch

# Resolutions the shipped post-process traces cover, as (width, height).
# Anything else -- including HD and above -- falls back to the on-demand
# one-frame probe, which is exact and disk-cached. The shipped table is a
# startup-cost optimisation for common cases, not a correctness requirement,
# so it is kept to sizes that measure quickly.
PRESET_RESOLUTIONS = (
    (480, 270),     # PREVIEW / LD
    (854, 480),     # MD
    (1280, 720),
)

# Frame counts each trace is fitted and verified across. Two solve the affine
# element counts and the third is pure verification -- it is what catches an
# allocation that is not actually affine in the frame count. Deeper sweeps run
# in the unit tests, where the frames are small enough to be free.
TRACE_FRAME_COUNTS = (1, 2, 3)


def _postprocess_configs():
    """(label, height, width, channels, dtype, aa, processes, fxaa, hdr)."""
    from algan.rendering.post_processing.bloom import bloom_filter

    configs = []
    for width, height in PRESET_RESOLUTIONS:
        for channels in (4, 5):
            for aa in (1, 2):
                for processes, fxaa, hdr in (
                    ((), False, False),
                    ((), True, False),
                    ((bloom_filter,), False, False),
                    ((bloom_filter,), True, False),
                    ((bloom_filter,), False, True),
                ):
                    configs.append((
                        f"{width}x{height}c{channels}aa{aa}"
                        f"{'b' if processes else ''}{'f' if fxaa else ''}"
                        f"{'h' if hdr else ''}",
                        height, width, channels,
                        torch.float32 if hdr else torch.float32,
                        aa, processes, fxaa, hdr))
    return configs


def collect_postprocess_observations():
    """Drive post-processing directly across the frame sweep.

    Driving it directly rather than through a render is deliberate: the frame
    count per render chunk is chosen by the batcher, so a render cannot supply
    the controlled sweep the affine fit needs.
    """
    from algan.rendering.post_processing.post_process import (
        post_process_frames,
    )
    from algan.rendering.raytracing import settings as rt_settings
    from algan.utils.calibrate_memory import observations_from_recorders
    from algan.utils.memory_utils import ManualMemory

    from algan.utils.memory_utils import InsufficientMemoryException

    saved_tonemap = rt_settings.POST_PROCESS_TONEMAP
    observations = []
    try:
        for (label, height, width, channels, dtype, aa,
             processes, fxaa, hdr) in _postprocess_configs():
            rt_settings.POST_PROCESS_TONEMAP = hdr
            for frames_count in TRACE_FRAME_COUNTS:
                shape = (frames_count, height, width, channels)
                if dtype == torch.uint8:
                    frames = torch.randint(1, 220, shape, dtype=dtype)
                    frames[..., 3] = 100
                else:
                    # Non-zero glow and alpha: bloom short-circuits on an
                    # all-black glow channel, and a corpus that hit that path
                    # would record a shorter allocation stream and fail the
                    # trace build with a spurious "data-dependent control
                    # flow" error.
                    frames = torch.rand(shape, dtype=dtype) * 2.0
                    frames[..., 3] = 0.7
                # Sized from the workload rather than a fraction of a global
                # budget: FXAA and bloom hold a double-digit multiple of the
                # frame in scratch, and an arena that cannot fit the largest
                # config would silently drop it from the corpus.
                frame_bytes = frames.numel() * frames.element_size()
                memory = ManualMemory(
                    0, device=torch.device("cpu"), managed=True,
                    num_bytes=max(64 << 20, frame_bytes * 24))
                try:
                    with memory.recording() as recorder:
                        post_process_frames(
                            memory, frames, aa, post_processes=processes,
                            apply_fxaa=fxaa)
                except InsufficientMemoryException:
                    print(f"[calibrate] postprocess {label} f{frames_count}: "
                          f"arena too small, skipping")
                    continue
                observations.extend(observations_from_recorders(
                    [recorder], f"postprocess:{label}"))
    finally:
        rt_settings.POST_PROCESS_TONEMAP = saved_tonemap
    return observations


# --------------------------------------------------------------------------
# Render scenes
# --------------------------------------------------------------------------

def _scene_shapes():
    from algan import BLUE, GREEN, RED, RIGHT, UP, Circle, Square, Triangle

    Square(color=RED).spawn().move(RIGHT)
    Circle(color=BLUE).spawn().move(UP)
    Triangle(color=GREEN).spawn().rotate(45, UP)


def _scene_text():
    from algan import DOWN, Tex, Text

    Text("Calibration corpus").spawn().move(DOWN)
    Tex(r"\int_0^1 x^2 \, dx = \frac{1}{3}").spawn()


def _scene_mesh():
    from algan import OUT, ORANGE, RIGHT, Sphere

    Sphere(color=ORANGE).spawn().rotate(90, OUT).move(RIGHT)


def _scene_dense_text():
    """Many glyphs: drives frames x primitives high for raster_precompute."""
    from algan import DOWN, Text

    for index in range(6):
        Text(f"line number {index} of dense calibration text").spawn().move(
            DOWN * (index - 3) * 0.4)


def _scene_glow():
    from algan import RIGHT, YELLOW, Circle

    circle = Circle(color=YELLOW).spawn()
    circle.glow = 3.0
    circle.move(RIGHT)


def _scene_refractive():
    """Glass, to break the ``pool == primary`` collinearity.

    With no splitting material the continuation pool is exactly one slot per
    primary ray, so a corpus of opaque scenes can never separate the per-slot
    cost from the per-primary cost -- the solve folds one into the other and
    is then silently wrong for any scene that does split. A refractive object
    raises the pool ratio above one and makes the two columns independent.
    """
    from algan import (
        BLUE, LEFT, MeshPhysicalMaterial, OUT, RIGHT, Sphere, Square,
    )

    # Something behind the glass, so refracted rays actually spawn
    # continuations rather than escaping to the background.
    Square(color=BLUE).scale(0.5).move(LEFT * 0.5 - OUT * 2.0).spawn()
    glass = Sphere().scale(1.2)
    glass.set_material(
        MeshPhysicalMaterial(transmission=0.95, roughness=0.02, ior=1.5))
    glass.spawn()
    glass.move(RIGHT * 0.3)


# (label, scene, raytracing setting overrides). The overrides exist to reach
# routes an ordinary scene cannot: the Monte Carlo path tracer allocates a
# sample accumulator that the deterministic renderer never does, and a driver
# that is zero in every sample has no measurable cost.
def _scene_area_light():
    """An extended light, to separate the packed light position and colour.

    A plain point light packs three floats of position and three of colour, so
    the two arrays are the same size in every frame and their per-element costs
    cannot be told apart. Area and environment lights widen the colour row to
    sixteen, which breaks the tie.
    """
    from algan import (
        AmbientLight, ORIGIN, RIGHT, Sphere, UP, WHITE, RectAreaLight,
    )
    from algan.scene_manager import SceneManager

    SceneManager.instance().current_scene.light_sources = [
        RectAreaLight(location=UP * 4 + RIGHT * 2, target=ORIGIN,
                      width=4.0, height=4.0, samples=4,
                      color=WHITE, intensity=1.1).spawn(animate=False),
        AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False),
    ]
    Sphere(color=WHITE).spawn().move(RIGHT)


def _scene_mixed():
    """Both geometry families at once.

    The raster precompute tables are emitted per family, so a route exercised
    only by 2D shapes leaves the triangle table at zero rows and its cost
    unmeasured -- 2D shapes are bezier circuits, not triangles. Route-coverage
    scenes therefore carry a mesh as well.
    """
    from algan import BLUE, ORANGE, RIGHT, UP, Circle, Sphere, Square

    Square(color=BLUE).spawn().move(RIGHT)
    Circle(color=BLUE).spawn().move(UP)
    Sphere(color=ORANGE).spawn()


RENDER_SCENES = (
    ("shapes", _scene_shapes, {}),
    ("area_light", _scene_area_light, {}),
    ("text", _scene_text, {}),
    ("mesh", _scene_mesh, {}),
    ("dense_text", _scene_dense_text, {}),
    ("glow", _scene_glow, {}),
    ("refractive", _scene_refractive, {}),
    ("monte_carlo", _scene_shapes, {"SAMPLES_PER_PIXEL": 2}),
    # Non-default routes. Each allocates differently, so without these the
    # table has no entry for them and the runtime must fall back to the
    # largest measured route -- correct, but needlessly conservative. The
    # uint8 frame buffer in particular is four times smaller than the HDR one,
    # so guessing it from the float route would quarter the batch size.
    ("no_tonemap", _scene_mixed, {"POST_PROCESS_TONEMAP": False}),
    # A second scene of a different length, so the byte frame buffer has two
    # samples with different element counts. One sample cannot separate a
    # per-element cost from a fixed one.
    ("no_tonemap_dense", _scene_dense_text, {"POST_PROCESS_TONEMAP": False}),
    ("no_analytic_aa", _scene_mixed, {"ANALYTIC_AA": False}),
    # The projection table's size is proportional to the bounds table's
    # whenever triangles exist, so telling their costs apart needs a
    # circuit-only scene and a triangle-only scene on this route as well.
    ("no_analytic_aa_bez", _scene_shapes, {"ANALYTIC_AA": False}),
    ("no_analytic_aa_tri", _scene_mesh, {"ANALYTIC_AA": False}),
    ("no_sparse_coverage", _scene_mixed, {"RASTER_SPARSE_COVERAGE": False}),
    ("no_hybrid_raster", _scene_mixed, {"HYBRID_RASTER": False}),
    # Glass again, on the other two wavefront routes. Each route is its own
    # table key, and within a key an opaque scene always has one pool slot per
    # primary ray -- so without a splitting material on *that* route, the
    # per-slot and per-primary costs stay indistinguishable there too.
    ("refractive_dense", _scene_refractive, {"RASTER_SPARSE_COVERAGE": False}),
    ("refractive_classic", _scene_refractive, {"HYBRID_RASTER": False}),
)


def collect_render_observations(video_settings=None):
    """Render every corpus scene with the recorder armed."""
    from algan.settings import LD
    from algan.utils.calibrate_memory import collect_from_render

    # LD rather than SMOKE_TEST: at 32x32 the raster precompute and wavefront
    # tile terms are too small to separate from their fixed costs, and the
    # sparse-coverage densities would be measured on a handful of fragments.
    from algan.rendering.raytracing import settings as rt_settings

    settings = video_settings if video_settings is not None else LD
    observations = []
    failures = []
    for name, scene_func, overrides in RENDER_SCENES:
        # Written straight onto the settings module, as the benchmark scripts
        # do: most of these are experimental switches whose public setter
        # refuses direct assignment, and the engine reads the module globals
        # live in any case.
        saved = {key: getattr(rt_settings, key) for key in overrides}
        try:
            for key, value in overrides.items():
                setattr(rt_settings, key, value)
            observations.extend(
                collect_from_render(scene_func, settings, name))
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"[calibrate] corpus scene {name!r} failed to render: "
                  f"{type(exc).__name__}: {exc}")
        finally:
            for key, value in saved.items():
                setattr(rt_settings, key, value)
    if failures:
        # Refuse to emit a table measured from a partial corpus. A scope that
        # silently loses all its samples produces *no* entry, the runtime then
        # falls back for every route, and the regression is invisible until
        # something over- or under-reserves in production. Better to fail the
        # regeneration and keep the previous table.
        summary = ", ".join(
            f"{name} ({type(exc).__name__}: {exc})" for name, exc in failures)
        raise RuntimeError(
            f"{len(failures)} of {len(RENDER_SCENES)} corpus scenes failed to "
            f"render, so the measurement is incomplete: {summary}")
    return observations


def run_corpus(video_settings=None):
    """Measure the whole corpus; returns (observations, corpus labels)."""
    observations = []
    observations.extend(collect_postprocess_observations())
    observations.extend(collect_render_observations(video_settings))
    labels = (
        [f"postprocess:{label}" for (label, *_rest) in _postprocess_configs()]
        + [f"render:{name}" for name, _scene, _over in RENDER_SCENES]
    )
    return observations, labels
