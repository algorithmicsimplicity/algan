"""Where the path tracer's time goes: the first PT performance harness.

``SETTINGS.raytracing.samples_per_pixel > 1`` selects the Monte Carlo path
tracer, and until now nothing measured it. This script renders one of three
small scenes through it and prints, off the **warm** run, a compact per-stage
table: the six kernels the PT loop is made of, the denoiser, everything else
that ran on the device, and the host remainder. That is what lets the
path-tracer roadmap be ranked by measured cost rather than by guesswork.

Scenes (``--scene``)
--------------------
lit
    The lit-3-D case: a floor, two spheres and two boxes in
    ``MeshStandardMaterial`` / ``MeshPhysicalMaterial``, three lights (two
    point, one spot) and ``shadows=True``. **Every mob is fully opaque**
    (``set_opacity(1)`` on each, because ``Cube``/``Prism`` default to
    ``fill_opacity=0.75``), which is the precondition for the all-opaque
    any-hit shadow and closest-hit traversal rollouts: a single translucent
    solid in the batch takes them off and the numbers stop being about the
    case they claim to be about.
many_lights
    The same solids under 64 ``PointLight``s on a ring, which is the
    fallback's headline case. Note that ``max_shadow_lights`` (16,
    ``ALGAN_MAX_SHADOW_LIGHTS``) caps how many of them cast shadows -- the
    rest light without shadowing and the render records a truncation. That is
    the shipped behaviour, so it is what this arm measures; raise the env var
    if the question is what 64 *shadowed* lights cost.
text_2d
    Unlit 2-D content: a ``Text`` paragraph plus overlapping translucent
    ``Square``/``Circle`` shapes. This is the deterministic camera-segment
    peel that every one of the ``spp`` samples repeats, so it sizes the part
    of the path tracer that does not converge with more samples -- it just
    costs ``spp`` times as much.
    ``Text`` needs ``manimpango``, which is the ``pango`` extra and is **not**
    in the default environment (``uv sync --extra pango`` installs it; the
    Kaggle image gets it from ``make_notebook.py``'s ``--extras`` default).
    Without it the scene drops the paragraph, keeps the shapes, and says so
    loudly in the log: a quieter fallback would silently change what the arm
    measures, and shapes-only numbers must not be compared with numbers that
    had text in them.

Each scene animates for one second at 5 fps, so a run renders several frames
rather than one (a single frame hides the per-batch host work).

Arms
----
One process per arm. Settings behind a ``ti.static`` gate are baked when the
kernel compiles, so two arms in one process would silently share the first
one's code (see ``agent_guidance/taichi.md``); the CLI knobs and the
``ALGAN_*`` codegen variables therefore both ride in the profile tag, which
keeps two arms of one session from sharing output filenames -- and with them
the digests ``scripts/kaggle/runner.py`` reports.

``--deterministic`` renders the same scene at ``samples_per_pixel=1`` through
the wavefront renderer instead. The path tracer's cost is only meaningful
against what the user was already paying, and that is this number.

Usage
-----
::

    uv run python benchmarks/performance/pt_baseline.py --scene lit
    uv run python benchmarks/performance/pt_baseline.py --scene many_lights \
        --resolution 96x54 --spp 4
    uv run python benchmarks/performance/pt_baseline.py --scene lit --deterministic

The default 320x180 / 16 spp / 4 bounces runs in a few minutes on a CPU box;
96x54 at 4 spp is the "does it work" size. The first PT render in a process
pays the megakernel's cold compile, which is why RUN 1 is never the
measurement -- read RUN 2, as ``agent_guidance/gpu_harnesses.md`` says.

The last line is ``RESULTS <json>``: the stage table, the launch counts and
the resolved configuration, on one line, for ``scripts/kaggle/read_output.py``
to lift out of a harness log. The script exits 0 on success and non-zero if
the render did not go through the renderer the arm asked for -- ``pt_shade``
launches only under the path tracer, so its count is what proves an arm
measured the renderer it named.

First readings (4 vCPU linux box, ``arch=x64``, 96x54, ``--spp 4``, warm RUN 2,
so a plumbing reference and not a benchmark)::

    lit          0.76 s   pt_shade 10 launches / 2 waves, 11.4 ms device
    many_lights  0.95 s   pt_shade 10 launches / 2 waves, 15.6 ms device
    text_2d      0.35 s   pt_shade  3 launches / 2 waves,  9.3 ms device

Two things to carry into a real run from those. **At this resolution the host
side is ~85% of the wall clock** -- geometry prep, the merge and the encode --
so a table taken at 96x54 ranks host work, not the path tracer; use the
default 320x180 (or larger) when the kernels are the question. And
``many_lights`` costs about what ``lit`` does, because next-event estimation
samples ``pt_light_samples`` emitters per bounce whatever the rig holds: the
light count moves variance, not per-bounce work. That is a result about the
current estimator, and it is the sort of thing this harness exists to say out
loud rather than assume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *  # noqa: F403
from algan.rendering.taichi_runtime import _live_arch, _taichi_arch
from algan.settings import _startup
from algan.utils.profiling_utils import TIMERS, profile_scene

# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
DURATION = 1.0


def _opaque(mob):
    """Force full opacity, whatever the class's default fill is.

    ``Cube``/``Prism`` ship ``fill_opacity=0.75``. A translucent solid in the
    batch is not a small error here: it takes the all-opaque traversal
    rollouts off for the whole batch.
    """
    mob.set_opacity(1.0)
    return mob


def _solids():
    """The lit scene's geometry: floor, two spheres, two boxes. All opaque."""
    floor = Prism(width=10.0, height=0.3, depth=6.0)
    floor.set_material(MeshStandardMaterial(color=WHITE, roughness=0.9))
    _opaque(floor).move(DOWN * 1.7)
    floor.spawn(animate=False)

    ball = Sphere(radius=0.9)
    ball.set_material(MeshStandardMaterial(color=BLUE, roughness=0.35, metalness=0.1))
    _opaque(ball).move(LEFT * 1.9 + DOWN * 0.6)
    ball.spawn(animate=False)

    metal_ball = Sphere(radius=0.7)
    metal_ball.set_material(
        MeshStandardMaterial(color=WHITE, roughness=0.25, metalness=1.0)
    )
    _opaque(metal_ball).move(RIGHT * 2.1 + DOWN * 0.8)
    metal_ball.spawn(animate=False)

    box = Prism(width=1.2, height=1.2, depth=1.2)
    box.set_material(MeshStandardMaterial(color=RED, roughness=0.6))
    _opaque(box).move(DOWN * 0.9)
    box.spawn(animate=False)

    coated = Prism(width=1.0, height=1.6, depth=1.0)
    coated.set_material(
        MeshPhysicalMaterial(color=GREEN, clearcoat=1.0, clearcoat_roughness=0.1)
    )
    _opaque(coated).move(LEFT * 0.2 + UP * 1.1 + IN * 1.0)
    coated.spawn(animate=False)

    return ball, box, coated


def scene_lit():
    """A few opaque PBR solids under three lights, with shadows on."""
    SETTINGS.raytracing.set(shadows=True)
    Scene.set_background(BLACK)
    Scene.clear_lights()

    with Off():
        PointLight(
            location=UP * 3.5 + OUT * 4.0 + LEFT * 2.0, color=WHITE, intensity=2.0
        ).spawn(animate=False)
        PointLight(
            location=UP * 2.0 + OUT * 3.0 + RIGHT * 3.0, color=WHITE, intensity=1.0
        ).spawn(animate=False)
        SpotLight(
            location=UP * 4.5 + OUT * 1.5,
            target=ORIGIN,
            color=WHITE,
            intensity=2.5,
            cone_angle=45.0,
            penumbra=0.35,
        ).spawn(animate=False)
        ball, box, coated = _solids()

    with Sync(runtime=DURATION):
        ball.move(UP * 0.9)
        box.rotate(60, UP)
        coated.rotate(45, RIGHT)


def scene_many_lights():
    """The same opaque solids under 64 point lights on a ring."""
    SETTINGS.raytracing.set(shadows=True)
    Scene.set_background(BLACK)
    Scene.clear_lights()

    import math

    num_lights = 64
    with Off():
        for i in range(num_lights):
            angle = 2.0 * math.pi * i / num_lights
            PointLight(
                location=(
                    RIGHT * (5.0 * math.cos(angle))
                    + UP * (2.5 + 2.0 * math.sin(angle))
                    + OUT * (4.0 * math.sin(angle * 2.0))
                ),
                color=WHITE,
                # 64 lights at full strength is a white frame, which measures
                # nothing: keep the total roughly the lit scene's.
                intensity=4.0 / num_lights,
            ).spawn(animate=False)
        ball, box, coated = _solids()

    with Sync(runtime=DURATION):
        ball.move(UP * 0.9)
        box.rotate(60, UP)
        coated.rotate(45, RIGHT)


def scene_text_2d():
    """Unlit 2-D: a text paragraph and overlapping translucent shapes."""
    SETTINGS.raytracing.set(shadows=False)
    Scene.set_background(BLACK)

    with Off():
        back = Square(size=7.0, color=BLUE).set_opacity(0.55)
        back.move(IN * 2.0)
        back.spawn(animate=False)

        red = Square(size=2.8, color=RED).set_opacity(0.5)
        red.move(LEFT * 1.1 + UP * 0.4)
        red.spawn(animate=False)

        green = Circle(radius=1.5, color=GREEN).set_opacity(0.5)
        green.move(RIGHT * 0.5 + DOWN * 0.4)
        green.spawn(animate=False)

        yellow = Circle(radius=1.1, color=YELLOW).set_opacity(0.45)
        yellow.move(RIGHT * 2.2 + UP * 1.0)
        yellow.spawn(animate=False)

        label = None
        try:
            label = Text(
                "The deterministic camera-segment peel\n"
                "is repeated by every path sample.",
                font_size=32,
            )
            label.move(DOWN * 2.2)
            label.spawn(animate=False)
        except Exception as exc:  # manimpango missing, or a font failure
            label = None
            print(
                "!! pt_baseline: Text is unavailable on this box "
                f"({type(exc).__name__}: {exc}); the text_2d arm is rendering "
                "SHAPES ONLY and its numbers are not comparable to a run that "
                "had text. Install the 'pango' extra.",
                flush=True,
            )

    with Sync(runtime=DURATION):
        red.move(RIGHT * 1.2)
        green.move(UP * 0.6)
        if label is not None:
            label.move(RIGHT * 0.8)


SCENES = {
    "lit": scene_lit,
    "many_lights": scene_many_lights,
    "text_2d": scene_text_2d,
}


# ---------------------------------------------------------------------------
# The stage table
# ---------------------------------------------------------------------------
#: Report row -> the kernel names that feed it. Both spellings of the two
#: arena-packed kernels are listed: the launch site calls ``pt_shade`` /
#: ``wavefront_traverse_events``, which are plain python wrappers, and what the
#: profiler hooks and Taichi's own profiler see is the ``_arena`` kernel behind
#: them. Matching only the launch-site name would report zeros.
KERNEL_GROUPS = (
    ("pt_generate", ("pt_generate",)),
    (
        "wavefront_traverse_events",
        ("wavefront_traverse_events", "wavefront_traverse_events_arena"),
    ),
    ("pt_shade", ("pt_shade", "pt_shade_arena")),
    ("compact_ray_slots", ("compact_ray_slots",)),
    ("pt_reduce", ("pt_reduce",)),
    ("finalize_samples", ("finalize_samples",)),
)
DENOISE_STAGE = "denoise (torch UNet)"
KERNEL_PREFIX = "kernel: "


def install_denoiser_hook():
    """Time the denoiser as a stage of its own.

    It is a torch network, not a Taichi kernel, so no hook in
    ``profiling_utils`` reaches it and its cost lands unattributed inside the
    render total. Wrapping ``Denoiser.__call__`` on the class is enough: the
    tracer calls the instance, and python looks a dunder up on the type.
    Returns True if the hook went on.
    """
    try:
        from algan.rendering.denoise.denoise import Denoiser

        TIMERS.wrap_function(Denoiser, "__call__", DENOISE_STAGE)
        return True
    except Exception as exc:  # pragma: no cover - degrade, never break the run
        print(f"[pt_baseline] could not hook the denoiser: {exc}", flush=True)
        return False


def _device_ms(res):
    """``{kernel name: (total ms, records)}`` from Taichi's kernel profiler."""
    return {r["name"]: (r["total_ms"], r["records"]) for r in res.get("kernel_gpu", [])}


def stage_table(res):
    """Build the warm-run stage rows plus the run's summary counters."""
    walls, launches = {}, {}
    for name, secs in res["times"].items():
        if name.startswith(KERNEL_PREFIX):
            walls[name[len(KERNEL_PREFIX) :]] = secs
            launches[name[len(KERNEL_PREFIX) :]] = res["counts"].get(name, 0)
    gpu = _device_ms(res)

    rows, claimed = [], set()
    for label, aliases in KERNEL_GROUPS:
        present = [a for a in aliases if a in walls or a in gpu]
        claimed.update(present)
        rows.append(
            {
                "stage": label,
                "launches": sum(launches.get(a, 0) for a in present),
                "wall_s": sum(walls.get(a, 0.0) for a in present),
                "device_ms": sum(gpu.get(a, (0.0, 0))[0] for a in present),
                "records": sum(gpu.get(a, (0.0, 0))[1] for a in present),
            }
        )

    rest = [k for k in sorted(set(walls) | set(gpu)) if k not in claimed]
    rows.append(
        {
            "stage": f"other kernels ({len(rest)})",
            "launches": sum(launches.get(k, 0) for k in rest),
            "wall_s": sum(walls.get(k, 0.0) for k in rest),
            "device_ms": sum(gpu.get(k, (0.0, 0))[0] for k in rest),
            "records": sum(gpu.get(k, (0.0, 0))[1] for k in rest),
        }
    )

    denoise_wall = res["times"].get(DENOISE_STAGE, 0.0)
    rows.append(
        {
            "stage": DENOISE_STAGE,
            "launches": res["counts"].get(DENOISE_STAGE, 0),
            "wall_s": denoise_wall,
            "device_ms": 0.0,
            "records": 0,
        }
    )
    kernel_wall = sum(walls.values())
    rows.append(
        {
            "stage": "host: prep, merge, encode, ...",
            "launches": 0,
            "wall_s": max(0.0, res["total"] - kernel_wall - denoise_wall),
            "device_ms": 0.0,
            "records": 0,
        }
    )

    summary = {
        # pt_shade is launched once per bounce iteration per wave, so its
        # launch count is the iteration count; pt_reduce is launched once per
        # wave, which makes it the wave counter.
        "pt_shade_launches": next(
            r["launches"] for r in rows if r["stage"] == "pt_shade"
        ),
        "waves": next(r["launches"] for r in rows if r["stage"] == "pt_reduce"),
        "other_kernel_names": rest,
        "kernel_wall_s": kernel_wall,
        "device_profiler": bool(gpu),
    }
    return rows, summary


def run_label(runs):
    """What the last run is: only a second one is warm.

    ``--runs 1`` is a plumbing check, not a measurement -- the one run it does
    pays the path tracer's cold megakernel compile -- so the table has to say
    so rather than print "warm" over a JIT number.
    """
    return f"RUN {runs} (warm)" if runs >= 2 else "RUN 1 (COLD -- includes the JIT)"


def print_stage_table(res, rows, summary, runs=2):
    total = res["total"]
    print("")
    print("=" * 78)
    print(f"{run_label(runs)} stage table -- end-to-end {total:.3f}s")
    print("=" * 78)
    if not summary["device_profiler"]:
        print(
            "  (Taichi's kernel profiler reported nothing on this runtime: the "
            "device ms column is empty and only wall time is a measurement.)"
        )
    print(
        f"  {'stage':<32}{'launches':>9}{'wall s':>9}{'wall %':>8}"
        f"{'device ms':>11}{'recs':>7}"
    )
    for row in rows:
        pct = 100.0 * row["wall_s"] / total if total else 0.0
        dev = f"{row['device_ms']:>11.3f}" if row["device_ms"] else f"{'-':>11}"
        recs = f"{row['records']:>7}" if row["records"] else f"{'-':>7}"
        launches = f"{row['launches']:>9}" if row["launches"] else f"{'-':>9}"
        print(
            f"  {row['stage']:<32}{launches}{row['wall_s']:>9.3f}{pct:>7.1f}%{dev}{recs}"
        )
    print(f"  {'TOTAL':<32}{'':>9}{total:>9.3f}{100.0:>7.1f}%")
    waves = summary["waves"]
    iters = summary["pt_shade_launches"]
    per_wave = f"{iters / waves:.2f}" if waves else "n/a"
    print(
        f"  pt_shade launches (bounce iterations): {iters}   "
        f"waves (pt_reduce launches): {waves}   iterations/wave: {per_wave}"
    )
    if summary["other_kernel_names"]:
        print(f"  other kernels: {', '.join(summary['other_kernel_names'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#: Codegen switches that ride in the tag, as ``nn_ablation.py`` does it: two
#: arms that share a tag share their output filenames, and then every digest
#: the Kaggle runner reports comes from whichever step ran last.
CODEGEN_ENV = (
    ("ALGAN_PT_WAVE", "wave"),
    ("ALGAN_SHADOW_ANYHIT", "ah"),
    ("ALGAN_OPT_LEVEL", "opt"),
    ("ALGAN_ADV_OPT", "adv"),
    ("ALGAN_MAX_SHADOW_LIGHTS", "msl"),
)


def parse_resolution(text):
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"resolution must look like 320x180, got {text!r}"
        ) from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", choices=sorted(SCENES), default="lit")
    parser.add_argument("--resolution", type=parse_resolution, default="320x180")
    parser.add_argument("--spp", type=int, default=16, help="samples per pixel")
    parser.add_argument("--bounces", type=int, default=4, help="max_bounces")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="render at samples_per_pixel=1 (the wavefront renderer) for reference",
    )
    parser.add_argument("--runs", type=int, default=2, help="cold + warm (default 2)")
    parser.add_argument(
        "--no-kernel-profiler",
        action="store_true",
        help="skip Taichi's kernel profiler (wall time only)",
    )
    parser.add_argument("--tag", default="", help="extra text for the profile tag")
    # None everywhere below means "leave the shipped default alone".
    parser.add_argument("--pt-light-samples", type=int, default=None)
    parser.add_argument("--pt-rr-start-bounce", type=int, default=None)
    parser.add_argument("--pt-firefly-clamp", type=float, default=None)
    parser.add_argument(
        "--denoise",
        type=int,
        choices=(0, 1),
        default=None,
        help="force the denoiser on (1) or off (0)",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=None,
        help="pin SETTINGS.computing.available_memory_override, which pins the "
        "frame batching and the PT tile/wave split -- worth setting when two "
        "arms must be comparable",
    )
    args = parser.parse_args(argv)
    if isinstance(args.resolution, str):
        args.resolution = parse_resolution(args.resolution)
    return args


def knob_values(args):
    """Every knob this arm resolved, in the order they are printed."""
    return {
        "scene": args.scene,
        "mode": "deterministic" if args.deterministic else "path_tracer",
        "resolution": f"{args.resolution[0]}x{args.resolution[1]}",
        "fps": args.fps,
        "samples_per_pixel": 1 if args.deterministic else args.spp,
        "max_bounces": args.bounces,
        "pt_light_samples": args.pt_light_samples,
        "pt_rr_start_bounce": args.pt_rr_start_bounce,
        "pt_firefly_clamp": args.pt_firefly_clamp,
        "denoise": args.denoise,
        "available_memory_mb": args.memory_mb,
    }


def build_tag(args):
    parts = [
        "pt",
        args.scene,
        "det" if args.deterministic else f"spp{args.spp}",
        f"b{args.bounces}",
        f"{args.resolution[0]}x{args.resolution[1]}",
    ]
    if args.pt_light_samples is not None:
        parts.append(f"ls{args.pt_light_samples}")
    if args.pt_rr_start_bounce is not None:
        parts.append(f"rr{args.pt_rr_start_bounce}")
    if args.pt_firefly_clamp is not None:
        parts.append(f"fc{args.pt_firefly_clamp:g}")
    if args.denoise is not None:
        parts.append(f"dn{args.denoise}")
    if args.memory_mb is not None:
        parts.append(f"mem{args.memory_mb}")
    if args.tag:
        parts.append(args.tag)
    parts += [
        f"{short}{os.environ[name]}"
        for name, short in CODEGEN_ENV
        if name in os.environ
    ]
    return "_".join(parts)


def apply_settings(args):
    """Push the arm's configuration into ``SETTINGS`` (re-run per profiling run)."""
    SETTINGS.raytracing.set(
        samples_per_pixel=1 if args.deterministic else args.spp,
        max_bounces=args.bounces,
    )
    if args.denoise is not None:
        SETTINGS.raytracing.set(denoise=bool(args.denoise))
    experimental = {}
    if args.pt_light_samples is not None:
        experimental["pt_light_samples"] = args.pt_light_samples
    if args.pt_rr_start_bounce is not None:
        experimental["pt_rr_start_bounce"] = args.pt_rr_start_bounce
    if args.pt_firefly_clamp is not None:
        experimental["pt_firefly_clamp"] = args.pt_firefly_clamp
    if experimental:
        SETTINGS.raytracing.experimental.set(**experimental)
    if args.memory_mb is not None:
        SETTINGS.computing.set(
            available_memory_override=int(args.memory_mb) * 1024 * 1024
        )


def print_header(args, tag):
    device = _startup.render_device()
    arch = _taichi_arch()
    print("=" * 78)
    print("pt_baseline -- path-tracer performance harness")
    print("=" * 78)
    print(f"  arm tag          {tag}")
    print(f"  render device    {getattr(device, 'type', device)}")
    print(f"  taichi arch      {getattr(arch, 'name', arch)} (selected)")
    for name, value in knob_values(args).items():
        shown = "(default)" if value is None else value
        print(f"  {name:<16} {shown}")
    for name, _short in CODEGEN_ENV:
        print(f"  {name:<16} {os.environ.get(name, '(default)')}")
    print(f"  runs             {args.runs}")
    print("", flush=True)


def main(argv=None):
    args = parse_args(argv)
    tag = build_tag(args)
    print_header(args, tag)

    settings = PREVIEW.set(resolution=args.resolution, frames_per_second=args.fps)
    denoise_hooked = install_denoiser_hook()

    # What the scene actually built, read back after it is authored. The
    # many_lights arm is about a light count, and a rig that silently
    # registered one light would otherwise be invisible in the numbers.
    authored = {}

    def scene():
        apply_settings(args)
        SCENES[args.scene]()
        authored["lights"] = len(Scene.get_light_sources())

    results = profile_scene(
        scene,
        settings,
        tag,
        runs=args.runs,
        kernel_profiler=not args.no_kernel_profiler,
        # The encoder is inside the measurement; a slow software preset would
        # be most of a small render's wall time on a CPU box.
        save_video_kwargs={"ffmpeg_params": ["-crf", "17", "-preset", "ultrafast"]},
    )
    if not results:
        print("!! no profiling runs completed", flush=True)
        return 1

    warm = results[-1]
    rows, summary = stage_table(warm)
    print_stage_table(warm, rows, summary, runs=args.runs)
    print(f"  lights in the scene: {authored.get('lights')}")

    live = _live_arch()
    payload = {
        "benchmark": "pt_baseline",
        "tag": tag,
        "device": str(getattr(_startup.render_device(), "type", "")),
        "taichi_arch": str(getattr(live, "name", live)),
        "knobs": knob_values(args),
        "runs": args.runs,
        "reported_run": run_label(args.runs),
        "lights": authored.get("lights"),
        "denoiser_hooked": denoise_hooked,
        "cold_total_s": round(results[0]["total"], 3),
        "warm_total_s": round(warm["total"], 3),
        "peak_alloc_mb": round(warm["peak_alloc_mb"], 1),
        "pt_shade_launches": summary["pt_shade_launches"],
        "waves": summary["waves"],
        "device_profiler": summary["device_profiler"],
        "stages": {
            row["stage"]: {
                "launches": row["launches"],
                "wall_s": round(row["wall_s"], 4),
                "device_ms": round(row["device_ms"], 3),
            }
            for row in rows
        },
    }

    # The arm has to have rendered through the renderer it claims: a path-tracer
    # arm that silently fell back to the wavefront renderer would report a
    # perfectly plausible table about the wrong thing. pt_shade runs only under
    # the path tracer, so its launch count is the check.
    launched_pt = summary["pt_shade_launches"] > 0
    ok = launched_pt is not args.deterministic
    payload["ok"] = ok
    print("")
    print("RESULTS " + json.dumps(payload, separators=(",", ":")), flush=True)
    if not ok:
        expected = "deterministic wavefront" if args.deterministic else "path tracer"
        print(
            f"!! this arm asked for the {expected} renderer but "
            f"pt_shade launched {summary['pt_shade_launches']} times; "
            "the numbers above are about the other renderer.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
