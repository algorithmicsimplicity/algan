"""Price ``torch.compile`` for one function at a time, on a real render's inputs.

``benchmarks/_torch_compile_ab.py`` measures the *pipeline*: one wall time per
arm. That answers "is the shipped set worth it" and nothing else. This script
answers the other question -- **would compiling this particular function pay,
and would it change the picture** -- for any function named on the command
line, without editing it.

Each candidate is wrapped for the duration of one ordinary render. Every call
runs **both** arms on the same inputs: the eager original (timed, and its
result is what the render consumes, so the render is unperturbed) and the
compiled wrapper (timed, and its result compared against the eager one, tensor
by tensor). What comes out per candidate is calls, warm per-call time in each
arm, the speedup, and whether the two arms agreed **bit for bit** -- which is
the bar this codebase holds compiled regions to, because an ulp in anything a
subdivision level is measured from is a whole level (``rendering/logical_pn.py``
has that account).

The render is roughly twice its normal cost with candidates installed, and the
per-call timings include no Dynamo compile: the first call of each arm is
reported separately as ``cold`` and excluded from the warm statistics.

Usage::

    uv run python benchmarks/_compile_candidates_ab.py [--scene SCENE]
        [--group pn|geometry|timeline|all] [--candidate module:function ...]
        [--quality PREVIEW] [--json out.json]

``--group`` names a curated set (``--list`` prints them); ``--candidate``
adds any ``module:qualname`` on top, so a function this file has never heard
of can be priced without editing it. Patching is by identity: every loaded
``algan`` module holding the original function object is repointed, so a
``from x import f`` call site is covered too.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

# A warm daemon would carry state between runs, and this is a measurement.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_PROGRESS", "none")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "benchmarks" / "_pn_geometry_scene.py"
FONT_DIR = ROOT / "tests" / "assets" / "fonts"

#: Curated candidate sets. Every entry is ``module:qualname`` of a function
#: that is *not* decorated today; the point of the script is to price it.
GROUPS = {
    # The three PN control-net builders logical_pn.py keeps eager on purpose,
    # plus the two small per-patch helpers next to them.
    "pn": [
        "algan.rendering.logical_pn:logical_pn_control_points",
        "algan.rendering.logical_pn:logical_pn_normal_control_points",
        "algan.rendering.logical_pn:logical_pn_edge_control_points",
        "algan.rendering.logical_pn:evaluate_cubic_curve",
        "algan.rendering.logical_pn:mean_patch_edge_length",
    ],
    # Geometry helpers: elementwise chains over the batch's big arrays, in the
    # bezier-circuit build and the PN level searches.
    "geometry": [
        "algan.rendering.raytracing.primitives:_evaluate_cubic_bezier_batch",
        "algan.rendering.raytracing.primitives:_evaluate_cubic_bezier_derivative_batch",
        "algan.rendering.raytracing.primitives:_uniform_cubic_subcurves",
        "algan.rendering.raytracing.primitives:_point_to_segment_distance_squared",
        "algan.rendering.raytracing.primitives:_circuit_parity_gathered",
        "algan.rendering.raytracing.primitives:_bezier_connection_visibility",
        "algan.rendering.raytracing.primitives:"
        "LogicalPNTrianglePrimitive._project_to_output_pixels",
    ],
    # Timeline materialization: what a frame batch runs per attribute.
    "timeline": [
        "algan.animation_timeline.timeline:generate_array_states",
        "algan.animation_timeline.timeline:_query_row_states",
        "algan.animation_timeline.timeline:SegmentWindow.evaluate",
    ],
}


#: Sentinel for "the attribute was inherited, not in the owner's own dict".
_MISSING = object()


def _register_test_fonts():
    """Scenes that pin ``Algan Test Sans`` need the vendored faces visible."""
    try:
        import manimpango
    except ImportError:
        return
    for face in sorted(FONT_DIR.glob("*.ttf")):
        manimpango.register_font(str(face))


def _load_scene(scene_file):
    spec = importlib.util.spec_from_file_location("_algan_cand_scene", scene_file)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_algan_cand_scene", None)


def _resolve(spec):
    """``module:qualname`` -> ``(owner, attribute, function)``.

    ``owner`` is the module for a plain function and the class for a method, so
    the caller can put the probe back where it found it.
    """
    module_name, _, qualname = spec.partition(":")
    if not qualname:
        raise SystemExit(f"candidate {spec!r} is not 'module:qualname'")
    owner = importlib.import_module(module_name)
    parts = qualname.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1], getattr(owner, parts[-1])


def _compare(a, b):
    """``(bit_identical, max_abs_difference)`` over two arbitrary results."""
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, float("inf")
        if a.dtype == torch.bool or not a.is_floating_point():
            same = bool(torch.equal(a, b))
            return same, 0.0 if same else 1.0
        same = bool(torch.equal(a, b))
        diff = (a.float() - b.float()).abs()
        return same, float(diff.max()) if diff.numel() else 0.0
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False, float("inf")
        same, worst = True, 0.0
        for x, y in zip(a, b):
            s, d = _compare(x, y)
            same &= s
            worst = max(worst, d)
        return same, worst
    if hasattr(a, "_fields") and hasattr(b, "_fields"):  # NamedTuple
        return _compare(tuple(a), tuple(b))
    return a == b, 0.0


class Candidate:
    """One function measured in both arms on every call the render makes."""

    def __init__(self, spec):
        self.spec = spec
        self.owner, self.attribute, self.original = _resolve(spec)
        # What ``owner.__dict__`` holds, so uninstall puts back exactly what
        # was there -- a ``staticmethod`` object stays a staticmethod.
        self.raw = self.owner.__dict__.get(self.attribute, _MISSING)
        self.is_static = isinstance(self.raw, staticmethod)
        self.eager_times = []
        self.compiled_times = []
        self.calls = 0
        self.mismatches = 0
        self.worst_difference = 0.0
        self.failure = ""
        self.wrapper = None

    def install(self):
        from algan.utils.torch_compile import compiled

        target = self.original  # already the plain function, static or not
        self.wrapper = compiled(target)

        def probe(*args, **kwargs):
            started = time.perf_counter()
            eager = target(*args, **kwargs)
            self.eager_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            try:
                candidate = self.wrapper(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- a candidate may not compile
                self.compiled_times.append(float("nan"))
                if not self.failure:
                    self.failure = f"{type(exc).__name__}: {exc}"
                self.calls += 1
                return eager
            self.compiled_times.append(time.perf_counter() - started)
            same, difference = _compare(eager, candidate)
            self.mismatches += not same
            self.worst_difference = max(self.worst_difference, difference)
            self.calls += 1
            # The render consumes the EAGER result, so installing the probe
            # cannot change a single pixel of the frames it measures on.
            return eager

        probe.__name__ = getattr(self.original, "__name__", self.attribute)
        installed = staticmethod(probe) if self.is_static else probe
        setattr(self.owner, self.attribute, installed)
        # A module-level function is also bound by name in every module that
        # imported it; repoint those too, by identity so nothing else moves.
        self.rebound = []
        if not isinstance(self.owner, type):
            for module in list(sys.modules.values()):
                name = getattr(module, "__name__", "")
                if not name.startswith("algan.") or module is self.owner:
                    continue
                if getattr(module, self.attribute, None) is self.original:
                    self.rebound.append(module)
                    setattr(module, self.attribute, probe)

    def uninstall(self):
        if self.raw is _MISSING:
            delattr(self.owner, self.attribute)
        else:
            setattr(self.owner, self.attribute, self.raw)
        for module in getattr(self, "rebound", ()):
            setattr(module, self.attribute, self.original)

    def summary(self):
        """Warm per-call medians, excluding each arm's first (compiling) call."""
        eager = self.eager_times[1:] or self.eager_times
        comp = [t for t in self.compiled_times[1:] if t == t] or [
            t for t in self.compiled_times if t == t
        ]
        record = {
            "candidate": self.spec,
            "calls": self.calls,
            "eager_ms": statistics.median(eager) * 1e3 if eager else float("nan"),
            "eager_total_ms": sum(self.eager_times) * 1e3,
            "compiled_ms": statistics.median(comp) * 1e3 if comp else float("nan"),
            "compiled_cold_ms": (
                self.compiled_times[0] * 1e3 if self.compiled_times else float("nan")
            ),
            "bit_identical": self.calls > 0 and self.mismatches == 0,
            "mismatched_calls": self.mismatches,
            "worst_difference": self.worst_difference,
            "failure": self.failure,
        }
        record["speedup"] = (
            record["eager_ms"] / record["compiled_ms"]
            if record["compiled_ms"]
            else float("nan")
        )
        record["saved_total_ms"] = (
            record["eager_total_ms"]
            - record["compiled_ms"] * max(0, record["calls"] - 1)
            - record["eager_ms"]
            if record["calls"]
            else 0.0
        )
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--group", default="all", help=f"one of {sorted(GROUPS)}, or all"
    )
    parser.add_argument(
        "--candidate", action="append", default=[], help="extra module:qualname"
    )
    parser.add_argument("--quality", default="PREVIEW", help="a video preset name")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="print the groups and exit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks" / "algan_outputs" / "compile_candidates",
    )
    args = parser.parse_args()

    if args.list:
        for name, specs in GROUPS.items():
            print(f"{name}:")
            for spec in specs:
                print(f"  {spec}")
        return 0

    specs = list(args.candidate)
    if args.group == "all":
        for group in GROUPS.values():
            specs.extend(group)
    elif args.group in GROUPS:
        specs.extend(GROUPS[args.group])
    elif args.group:
        raise SystemExit(f"unknown group {args.group!r}; try --list")

    _register_test_fonts()
    args.scene = args.scene.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.json is not None:
        args.json = args.json.resolve()

    import torch

    import algan
    from algan import SETTINGS, Scene
    from algan.scene_manager import SceneManager
    from algan.utils.torch_compile import torch_compile_support

    quality = getattr(algan, args.quality)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.paths.set(
        output_root=str(out_dir),
        output_directory=".",
        cache_directory=str(out_dir / "cache"),
    )
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    # Both arms of every candidate run whatever the switch says; the switch
    # itself must be ON or the compiled arm is the eager one again.
    SETTINGS.computing.set(torch_compile=True)
    os.chdir(args.scene.parent)

    supported, reason = torch_compile_support()
    print(f"platform      : {platform.platform()} python {platform.python_version()}")
    print(f"torch         : {torch.__version__}")
    print(f"render device : {SETTINGS.computing.render_device}")
    print(f"compile ok?   : {supported}{'' if supported else ' -- ' + reason}")
    print(f"scene         : {args.scene}")
    print(f"quality       : {args.quality}")
    print(f"candidates    : {len(specs)}", flush=True)

    candidates = [Candidate(spec) for spec in specs]
    for candidate in candidates:
        candidate.install()
    SceneManager.reset()
    started = time.perf_counter()
    try:
        with Scene() as scene:
            _load_scene(args.scene)
            scene.save_video(
                out_dir / "probe.mp4",
                video_settings=quality,
                overwrite=True,
                animate_fade_out=True,
            )
    finally:
        for candidate in candidates:
            candidate.uninstall()
    elapsed = time.perf_counter() - started
    print(f"render (both arms on every call): {elapsed:.2f}s\n", flush=True)

    records = [candidate.summary() for candidate in candidates]
    header = (
        f"{'candidate':<62}{'calls':>6}{'eager ms':>10}{'comp ms':>9}"
        f"{'speedup':>9}{'saved ms':>10}  parity"
    )
    print(header)
    print("-" * len(header))
    for record in sorted(records, key=lambda r: -r["saved_total_ms"]):
        if not record["calls"]:
            parity = "never called"
        elif record["failure"]:
            parity = f"did not compile ({record['failure'][:60]})"
        elif record["bit_identical"]:
            parity = "bit-identical"
        else:
            parity = (
                f"DIFFERS on {record['mismatched_calls']}/{record['calls']} "
                f"calls, max {record['worst_difference']:.3e}"
            )
        print(
            f"{record['candidate'].split(':')[-1]:<62}{record['calls']:>6}"
            f"{record['eager_ms']:>10.3f}{record['compiled_ms']:>9.3f}"
            f"{record['speedup']:>9.2f}{record['saved_total_ms']:>10.1f}  {parity}"
        )

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "scene": str(args.scene),
                    "quality": args.quality,
                    "probe_render_seconds": elapsed,
                    "candidates": records,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
