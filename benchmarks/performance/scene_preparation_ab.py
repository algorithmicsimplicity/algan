"""Alternate scene-construction changes while keeping the renderer identical.

The baseline methods come from git, so this also works in a fresh checkout.
Use --mode author for CPU timings, --profile for diagnostic cProfile dumps,
and --mode render to measure preparation and check lossless frame parity.
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import gc
import importlib
import json
import os
import pstats
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

import torch  # noqa: E402

import algan  # noqa: E402
from algan import Scene  # noqa: E402
from algan.render_loop import RenderLoopMixin  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.settings._startup import render_device  # noqa: E402
from benchmarks._sheet_stream_check import compare_videos  # noqa: E402


def baseline_method(ref, module, cls, name):
    path = module.__name__.replace(".", "/") + ".py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    body = next(
        n.body
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == cls.__name__
    )
    method = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)

    class ExplicitSuper(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "super"
                and not node.args
            ):
                node.args = [
                    ast.Name(id="_baseline_class", ctx=ast.Load()),
                    ast.Name(id="self", ctx=ast.Load()),
                ]
            return node

    tree = ast.fix_missing_locations(
        ExplicitSuper().visit(ast.Module(body=[method], type_ignores=[]))
    )
    namespace = dict(vars(module), _baseline_class=cls)
    exec(compile(tree, f"{ref}/{path}", "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("author", "render"), default="author")
    parser.add_argument(
        "--scene", choices=("explainer", "graphics"), default="explainer"
    )
    parser.add_argument("--quality", choices=("MD", "UHD", "PREVIEW"), default="MD")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--arms", default="legacy,current")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--tag", default="scene_preparation_ab")
    parser.add_argument(
        "--baseline-ref", default="d18ec56f7d0ed1cf3aa13d39b9c7ce3954e09a6a"
    )
    args = parser.parse_args()
    names = args.arms.split(",")
    if (
        args.frames < 1
        or args.pairs < 1
        or args.warmups < 1
        or "legacy" not in names
        or any(n not in ("legacy", "current", "grid", "text", "move") for n in names)
    ):
        parser.error("positive counts and known arms including legacy required")
    device = render_device()
    if args.mode == "render":
        if device.type != "cuda":
            raise RuntimeError(f"CUDA required, got {device}")
        torch.cuda.set_device(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    targets = []
    for key, module_name, class_name, name in (
        ("grid", "algan.mobs.group", "Group", "arrange_in_grid"),
        ("text", "algan.mobs.text", "Tex", "__init__"),
        ("move", "algan.mobs.manim_compat", "ManimCompatMob", "move"),
    ):
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        targets.append(
            (
                key,
                cls,
                name,
                getattr(cls, name),
                baseline_method(args.baseline_ref, module, cls, name),
            )
        )
    preset = getattr(algan, args.quality)
    record = importlib.import_module(f"benchmarks.performance.{args.scene}_scene").scene
    out = Path("algan_outputs") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    measurements = []
    data = {
        "arguments": vars(args),
        "device": str(device),
        "measurements": measurements,
    }
    original_prepare = RenderLoopMixin._get_batch_of_primitives
    preparation_times = []

    def prepare(self, *a, **kw):
        profiler = cProfile.Profile() if args.profile else None
        start = time.perf_counter()
        try:
            if profiler:
                profiler.enable()
            return original_prepare(self, *a, **kw)
        finally:
            if profiler:
                profiler.disable()
            preparation_times.append(time.perf_counter() - start)
            if profiler:
                profiler.dump_stats(str(out / f"{arm}_preparation.prof"))
                with (out / f"{arm}_preparation_profile.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                        "cumulative"
                    ).print_stats(70)

    RenderLoopMixin._get_batch_of_primitives = prepare
    sequence = names * args.warmups + (names + names[::-1]) * args.pairs
    try:
        for index, arm in enumerate(sequence):
            for key, cls, name, current, legacy in targets:
                setattr(cls, name, current if arm in ("current", key) else legacy)
            SceneManager.reset()
            scene = Scene()
            scene.set_video_settings(preset)
            gc.collect()
            preparation_times.clear()
            profiler = cProfile.Profile() if args.profile else None
            start = time.perf_counter()
            if profiler:
                profiler.enable()
            record(args.frames / preset.frames_per_second)
            if profiler:
                profiler.disable()
            authored = time.perf_counter()
            if args.mode == "render":
                result = scene.save_video(
                    str(out / f"{arm}.mp4"),
                    preset,
                    overwrite=True,
                    animate_fade_out=False,
                    ffmpeg_params=["-crf", "0", "-preset", "ultrafast"],
                )
                torch.cuda.synchronize()
            finished = time.perf_counter()
            item = {
                "arm": arm,
                "warm": index >= len(names) * args.warmups,
                "author_seconds": authored - start,
                "render_seconds": finished - authored,
                "total_seconds": finished - start,
                "preparation_seconds": sum(preparation_times),
                "preparation_calls": len(preparation_times),
            }
            if args.mode == "render":
                item["output"] = str(result.output_path)
            if profiler:
                profiler.dump_stats(str(out / f"{arm}.prof"))
                with (out / f"{arm}_profile.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                        "cumulative"
                    ).print_stats(70)
            measurements.append(item)
            (out / "summary.json").write_text(json.dumps(data, indent=2))
            print("PREPARATION_AB " + json.dumps(item), flush=True)
        data["medians"] = {
            field: {
                arm: statistics.median(
                    i[field] for i in measurements if i["warm"] and i["arm"] == arm
                )
                for arm in names
            }
            for field in (
                "author_seconds",
                "render_seconds",
                "total_seconds",
                "preparation_seconds",
            )
        }
        if args.mode == "render":
            data["parity"] = {
                arm: compare_videos(out / "legacy.mp4", out / f"{arm}.mp4")
                for arm in names
                if arm != "legacy"
            }
        (out / "summary.json").write_text(json.dumps(data, indent=2))
        print("PREPARATION_AB " + json.dumps(data), flush=True)
        if any(
            p["max_channel_difference"] > 2 for p in data.get("parity", {}).values()
        ):
            raise AssertionError(data["parity"])
    finally:
        RenderLoopMixin._get_batch_of_primitives = original_prepare
        for _, cls, name, current, _ in targets:
            setattr(cls, name, current)


if __name__ == "__main__":
    main()
