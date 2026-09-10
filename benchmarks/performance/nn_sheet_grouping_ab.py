"""Alternating warm sheet-grouping A/B on the canonical nn_scene_UHD workload.

Loads the scene factory, quality and encoder settings from nn_scene_UHD.py.
A uses the reference grouping; B uses local Metal grouping. Both arms are
warmed before whole-save_video timings. Authoring is outside the timer;
preparation, rendering, output copies and encoder drain are included. No
stage profiler or per-stage synchronization is installed. Separate renders
compare all raw frames before encoding, allowing at most two channel values.

Use --component class, unique or metadata to isolate one operation, both
for grouping alone, or all (default) for the combined opt-in gate. --skip-parity
is for timing-only confirmation after identical code has passed raw parity.

The default ABBABAAB order balances both adjacent pairs and positions in the
sequence. Report means, medians and adjacent-pair ratios together, rather than
choosing the statistic that favours a candidate. Coarse operation timers add no
synchronization; their times can include earlier queued work, not just kernels.
A reference commit supplies two reference functions, not a second full checkout.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import runpy
import statistics
import subprocess
import time
from pathlib import Path

os.environ["ALGAN_USE_DAEMON"] = "0"


def emit(event, **fields):
    row = {"event": event, **fields}
    print("NN_SHEET_GROUPING_AB " + json.dumps(row, sort_keys=True), flush=True)
    return row


def _validate_sequence(sequence):
    """Require one control and one candidate in every adjacent pair."""
    if (
        not sequence
        or len(sequence) % 2
        or any(
            set(sequence[i : i + 2]) != {"A", "B"}
            for i in range(0, len(sequence), 2)
        )
    ):
        raise ValueError("sequence must consist of AB or BA pairs")


def _timing_summary(rows):
    measured = [row for row in rows if row["phase"] == "measured"]
    summary = {}
    for arm in "AB":
        values = [row["seconds"] for row in measured if row["grouping"]["arm"] == arm]
        summary[arm] = {"seconds": values}
        if values:
            summary[arm].update(
                median=statistics.median(values),
                mean=statistics.mean(values),
                minimum=min(values),
                maximum=max(values),
            )
    if measured:
        _validate_sequence("".join(row["grouping"]["arm"] for row in measured))
        pairs = []
        for i in range(0, len(measured), 2):
            values = {
                row["grouping"]["arm"]: row["seconds"] for row in measured[i : i + 2]
            }
            pairs.append(
                {
                    "A": values["A"],
                    "B": values["B"],
                    "ratio_b_over_a": values["B"] / values["A"],
                }
            )
        summary["adjacent_pairs"] = pairs
        for statistic in ("mean", "median"):
            summary[statistic + "_reduction_percent"] = 100 * (
                1 - summary["B"][statistic] / summary["A"][statistic]
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=("all", "both", "class", "unique", "metadata"),
        default="all",
    )
    parser.add_argument("--sequence", default="ABBABAAB")
    parser.add_argument(
        "--reference-commit",
        help="Load reference class and unique functions from this immutable commit",
    )
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()
    if args.parity_only and args.skip_parity:
        parser.error("--parity-only and --skip-parity are mutually exclusive")
    try:
        _validate_sequence(args.sequence)
    except ValueError as error:
        parser.error(str(error))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "renders.json").exists() or list(
        (args.out / "raw_control").glob("*.npy")
    ):
        parser.error("output contains an earlier experiment; use a new --out directory")

    import numpy as np
    import psutil
    import torch

    from algan import SETTINGS, Scene, render_loop
    from algan.rendering import taichi_runtime
    from algan.rendering.raytracing import sheet_grouping, sheets
    from algan.scene_manager import SceneManager
    from algan.taichi_compat import ti
    from algan.utils import profiling_utils

    if SETTINGS.computing.render_device.type != "mps":
        raise RuntimeError("This measurement requires the actual Metal GPU")
    original_settings = SETTINGS.snapshot()
    original_profile = profiling_utils.profile_scene
    captured = {}

    def capture(factory, quality, tag, **kwargs):
        captured.update(factory=factory, quality=quality, kwargs=kwargs)

    profiling_utils.profile_scene = capture
    try:
        runpy.run_path(
            str(Path(__file__).with_name("nn_scene_UHD.py")), run_name="__main__"
        )
    finally:
        profiling_utils.profile_scene = original_profile
    save_kwargs = captured["kwargs"].get("save_video_kwargs", {})
    original_groups = sheets._sheet_class_groups
    original_unique = sheets._unique_sorted_ids
    original_gate = sheets.sheet_mps_grouping
    original_metadata = sheets._sheet_fragment_metadata
    reference_groups = None
    reference_unique = None
    if args.reference_commit:
        # Isolated namespace: do not replace shared functions while prefetch runs.
        from algan.rendering import mps_compat

        source = subprocess.check_output(
            ["git", "show", f"{args.reference_commit}:algan/rendering/mps_compat.py"],
            text=True,
        )
        definition = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "band_class_groups"
        )
        namespace = dict(vars(mps_compat))
        exec(
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                "<reference-band-class-groups>",
                "exec",
            ),
            namespace,
        )
        reference_groups = namespace["band_class_groups"]
        source = subprocess.check_output(
            [
                "git",
                "show",
                f"{args.reference_commit}:algan/rendering/raytracing/sheets.py",
            ],
            text=True,
        )
        definition = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "_unique_sorted_ids"
        )
        namespace = dict(vars(sheets))
        exec(
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                "<reference-unique-sorted-ids>",
                "exec",
            ),
            namespace,
        )
        reference_unique = namespace["_unique_sorted_ids"]
    original_env = os.environ.get("ALGAN_TORCH_COMPILE")
    original_writer = render_loop.write_frames_from_queue
    rows = []
    grouping = {}
    parity_rows = []
    writer_errors = []
    raw = args.out / "raw_control"
    raw.mkdir(exist_ok=True)
    parity_mode = None
    helper_names = {
        "class": "class_groups",
        "unique": "unique_sorted_ids",
        "metadata": "prepare_fragments",
    }
    original_helpers = {
        name: getattr(sheet_grouping, helper) for name, helper in helper_names.items()
    }

    def counted_helper(name, function):
        @functools.wraps(function)
        def counted(*values, **kwargs):
            result = function(*values, **kwargs)
            if values[0].numel():
                grouping[name + "_local_calls"] += 1
            return result

        return counted

    def groups(bands, classes, starts, nb):
        operation_start = time.perf_counter()
        enabled = grouping["arm"] == "B" and args.component in ("all", "both", "class")
        previous = sheets.sheet_mps_grouping
        sheets.sheet_mps_grouping = enabled
        try:
            result = (
                reference_groups(bands, classes, sheets._SHADE_CLASS_BASE)
                if grouping["arm"] == "A" and reference_groups is not None
                else original_groups(bands, classes, starts, nb)
            )
        finally:
            sheets.sheet_mps_grouping = previous
        grouping["class_host_seconds"] = (
            grouping.get("class_host_seconds", 0.0)
            + time.perf_counter()
            - operation_start
        )
        grouping["class_calls"] += 1
        grouping["class_rows"] += bands.numel()
        return result

    def unique(keys):
        operation_start = time.perf_counter()
        enabled = grouping["arm"] == "B" and args.component in ("all", "both", "unique")
        previous = sheets.sheet_mps_grouping
        sheets.sheet_mps_grouping = enabled
        try:
            result = (
                reference_unique(keys)
                if grouping["arm"] == "A" and reference_unique is not None
                else original_unique(keys)
            )
        finally:
            sheets.sheet_mps_grouping = previous
        grouping["unique_host_seconds"] = (
            grouping.get("unique_host_seconds", 0.0)
            + time.perf_counter()
            - operation_start
        )
        grouping["unique_calls"] += 1
        grouping["unique_rows"] += keys.numel()
        return result

    def metadata_hook(keys, refs, masks, objects, ppf, time_start):
        operation_start = time.perf_counter()
        previous = sheets.sheet_mps_grouping
        sheets.sheet_mps_grouping = grouping["arm"] == "B" and args.component in (
            "all",
            "metadata",
        )
        try:
            result = original_metadata(keys, refs, masks, objects, ppf, time_start)
        finally:
            sheets.sheet_mps_grouping = previous
        grouping["metadata_host_seconds"] = (
            grouping.get("metadata_host_seconds", 0.0)
            + time.perf_counter()
            - operation_start
        )
        grouping["metadata_calls"] += 1
        return result

    def parity_writer(queue, writer):
        index = 0
        while True:
            frame = queue.get()
            if frame is None:
                break
            image = frame.numpy()
            path = raw / f"{index:04d}.npy"
            try:
                if parity_mode == "A":
                    np.save(path, image, allow_pickle=False)
                else:
                    reference = np.load(path, allow_pickle=False)
                    if reference.shape != image.shape:
                        raise AssertionError("Raw frame dimensions differ")
                    delta = np.abs(reference.astype(np.int16) - image.astype(np.int16))
                    row = {
                        "frame": index,
                        "shape": list(image.shape),
                        "max_channel_delta": int(delta.max()),
                        "mean_channel_delta": float(delta.mean()),
                        "pixels_over_2": int(np.any(delta > 2, axis=-1).sum()),
                    }
                    parity_rows.append(row)
                    if index == 0 or (row["pixels_over_2"] and not writer_errors):
                        from PIL import Image

                        Image.fromarray(reference).save(args.out / "control.png")
                        Image.fromarray(image).save(args.out / "candidate.png")
                        Image.fromarray(
                            np.minimum(delta * 16, 255).astype(np.uint8)
                        ).save(args.out / "difference_x16.png")
                    if row["pixels_over_2"]:
                        writer_errors.append(
                            f"Frame {index}: delta {row['max_channel_delta']}"
                        )
                    path.unlink()
            except Exception as error:
                # Never kill the consumer and deadlock the bounded writer queue.
                writer_errors.append(repr(error))
            writer.write_frame(image)
            index += 1
        grouping["raw_frames"] = index
        if index != grouping["expected_frames"]:
            writer_errors.append(
                f"Expected {grouping['expected_frames']} raw frames, received {index}"
            )

    def render(arm, phase, index):
        nonlocal grouping
        SETTINGS.computing.torch_compile = False
        os.environ["ALGAN_TORCH_COMPILE"] = "0"
        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        scene = SceneManager.reset()
        scene.set_video_settings(captured["quality"])
        captured["factory"]()
        grouping = {
            "arm": arm,
            "expected_frames": int(
                (scene.max_time - scene.min_time) * scene.frames_per_second
            ),
            "metadata_calls": 0,
            "metadata_local_calls": 0,
            "class_calls": 0,
            "class_rows": 0,
            "class_local_calls": 0,
            "unique_calls": 0,
            "unique_rows": 0,
            "unique_local_calls": 0,
        }
        from algan.rendering.mps_zero_copy import LEFT_ON_THE_BUS, STATS

        taichi_runtime._sync_devices()
        zero_copy_before = dict(STATS)
        started = time.perf_counter()
        Scene.save_video(
            str(args.out / f"{phase}_{index}_{arm}.mp4"),
            video_settings=captured["quality"],
            reset=True,
            **save_kwargs,
        )
        taichi_runtime._sync_devices()
        seconds = time.perf_counter() - started
        architecture = ti.lang.impl.current_cfg().arch
        if architecture != ti.metal:
            raise AssertionError(f"MPS inputs did not execute on Metal: {architecture}")
        if arm == "B":
            expected_components = {
                "all": ("class", "unique", "metadata"),
                "both": ("class", "unique"),
            }.get(args.component, (args.component,))
            for name in expected_components:
                if not grouping[name + "_local_calls"]:
                    raise AssertionError(f"The {name} optimization never engaged")
        memory = {
            "rss_mib": psutil.Process().memory_info().rss / 2**20,
            "available_mib": psutil.virtual_memory().available / 2**20,
        }
        if SETTINGS.computing.render_device.type == "mps":
            memory.update(
                mps_driver_mib=torch.mps.driver_allocated_memory() / 2**20,
                mps_live_mib=torch.mps.current_allocated_memory() / 2**20,
            )
        row = emit(
            "render",
            zero_copy=dict(STATS),
            zero_copy_delta={
                key: value - zero_copy_before.get(key, 0)
                for key, value in STATS.items()
            },
            render_settings={
                name: getattr(SETTINGS.raytracing, name)
                for name in (
                    "samples_per_pixel", "analytic_aa", "max_bounces", "shadows"
                )
            },
            staging_reasons=sorted(LEFT_ON_THE_BUS),
            phase=phase,
            index=index,
            seconds=seconds,
            grouping=dict(grouping),
            memory=memory,
        )
        rows.append(row)
        (args.out / "renders.json").write_text(json.dumps(rows, indent=2))

    versions = {}
    for name in ("torch", "algan-quadrants", "quadrants"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    metadata = emit(
        "metadata",
        component=args.component,
        reference_commit=args.reference_commit,
        device=str(SETTINGS.computing.render_device),
        platform=platform.platform(),
        python=platform.python_version(),
        versions=versions,
        cpu_threads=torch.get_num_threads(),
        ram_mib=psutil.virtual_memory().total / 2**20,
        quality=str(captured["quality"]),
        encoder=save_kwargs,
        sequence=args.sequence,
        arena_override=SETTINGS.computing.available_memory_override is not None,
        available_memory_override=SETTINGS.computing.available_memory_override,
        source_sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/performance/nn_scene_UHD.py",
                "algan/rendering/raytracing/sheets.py",
                "algan/rendering/raytracing/sheet_grouping.py",
                "algan/rendering/raytracing/sheet_grouping_taichi.py",
                "algan/rendering/mps_compat.py",
            )
        },
    )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    sheets._sheet_class_groups = groups
    sheets._unique_sorted_ids = unique
    sheets._sheet_fragment_metadata = metadata_hook
    for name, helper in helper_names.items():
        setattr(sheet_grouping, helper, counted_helper(name, original_helpers[name]))
    try:
        if not args.parity_only:
            for index, arm in enumerate("AB"):
                render(arm, "warmup", index)
            for index, arm in enumerate(args.sequence):
                render(arm, "measured", index)
        summary = {
            "component": args.component,
            "parity_checked": False,
            **_timing_summary(rows),
        }
        if not args.parity_only:
            summary["arch"] = str(ti.lang.impl.current_cfg().arch)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        emit("timing_summary", **summary)
        if not args.skip_parity:
            render_loop.write_frames_from_queue = parity_writer
            for index, arm in enumerate("AB"):
                parity_mode = arm
                render(arm, "parity_unmeasured", index)
            if not parity_rows or list(raw.glob("*.npy")):
                writer_errors.append("Missing or mismatched raw frame counts")
            parity = {"frames": parity_rows, "errors": writer_errors}
            (args.out / "parity.json").write_text(json.dumps(parity, indent=2))
            emit(
                "parity",
                frames=len(parity_rows),
                errors=writer_errors,
                max_channel_delta=max(
                    (r["max_channel_delta"] for r in parity_rows), default=None
                ),
            )
            if writer_errors:
                raise AssertionError("Raw-frame parity failed; see parity.json")
            summary["parity_checked"] = True
            summary["raw_frames"] = len(parity_rows)
            summary["max_channel_delta"] = max(
                r["max_channel_delta"] for r in parity_rows
            )
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        sheets._sheet_class_groups = original_groups
        sheets._unique_sorted_ids = original_unique
        sheets._sheet_fragment_metadata = original_metadata
        sheets.sheet_mps_grouping = original_gate
        render_loop.write_frames_from_queue = original_writer
        for name, helper in helper_names.items():
            setattr(sheet_grouping, helper, original_helpers[name])
        SETTINGS.restore(original_settings)
        if original_env is None:
            os.environ.pop("ALGAN_TORCH_COMPILE", None)
        else:
            os.environ["ALGAN_TORCH_COMPILE"] = original_env
    emit("renders_complete")


if __name__ == "__main__":
    main()
