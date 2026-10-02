"""Run issue #132's original still-render call on MPS, refusing CPU fallback.

Use run_on_mac.yaml with arms=mac-mps, latex=true, taichi_wheel_run_id=none,
and ALGAN_TAICHI_FAST_LAUNCH=0 / ALGAN_USE_DAEMON=0. Before this script, install
the same graph-cache fix as test.yaml:
    uv pip install --python .venv/bin/python torch==2.13.0 torchaudio==2.11.0
    .venv/bin/python benchmarks/_issue132_mps_repro.py

The PNG and JSON report land in algan_outputs/issue132_mps for the harness to
upload. This checks crash freedom and output validity, not pixel parity.
"""

from __future__ import annotations

import hashlib
import json
import platform
import runpy
import subprocess
from contextlib import chdir
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import manimpango
import torch
from PIL import Image

from algan import LD, Scene
from algan.rendering.taichi_runtime import ensure_taichi_for_render
from algan.settings import SETTINGS
from algan.taichi_compat import ti
from algan.utils import taichi_fast_launch


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "algan_outputs" / "issue132_mps"
    output.mkdir(parents=True, exist_ok=True)
    assert torch.backends.mps.is_available(), "This reproduction requires real MPS"
    assert SETTINGS.computing.render_device.type == "mps", "CPU fallback is forbidden"
    ensure_taichi_for_render()
    assert ti.lang.impl.current_cfg().arch == ti.metal, "Metal backend required"
    assert taichi_fast_launch.skipped_reason() == "ALGAN_TAICHI_FAST_LAUNCH=0"

    video = LD.set(resolution=(320, 180), frames_per_second=10)
    timestamp = 16.225
    budget = 1536 * 1024**2
    SETTINGS.computing.set(available_memory_override=budget, torch_compile=False)
    report = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": version("torchaudio"),
        "algan_quadrants": version("algan-quadrants"),
        "render_device": str(SETTINGS.computing.render_device),
        "compiler_arch": str(ti.lang.impl.current_cfg().arch),
        "fast_launch_skipped_reason": taichi_fast_launch.skipped_reason(),
        "scene": "tests/full_renders/scenes/text_and_media.py",
        "requested_timestamp": timestamp,
        "frame_index": round(timestamp * video.frames_per_second),
        "sampled_timestamp": round(timestamp * video.frames_per_second)
        / video.frames_per_second,
        "resolution": video.resolution,
        "frames_per_second": video.frames_per_second,
        "supersampling": video.supersampling,
        "fxaa": video.fxaa,
        "available_memory_override": budget,
        "torch_compile": False,
        "denoise": False,
        "pixel_reference_compared": False,
        "status": "starting",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("ISSUE132 SETTINGS", json.dumps(report), flush=True)

    for font in (root / "tests/assets/fonts").glob("*.ttf"):
        assert manimpango.register_font(str(font)), f"Cannot register {font}"
    with chdir(root / "tests/full_renders"), Scene() as scene:
        runpy.run_path(str(root / report["scene"]), run_name="_repro")
        SETTINGS.raytracing.set(denoise=False)
        frames = scene.save_frame(
            output / "text_media.png", video_settings=video, at=[timestamp]
        )
    torch.mps.synchronize()
    assert SETTINGS.computing.render_device.type == "mps"
    assert ti.lang.impl.current_cfg().arch == ti.metal
    assert len(frames) == 1, "Expected exactly one frame"
    frame = frames[0]
    assert frame.rendered, "No new frame was rendered"
    with Image.open(frame.output_path) as image:
        image.load()
        assert image.size == (320, 180), image.size
        assert any(lo != hi for lo, hi in image.convert("RGB").getextrema()), (
            "Rendered image is uniform"
        )
    report.update(
        status="rendered",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        render_walltime_seconds=frame.walltime_seconds,
        output_path=str(frame.output_path.relative_to(root)),
        output_sha256=hashlib.sha256(frame.output_path.read_bytes()).hexdigest(),
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("ISSUE132 RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
