"""Both comparison bridges consume one set of material/camera defaults."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from algan import Camera, MeshPhysicalMaterial, Off, Scene
from benchmarks.renderer_audit.algan_render import (
    _build_material,
    _configure_camera,
    _normalize_spec,
)


@pytest.mark.parametrize(
    "spec",
    [
        {"objects": [{"geometry": {"type": "sphere"}}]},
        {
            "camera": {
                "position": [3, 2, 8],
                "target": [1, 1, 0],
                "up": [1, 1, 0],
                "fov": 55,
                "near": 0.2,
                "far": 90,
            },
            "objects": [
                {
                    "material": {
                        "type": "standard",
                        "color": [0.1, 0.2, 0.3],
                        "roughness": 0.3,
                    }
                }
            ],
        },
    ],
)
def test_python_and_javascript_normalize_to_the_same_scene(spec):
    if not shutil.which("node"):
        pytest.skip("Node is needed to execute the JavaScript contract")
    module = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/renderer_audit/scene_contract.mjs"
    )
    script = (
        "import {normalizeSpec} from "
        + json.dumps(module.as_uri())
        + "; console.log(JSON.stringify(normalizeSpec(JSON.parse(process.argv[1]))));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, json.dumps(spec)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert _normalize_spec(spec) == json.loads(result.stdout)


def test_algan_camera_bridge_applies_up_and_clipping():
    spec = _normalize_spec(
        {"camera": {"position": [0, 0, 12], "up": [1, 0, 0], "near": 0.2, "far": 90}}
    )
    with Scene(), Off():
        camera = Camera()
        _configure_camera(camera, spec["camera"])
        torch.testing.assert_close(
            camera.get_up_direction().reshape(-1), torch.tensor([1.0, 0, 0])
        )
        assert camera.near == 0.2
        assert camera.far == 90
        assert isinstance(_build_material({}), MeshPhysicalMaterial)


def test_unsupported_camera_fields_are_reported():
    with pytest.raises(ValueError, match="orthographic"):
        _normalize_spec({"camera": {"orthographic": True}})


def test_raster_parity_benchmark_setup_uses_current_entry_points():
    script = (
        Path(__file__).resolve().parents[2] / "benchmarks/_raster_bez_pre_parity.py"
    )
    # Import setup in a child so instrumentation cannot leak into pytest's renderer.
    import os
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])",
            str(script),
        ],
        env={**os.environ, "ALGAN_USE_DAEMON": "0"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
