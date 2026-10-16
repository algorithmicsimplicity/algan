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
                    "geometry": {"type": "sphere"},
                    "material": {
                        "type": "standard",
                        "color": [0.1, 0.2, 0.3],
                        "roughness": 0.3,
                    },
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


_AUDIT = Path(__file__).resolve().parents[2] / "benchmarks/renderer_audit"
_SPHERE = {"geometry": {"type": "sphere"}}
_SPOT = {"type": "spot", "position": [0, 4, 0], "target": [0, 0, 0], "angle": 20}
_RECT = {"type": "rect_area", "position": [0, 4, 0], "target": [0, 0, 0]}


def _javascript_outcomes(specs):
    """``normalizeSpec`` on each spec in one Node process: result or error."""
    if not shutil.which("node"):
        pytest.skip("Node is needed to execute the JavaScript contract")
    script = (
        "import fs from 'node:fs'; import {normalizeSpec} from "
        + json.dumps((_AUDIT / "scene_contract.mjs").as_uri())
        + "; const specs = JSON.parse(fs.readFileSync(0, 'utf8'));"
        " console.log(JSON.stringify(specs.map((spec) => {"
        " try { return {ok: normalizeSpec(spec)}; }"
        " catch (error) { return {error: error.message}; } })));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(specs),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _python_outcome(spec):
    try:
        return {"ok": json.loads(json.dumps(_normalize_spec(spec)))}
    except ValueError as error:
        return {"error": str(error)}


def test_every_audit_scene_normalizes_identically_in_both_bridges():
    paths = sorted((_AUDIT / "scenes").glob("*.json"))
    assert paths
    specs = [json.loads(path.read_text()) for path in paths]

    javascript = _javascript_outcomes(specs)

    for path, spec, outcome in zip(paths, specs, javascript):
        assert "ok" in outcome, (path.name, outcome)
        assert _python_outcome(spec) == outcome, path.name


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"objectz": []}, "Unsupported scene fields: objectz"),
        ({"render": {"widht": 9}}, "Unsupported render fields: widht"),
        (
            {"objects": [{**_SPHERE, "rotation_x": 5}]},
            "Unsupported objects[0] fields: rotation_x",
        ),
        ({"objects": [{"material": {}}]}, "Missing objects[0] fields: geometry"),
        (
            {"objects": [{"geometry": {"type": "torus"}}]},
            'Unsupported objects[0].geometry type: "torus"',
        ),
        (
            {"objects": [{"geometry": {}}]},
            "Unsupported objects[0].geometry type: null",
        ),
        (
            {"objects": [{"geometry": {"type": "sphere", "radiuss": 2}}]},
            "Unsupported objects[0].geometry fields: radiuss",
        ),
        (
            {"objects": [{"geometry": {"type": "box"}}]},
            "Missing objects[0].geometry fields: size",
        ),
        (
            {"objects": [{**_SPHERE, "material": {"roughnes": 0.2}}]},
            "Unsupported objects[0].material fields: roughnes",
        ),
        (
            {"objects": [{**_SPHERE, "material": {"type": "glass"}}]},
            'Unsupported objects[0].material type: "glass"',
        ),
        ({"lights": [{"type": "laser"}]}, 'Unsupported lights[0] type: "laser"'),
        ({"lights": [{"intensity": 1}]}, "Unsupported lights[0] type: null"),
        (
            {"lights": [{"type": "ambient", "intensty": 1}]},
            "Unsupported lights[0] fields: intensty",
        ),
        ({"lights": [{"type": "point"}]}, "Missing lights[0] fields: position"),
        (
            {"lights": [{k: v for k, v in _SPOT.items() if k != "angle"}]},
            "Missing lights[0] fields: angle",
        ),
        ({"lights": [_RECT]}, "Missing lights[0] fields: width, height"),
        (
            {"lights": [{"type": "directional"}]},
            "lights[0]: a directional light takes exactly one form",
        ),
        (
            {
                "lights": [
                    {
                        "type": "directional",
                        "direction": [0, -1, 0],
                        "position": [0, 5, 0],
                    }
                ]
            },
            "lights[0]: a directional light takes exactly one form",
        ),
    ],
)
def test_both_bridges_reject_the_same_malformed_scene(spec, message):
    python = _python_outcome(spec)
    (javascript,) = _javascript_outcomes([spec])

    assert python == javascript
    assert "error" in python
    assert python["error"].startswith(message)


def test_omitted_light_and_object_fields_take_the_shared_defaults():
    spec = {
        "_why": "annotations pass through at every level",
        "lights": [
            {"type": "point", "position": [1, 2, 3], "_note": "no decay given"},
            {"type": "hemisphere"},
            {"type": "directional", "position": [0, 5, 0]},
            {**_RECT, "width": 2, "height": 1, "samples": 8},
            _SPOT,
        ],
        "objects": [{**_SPHERE, "material": {"type": "toon", "_note": "x"}}],
    }

    python = _python_outcome(spec)
    (javascript,) = _javascript_outcomes([spec])

    assert python == javascript
    point, hemisphere, directional, rect, spot = python["ok"]["lights"]
    # three's own PointLight decay default is 2; the shared default is 0.
    assert (point["decay"], point["distance"]) == (0, 0)
    assert (point["color"], point["intensity"]) == ([1, 1, 1], 1)
    assert hemisphere["ground_color"] == [0, 0, 0]
    assert directional["target"] == [0, 0, 0]
    assert rect["samples"] == 8
    assert (spot["penumbra"], spot["decay"], spot["distance"]) == (0, 0, 0)
    (sphere,) = python["ok"]["objects"]
    assert sphere["position"] == [0, 0, 0]
    assert sphere["rotation_y"] == 0
    assert sphere["geometry"] == {"type": "sphere", "radius": 1, "segments": 64}
    assert sphere["material"]["bands"] == 3


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
