"""Measure Algan's authored-colour -> pixel transfer curve.

Renders a flat slab filling the frame at a series of authored greys, under
three conditions (unlit, one ambient light, one head-on directional light),
with and without the default tonemapper. The point is to establish, by
measurement rather than by reading the shader, what an authored ``0.5`` grey
becomes on screen -- which is the first thing that has to be settled before any
pixel comparison against another renderer means anything.

    <venv-python> benchmarks/renderer_audit/transfer_probe.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "out" / "transfer"

LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# A slab filling the frame, seen face-on, so the centre pixel is a clean
# readout with no perspective foreshortening and no silhouette AA.
BASE = {
    "render": {"width": 96, "height": 96, "background": [0, 0, 0]},
    "camera": {
        "position": [0, 0, 6],
        "target": [0, 0, 0],
        "up": [0, 1, 0],
        "fov": 40,
        "near": 0.1,
        "far": 200,
    },
    "objects": [
        {
            "name": "slab",
            "geometry": {"type": "box", "size": [12, 12, 0.5]},
            "position": [0, 0, 0],
            "material": {},
        }
    ],
}

CONDITIONS = {
    "unlit": ([], {"type": "basic"}),
    "ambient1": (
        [{"type": "ambient", "color": [1, 1, 1], "intensity": 1.0}],
        {"type": "standard", "roughness": 1.0, "metalness": 0.0},
    ),
    "directional1": (
        [
            {
                "type": "directional",
                "direction": [0, 0, -1],
                "color": [1, 1, 1],
                "intensity": 1.0,
            }
        ],
        {"type": "standard", "roughness": 1.0, "metalness": 0.0},
    ),
}


def _read_centre(path):
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    h, w = im.shape[:2]
    return int(im[h // 2, w // 2, 2])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for tonemap in (True, False):
        for cond, (lights, mat) in CONDITIONS.items():
            key = f"{cond}{'' if tonemap else '_notonemap'}"
            row = []
            for level in LEVELS:
                spec = json.loads(json.dumps(BASE))
                name = f"{key}_{level}"
                spec["name"] = name
                spec["lights"] = lights
                spec["objects"][0]["material"] = dict(mat, color=[level, level, level])
                spec_path = OUT / f"{name}.json"
                spec_path.write_text(json.dumps(spec))
                cmd = [
                    sys.executable,
                    str(_HERE / "algan_render.py"),
                    str(spec_path),
                    "--out",
                    str(OUT),
                    "--aa",
                    "1",
                ]
                if not tonemap:
                    cmd.append("--no-tonemap")
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                row.append(_read_centre(OUT / f"{name}.algan.png"))
            results[key] = row
            print(f"{key:26s} " + " ".join(f"{v:4d}" for v in row))
    print()
    print(
        "authored                   "
        + " ".join(f"{int(round(v * 255)):4d}" for v in LEVELS)
    )
    (OUT / "transfer.json").write_text(
        json.dumps({"levels": LEVELS, "results": results}, indent=2)
    )


if __name__ == "__main__":
    main()
