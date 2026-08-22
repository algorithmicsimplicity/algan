"""Render a renderer-audit scene spec with Algan.

The Algan half of the two-back-end comparison described in ``SPEC.md``. Its
Three.js counterpart is ``three_render.mjs``; both consume the same JSON so a
pixel difference between their outputs is a difference in the renderers.

Usage::

    <venv-python> benchmarks/renderer_audit/algan_render.py scenes/showcase.json --out out/

Nothing here is a test and nothing is baselined: it exists to be looked at.
Every knob that is not in the spec is left at Algan's default on purpose --
the audit is about what a user gets, not about what the engine can be talked
into.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

# Deterministic, comparable renders: no daemon reuse of a previous run's
# adaptive state (the benchmarks in this tree do the same).
os.environ.setdefault("ALGAN_USE_DAEMON", "0")


def _color(spec, default=(1.0, 1.0, 1.0)):
    from algan.constants.color import Color

    rgb = tuple(float(c) for c in (spec if spec is not None else default))
    return Color(rgb[:3])


def _vec(spec):
    """A spec position/direction as an Algan world vector.

    The spec uses Three.js's frame (+Z toward the viewer, right-handed).
    Algan's is the same frame with **Z negated**: its ``OUT`` -- the direction
    out of the screen toward the viewer -- is ``(0, 0, -1)``, and a new Scene's
    camera sits at ``z = -7``. So every position and direction crossing the
    boundary flips Z. (Nothing else changes: X is right and Y is up in both.)
    """
    import torch

    x, y, z = (float(c) for c in spec)
    return torch.tensor((x, y, -z), dtype=torch.get_default_dtype())


def _build_material(mat):
    """Translate a spec material into the Algan material class it names."""
    from algan import (
        MeshBasicMaterial,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
    )

    kind = mat.get("type", "physical")
    color = _color(mat.get("color"), (1.0, 1.0, 1.0))
    if kind == "basic":
        return MeshBasicMaterial(color=color, opacity=mat.get("opacity", 1.0))

    common = dict(
        color=color,
        roughness=mat.get("roughness", 1.0),
        metalness=mat.get("metalness", 0.0),
        emissive=_color(mat.get("emissive"), (0.0, 0.0, 0.0)),
        emissive_intensity=mat.get("emissive_intensity", 1.0),
        opacity=mat.get("opacity", 1.0),
    )
    if kind == "standard":
        return MeshStandardMaterial(**common)
    return MeshPhysicalMaterial(
        ior=mat.get("ior", 1.5),
        transmission=mat.get("transmission", 0.0),
        clearcoat=mat.get("clearcoat", 0.0),
        clearcoat_roughness=mat.get("clearcoat_roughness", 0.0),
        sheen=mat.get("sheen", 0.0),
        sheen_roughness=mat.get("sheen_roughness", 1.0),
        sheen_color=_color(mat.get("sheen_color"), (0.0, 0.0, 0.0)),
        specular_intensity=mat.get("specular_intensity", 1.0),
        specular_color=_color(mat.get("specular_color"), (1.0, 1.0, 1.0)),
        **common,
    )


def _build_object(obj):
    from algan import Prism, Sphere
    from algan.constants.spatial import UP

    geom = obj["geometry"]
    kind = geom["type"]
    if kind == "sphere":
        mob = Sphere(radius=float(geom.get("radius", 1.0)))
    elif kind == "box":
        mob = Prism(dimensions=tuple(float(v) for v in geom["size"]))
    else:
        raise ValueError(f"unsupported geometry type {kind!r}")

    mob.set_material(_build_material(obj.get("material", {})))
    # Negated with Z (see _vec): conjugating a Y-rotation by the Z flip turns it
    # into a rotation by the opposite angle.
    rot = -float(obj.get("rotation_y", 0.0))
    if rot:
        mob.rotate(rot, UP)
    mob.move_to(_vec(obj.get("position", (0, 0, 0))))
    return mob


def _build_light(light):
    from algan import AmbientLight, DirectionalLight, PointLight
    from algan.constants.spatial import ORIGIN

    kind = light["type"]
    color = _color(light.get("color"), (1.0, 1.0, 1.0))
    intensity = float(light.get("intensity", 1.0))
    if kind == "ambient":
        return AmbientLight(color=color, intensity=intensity)
    if kind == "directional":
        # The spec's `direction` points from the light toward the scene, so the
        # light sits at -direction (Three.js's DirectionalLight.position, with
        # the same target).
        d = _vec(light["direction"])
        d = d / d.norm()
        return DirectionalLight(
            location=-d * 50.0, target=ORIGIN, color=color, intensity=intensity
        )
    if kind == "point":
        return PointLight(
            location=_vec(light["position"]),
            color=color,
            intensity=intensity,
            decay=float(light.get("decay", 0.0)),
            distance=float(light.get("distance", 0.0)),
        )
    raise ValueError(f"unsupported light type {kind!r}")


def render(
    spec_path: Path,
    out_dir: Path,
    suffix: str = "algan",
    aa: int = 3,
    *,
    tonemap: bool = True,
    glossy: bool = False,
    bounces: int | None = None,
    shadows: bool = True,
):
    spec = json.loads(Path(spec_path).read_text())

    from algan import SETTINGS, Camera, Off, Scene, SceneManager, VideoSettings
    from algan.constants.spatial import CAMERA_ORIGIN

    # Algan's default scene initializer spawns a white PointLight beside the
    # camera. The spec says what the lights are, so that freebie has to go or
    # the comparison is against a scene Three.js was never given.
    def _bare_scene(scene):
        scene.camera = Camera(scene=scene, location=CAMERA_ORIGIN).spawn(animate=False)
        scene.light_sources = []

    SceneManager.set_scene_class(Scene, _bare_scene)
    SceneManager.instance().reset()

    r = spec.get("render", {})
    width = int(r.get("width", 640))
    height = int(r.get("height", 480))
    video = VideoSettings((width, height), 30, anti_alias_level=aa)
    SETTINGS.video.set(video)
    SETTINGS.raytracing.set(shadows=shadows, tonemapping=tonemap)
    if bounces is not None:
        SETTINGS.raytracing.set(max_bounces=bounces)
    if glossy:
        SETTINGS.raytracing.set(glossy_reflection=True)

    Scene.set_background_color(_color(r.get("background"), (0.0, 0.0, 0.0)))

    cam_spec = spec["camera"]
    with Off():
        for light in spec.get("lights", []):
            _build_light(light).spawn(animate=False)
        for obj in spec.get("objects", []):
            _build_object(obj).spawn(animate=False)

        camera = Scene.get_camera()
        camera.set_fov(float(cam_spec.get("fov", 40.0)))
        camera.move_to(_vec(cam_spec["position"]))
        camera.look_at(_vec(cam_spec.get("target", (0, 0, 0))))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.get('name', spec_path.stem)}.{suffix}.png"
    t0 = time.time()
    Scene.save_frame(str(out_path), video)
    seconds = time.time() - t0
    print(
        json.dumps(
            {
                "scene": spec.get("name", spec_path.stem),
                "backend": "algan",
                "output": str(out_path),
                "resolution": [width, height],
                "anti_alias_level": aa,
                "tonemap": tonemap,
                "glossy_reflection": glossy,
                "shadows": shadows,
                "seconds": round(seconds, 2),
            }
        )
    )
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path, default=_HERE / "out")
    ap.add_argument("--suffix", default="algan")
    ap.add_argument("--aa", type=int, default=3, help="anti_alias_level")
    ap.add_argument(
        "--no-tonemap",
        dest="tonemap",
        action="store_false",
        help="turn Algan's default PBR-Neutral tonemapper off, so the only "
        "transfer curve in play is the one the comparison is measuring",
    )
    ap.add_argument(
        "--glossy",
        action="store_true",
        help="turn on roughness-blurred reflections (off by default)",
    )
    ap.add_argument("--bounces", type=int, default=None)
    ap.add_argument("--no-shadows", dest="shadows", action="store_false")
    args = ap.parse_args(argv)
    render(
        args.scene,
        args.out,
        args.suffix,
        args.aa,
        tonemap=args.tonemap,
        glossy=args.glossy,
        bounces=args.bounces,
        shadows=args.shadows,
    )


if __name__ == "__main__":
    main()
