"""Reproduce the bright speckling on the glTF + PBR + normal-map sphere.

Renders the Act-3 model of ``tests/full_renders/scenes/text_and_media.py`` on
its own, at the same PREVIEW settings and the same lighting, and writes raw
PNGs (no video compression) plus a speckle map flagging any pixel more than 25
brighter than the median of its 3x3 neighbourhood.

Env knobs for the diagnosis:

``SPECKLE_OUT``      output directory (default ``benchmarks/_gltf_speckle_out``)
``SPECKLE_LIGHTS``   ``both`` (default) / ``ambient`` / ``directional``
``SPECKLE_NMAP``     ``1`` (default) / ``0`` to strip the model's normal maps
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "tests" / "full_renders"
os.chdir(ASSETS)

from algan import *  # noqa: E402
from algan import PREVIEW, SETTINGS, Scene  # noqa: E402

OUTDIR = Path(os.environ.get("SPECKLE_OUT", REPO / "benchmarks" / "_gltf_speckle_out"))
OUTDIR.mkdir(parents=True, exist_ok=True)
LIGHTS = os.environ.get("SPECKLE_LIGHTS", "both")
KEEP_NMAP = os.environ.get("SPECKLE_NMAP", "1") == "1"

SETTINGS.paths.set(output_root=str(OUTDIR), output_directory=".")
if "SPECKLE_BOUNCES" in os.environ:
    SETTINGS.raytracing.set(max_bounces=int(os.environ["SPECKLE_BOUNCES"]))

Scene.set_background(DARKER_GRAY)

with Off():
    if LIGHTS in ("both", "ambient"):
        AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    if LIGHTS in ("both", "directional"):
        DirectionalLight(
            location=RIGHT * 4 + UP * 5 + OUT * 4,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
        ).spawn(animate=False)
    model = Model3D(
        "assets/textured_icosphere.glb",
        fit_to_size=float(os.environ.get("SPECKLE_SIZE", "2.6")),
        pbr_materials=os.environ.get("SPECKLE_PBR", "1") == "1",
    ).move(UP * 0.2)
    if not KEEP_NMAP:
        stripped = 0
        stack = [model]
        while stack:
            mob = stack.pop()
            stack.extend(getattr(mob, "children", ()) or ())
            if getattr(mob, "normal_texture_map", None) is not None:
                mob.normal_texture_map = None
                stripped += 1
        print(f"stripped {stripped} normal maps")

    tex_mode = os.environ.get("SPECKLE_TEX", "orig")
    if tex_mode != "orig":
        import torch

        stack = [model]
        while stack:
            mob = stack.pop()
            stack.extend(getattr(mob, "children", ()) or ())
            tmap = getattr(mob, "texture_map", None)
            if tmap is None:
                continue
            w, h = tmap.shape[0], tmap.shape[1]
            new = tmap.clone()
            if tex_mode == "flat":
                new[..., 0] = 0.85
                new[..., 1] = 0.55
                new[..., 2] = 0.25
            elif tex_mode == "grad":
                # A smooth ramp: any UV error shows as a smooth shift, a wrong
                # primitive/side shows as a jump.
                uu = torch.linspace(0.0, 1.0, w, device=tmap.device).view(-1, 1)
                vv = torch.linspace(0.0, 1.0, h, device=tmap.device).view(1, -1)
                new[..., 0] = uu.expand(w, h)
                new[..., 1] = vv.expand(w, h)
                new[..., 2] = 0.2
            elif tex_mode == "big":
                # Same checker, but 2x2 cells: a one-texel UV error can no
                # longer flip the colour, a half-sphere error still can.
                uu = torch.arange(w, device=tmap.device).view(-1, 1) * 2 // w
                vv = torch.arange(h, device=tmap.device).view(1, -1) * 2 // h
                odd = ((uu + vv) % 2).float().expand(w, h)
                new[..., 0] = 1.0 * odd + 0.3 * (1 - odd)
                new[..., 1] = 0.66 * odd + 0.87 * (1 - odd)
                new[..., 2] = 0.25 * odd + 0.84 * (1 - odd)
            mob.texture_map = new
            print(f"replaced texture ({tex_mode}) on {type(mob).__name__} {w}x{h}")

with Seq():
    model.spawn(animate=False)
    with Sync(runtime=2.4):
        model.rotate(300, UP)

times = [float(t) for t in sys.argv[1:]] or [0.0, 0.6, 1.2, 1.8, 2.3]
QUALITY = {"preview": PREVIEW, "hd": HD, "uhd": UHD}[
    os.environ.get("SPECKLE_QUALITY", "preview")
]
Scene.save_frame("gltf", QUALITY, at=times)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

for t in times:
    path = OUTDIR / f"gltf_{t}.png"
    if not path.exists() and len(times) == 1:
        path = OUTDIR / "gltf.png"
    img = cv2.imread(str(path))
    if img is None:
        print("missing", path)
        continue
    med = cv2.medianBlur(img, 3)
    d = (img.astype(np.int32) - med.astype(np.int32)).max(2)
    ys, xs = np.where(d > 25)
    print(f"t={t}: {len(ys)} speckle pixels, max excess {d.max()}")
    for y, x in list(zip(ys, xs))[:25]:
        print("   ", x, y, img[y, x], med[y, x])
    vis = np.clip(d * 6, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUTDIR / f"speckle_{t}.png"), vis)
