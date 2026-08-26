"""Quantify merge-time expansion: static mobs beside one moving mob.

Intercepts the render's own _merge_scene call (aborting before any kernel
launches), then prints each merged tensor's shape, dtype, MB and whether its
time axis is 1 (deduped) or T (expanded). Arm A: everything static. Arm B:
one cube moves. CPU only.
"""

import os

os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ["ALGAN_RENDER_DEVICE"] = "cpu"
import sys

import torch

from algan import *
from algan.rendering.raytracing import tracer

MOVE = sys.argv[1] == "move" if len(sys.argv) > 1 else False

with Off():
    img = ImageMob("benchmarks/performance/world_map.png").move(LEFT * 2).spawn()
    sph = Sphere().move(RIGHT * 2).spawn()
    txt = Text("hello world").move(UP * 2).spawn()
    cube = Cube().move(DOWN * 2).spawn()
with Sync(run_time=2):
    if MOVE:
        cube.move(UP)
    else:
        txt.wait()

from algan.rendering.raytracing import scene_builder

captured = {}
real_merge = scene_builder._merge_scene


def fake_merge(prims):
    m = real_merge(prims)
    if "merged" not in captured:
        captured["merged"] = m
    return m


scene_builder._merge_scene = fake_merge
tracer._merge_scene = fake_merge
Scene.save_video("probe_time_expansion", PREVIEW, overwrite=True)

merged = captured["merged"]
rows = []
total = 0.0
for k, v in sorted(merged.items()):
    if not torch.is_tensor(v):
        continue
    mb = v.numel() * v.element_size() / 1e6
    total += mb
    rows.append((mb, k, tuple(v.shape), str(v.dtype).replace("torch.", "")))
rows.sort(reverse=True)
print(f"ARM={'move' if MOVE else 'static'} total={total:.1f} MB")
for mb, k, shape, dt in rows:
    if mb < 0.05:
        continue
    print(f"  {mb:9.2f} MB  {k:28s} {dt:8s} {shape}")

for k in ("textures", "tri_pos"):
    v = merged.get(k)
    if v is None or not torch.is_tensor(v) or v.shape[0] <= 1:
        continue
    same = [bool((v[i] == v[0]).all()) for i in range(1, v.shape[0])]
    print(f"  {k}: frames identical to frame0: {same}")
    if not all(same):
        d = (v[1].float() - v[0].float()).abs()
        print(f"    max|d| frame1 vs 0: {d.max().item():.3e}, differing elems: {int((d > 0).sum())}/{d.numel()}")
