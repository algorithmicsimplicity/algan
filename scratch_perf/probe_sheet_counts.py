"""Render the nn scene once at a preset and log per-chunk fragment/sheet counts.
usage: uv run python scratch_perf/probe_sheet_counts.py PREVIEW|UHD"""
import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
import algan.rendering.raytracing.sheets as sheets_mod
import algan.rendering.raytracing.raster_pipeline as rpl
preset = {"PREVIEW": PREVIEW, "UHD": UHD, "HD": HD}[sys.argv[1]]
run_time = 5 if sys.argv[1] == "PREVIEW" else 0.5
orig = sheets_mod.compact_sheets
rows = []
def wrapped(coverage, *a, **k):
    t0 = time.perf_counter(); out = orig(coverage, *a, **k); import torch; torch.cuda.synchronize(); dt = time.perf_counter() - t0
    rows.append((int(coverage["num_fragments"]), int(coverage["num_covered"]), int(out["num_sheets"]), int(out["num_groups"]), dt))
    return out
sheets_mod.compact_sheets = wrapped
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=run_time):
    nn.move(UP)
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
    label.move(RIGHT*2)
r = Scene.save_video(f"scratch_perf/sheetcount_{sys.argv[1]}.mp4", preset, overwrite=True, ffmpeg_params=["-preset", "ultrafast"])
print("render", f"{r.duration_seconds:.1f}s", "chunks", len(rows))
for f, c, s, g, dt in rows:
    print(f"frags {f:>10d} covered {c:>9d} sheets {s:>9d} groups {g:>9d} frags/covered {f/max(1,c):5.2f} sheets/covered {s/max(1,c):5.2f} compact {dt*1000:7.1f} ms")
