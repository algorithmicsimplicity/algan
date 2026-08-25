"""Render UHD twice; log arena size, chunk plan and per-tile primary counts; diff.
usage: uv run python scratch_perf/probe_tiles.py <tag> [fixed_tile_primaries]"""
import os, sys, subprocess
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
import algan.rendering.raytracing.tracer as tr
from algan.rendering.memory_model import ChunkMemoryModel
from algan.utils import memory_utils
tag = sys.argv[1]
fixed_tile = int(sys.argv[2]) if len(sys.argv) > 2 else 0
log = []
orig_plan = ChunkMemoryModel.plan
def plan(self, signature, requested, available):
    d = orig_plan(self, signature, requested, available); log[-1]["chunks"].append(int(d)); return d
ChunkMemoryModel.plan = plan
orig_tile = tr._auto_primary_per_tile
def tile(memory, *a, **k):
    n = orig_tile(memory, *a, **k)
    if fixed_tile: n = fixed_tile
    log[-1]["tiles"].append(int(n)); log[-1]["arena"] = len(memory); return n
tr._auto_primary_per_tile = tile
def scene():
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
        label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
    with Sync(run_time=0.5):
        nn.move(UP)
        x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
        label.move(RIGHT*2)
paths = []
for i in range(2):
    log.append({"chunks": [], "tiles": [], "arena": 0})
    SceneManager.reset(); scene()
    p = f"scratch_perf/tiles_{tag}_{i}.mp4"
    r = Scene.save_video(p, UHD, overwrite=True, reset=True, ffmpeg_params=["-preset", "ultrafast"])
    e = log[-1]
    print(f"render {i}: {r.duration_seconds:.1f}s arena {e['arena']/1e6:.0f} MB chunks {e['chunks']} tiles {sorted(set(e['tiles']))}", flush=True)
    paths.append(p)
out = subprocess.run([sys.executable, "benchmarks/_video_diff.py", *paths], capture_output=True, text=True).stdout
print("DIFF", "\n".join(out.splitlines()[:3]), flush=True)
