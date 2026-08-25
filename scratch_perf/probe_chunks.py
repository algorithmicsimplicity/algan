"""Render UHD twice in one process, log the render-chunk plan of each, diff outputs.
usage: uv run python scratch_perf/probe_chunks.py <tag> [fixed_chunk_frames]"""
import os, sys, subprocess, logging
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
import algan.render_loop as rl
from algan.rendering.memory_model import ChunkMemoryModel
tag = sys.argv[1]
fixed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
plans = []
orig_plan = ChunkMemoryModel.plan
def plan(self, signature, requested, available):
    d = orig_plan(self, signature, requested, available)
    if fixed:
        d = min(fixed, requested)
    plans[-1].append(int(d))
    return d
ChunkMemoryModel.plan = plan
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
    plans.append([])
    SceneManager.reset(); scene()
    p = f"scratch_perf/chunks_{tag}_{i}.mp4"
    r = Scene.save_video(p, UHD, overwrite=True, reset=True, ffmpeg_params=["-preset", "ultrafast"])
    print(f"render {i}: {r.duration_seconds:.1f}s chunk plan {plans[-1]}", flush=True)
    paths.append(p)
out = subprocess.run([sys.executable, "benchmarks/_video_diff.py", *paths], capture_output=True, text=True).stdout
print("DIFF", "\n".join(out.splitlines()[:3]), flush=True)
