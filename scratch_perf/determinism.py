"""Render the nn scene N times in one process and diff consecutive outputs.
usage: uv run python scratch_perf/determinism.py <preset> <runs> <tag> [max_batch]"""
import os, sys, subprocess
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
preset = {"PREVIEW": PREVIEW, "UHD": UHD, "HD": HD}[sys.argv[1]]
runs = int(sys.argv[2]); tag = sys.argv[3]
if len(sys.argv) > 4:
    SETTINGS.computing.set(max_animation_batch_size=int(sys.argv[4]))
run_time = 5 if sys.argv[1] == "PREVIEW" else 0.5
def scene():
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
        label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
    with Sync(run_time=run_time):
        nn.move(UP)
        x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
        label.move(RIGHT*2)
from algan.scene_manager import SceneManager
paths = []
for i in range(runs):
    SceneManager.reset()
    scene()
    p = f"scratch_perf/det_{tag}_{i}.mp4"
    r = Scene.save_video(p, preset, overwrite=True, reset=True)
    print("rendered", p, f"{r.duration_seconds:.1f}s", flush=True)
    paths.append(p)
for a, b in zip(paths, paths[1:]):
    out = subprocess.run([sys.executable, "benchmarks/_video_diff.py", a, b], capture_output=True, text=True).stdout
    print(f"DIFF {a} vs {b}:", "\n".join(out.splitlines()[:3]), flush=True)
