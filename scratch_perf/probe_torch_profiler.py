"""torch.profiler capture (CPU+CUDA activities) of a short UHD render. usage: probe_torch_profiler.py [frames]"""
import os, sys
os.environ["ALGAN_USE_DAEMON"] = "0"
import torch
from torch.profiler import profile, ProfilerActivity
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
frames = int(sys.argv[1]) if len(sys.argv) > 1 else 6
def scene(run_time):
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
        label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
    with Sync(run_time=run_time):
        nn.move(UP)
        x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
        label.move(RIGHT*2)
# warm run (compiles kernels), then the profiled run
scene(frames / 60); Scene.save_video("scratch_perf/tp_warm.mp4", UHD, overwrite=True, reset=True, ffmpeg_params=["-preset", "ultrafast"])
SceneManager.reset(); scene(frames / 60)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    r = Scene.save_video("scratch_perf/tp_prof.mp4", UHD, overwrite=True, reset=True, ffmpeg_params=["-preset", "ultrafast"])
print("rendered", f"{r.duration_seconds:.1f}s for {frames} frames")
ka = prof.key_averages()
print(ka.table(sort_by="cuda_time_total", row_limit=45, max_name_column_width=70))
print(ka.table(sort_by="cpu_time_total", row_limit=30, max_name_column_width=70))
