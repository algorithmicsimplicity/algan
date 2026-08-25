"""Active-ray counts per bounce iteration and shade time, UHD, a few frames."""
import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
import torch
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
import algan.rendering.raytracing.tracer as tr
import algan.rendering.raytracing.raster_pipeline as rpl
sizes = []
orig_select = tr._ArenaRayCompactor.select
def select(self, *a, **k):
    out = orig_select(self, *a, **k); sizes.append(int(out.numel())); return out
tr._ArenaRayCompactor.select = select
shade_t = []
orig_shade = tr.wavefront_shade
def shade(*a, **k):
    torch.cuda.synchronize(); t0 = time.perf_counter(); r = orig_shade(*a, **k); torch.cuda.synchronize(); shade_t.append((int(a[1]), time.perf_counter() - t0)); return r
tr.wavefront_shade = shade
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=4/60):
    nn.move(UP); x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5; label.move(RIGHT*2)
Scene.save_video("scratch_perf/bounce_probe.mp4", UHD, overwrite=True, ffmpeg_params=["-preset", "ultrafast"])
print("compactor sizes per select:", sizes[:40])
print("shade launches (active rays, seconds):", [(n, round(t, 3)) for n, t in shade_t[:40]])
print("total shade", round(sum(t for _, t in shade_t), 2), "s over", len(shade_t), "launches")
