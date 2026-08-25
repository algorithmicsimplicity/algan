"""Render UHD once, logging every accepted batch window and preflight verdict. usage: probe_windows.py <out.mp4>"""
import os, sys
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.render_loop import RenderLoopMixin
windows = []; verdicts = []
orig_rpb = RenderLoopMixin.render_primitive_batch
def rpb(self, primitive_batch, start_ind, end_ind, *a, **k):
    windows.append((int(start_ind), int(end_ind)))
    return orig_rpb(self, primitive_batch, start_ind, end_ind, *a, **k)
RenderLoopMixin.render_primitive_batch = rpb
orig_pf = RenderLoopMixin._prepared_batch_fits_render_arena
def pf(self, *a, **k):
    ok = orig_pf(self, *a, **k); verdicts.append((k.get("num_frames"), bool(ok), self._last_arena_preflight)); return ok
RenderLoopMixin._prepared_batch_fits_render_arena = pf
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=0.5):
    nn.move(UP)
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
    label.move(RIGHT*2)
if len(sys.argv) > 2: SETTINGS.computing.set(max_animation_batch_size=int(sys.argv[2]))
r = Scene.save_video(sys.argv[1], UHD, overwrite=True, ffmpeg_params=["-crf", "17", "-preset", "ultrafast"])
print("rendered", f"{r.duration_seconds:.1f}s", "windows", windows, "preflight", [(n, ok, None if p is None else tuple(int(v/1e6) for v in p)) for n, ok, p in verdicts])
