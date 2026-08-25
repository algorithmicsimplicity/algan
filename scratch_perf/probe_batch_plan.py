import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
import torch, resource
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=5):
    nn.move(UP)
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
    label.move(RIGHT*2)
sc = SceneManager.instance().current_scene
sc.set_video_settings(PREVIEW)
actors = [sc.camera, sc.camera.screen, *sc.light_sources, *sc.actors]
cpu = sum(a._get_memory_used_per_timestep() for a in sc.actors if hasattr(a, "get_render_primitives"))
gpu = sum(a._get_render_device_memory_used_per_timestep() for a in sc.actors if hasattr(a, "get_render_primitives"))
print(f"per-frame estimate: cpu {cpu/1e6:.1f} MB, render-device {gpu/1e6:.1f} MB; cpu budget {0.15*SETTINGS.computing.max_cpu_memory_used/1e6:.0f} MB; render budget {sc._render_device_prep_budget()/1e6:.0f} MB")
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
with sc.batch_prep_context():
    for T in (3, 12, 24):
        t0 = time.perf_counter(); prims, end, rs = sc.get_batch_of_primitives(0, T, actors, int(0.15*SETTINGS.computing.max_cpu_memory_used)); torch.cuda.synchronize(); t1 = time.perf_counter()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        print(f"requested T={T:3d} -> end={end:3d}  {(t1-t0)/max(1,end)*1000:6.1f} ms/frame  maxrss {rss:.0f} MB  cuda alloc {torch.cuda.memory_allocated()/1e6:.0f} MB peak {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
        del prims
