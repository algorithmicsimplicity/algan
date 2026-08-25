import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ["ALGAN_RENDER_DEVICE"] = "cpu"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
import torch
variant = sys.argv[1]
SETTINGS.raytracing.set(shadows=True)
fps = 10
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    if variant != "noimage":
        x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=5):
    nn.move(UP)
    if variant == "full":
        x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
    label.move(RIGHT*2)
sc = SceneManager.instance().current_scene
sc.set_video_settings(PREVIEW)
actors = [sc.camera, sc.camera.screen, *sc.light_sources, *sc.actors]
tl = sc.timeline_manager
with sc.batch_prep_context():
    for T in (12, 12, 12):
        times = torch.arange(0, T) / fps
        t0 = time.perf_counter(); tl.set_state_to_times(times, active_mobs=actors); t1 = time.perf_counter()
        t2 = time.perf_counter(); prims, end, rs = sc.get_batch_of_primitives(0, T, actors, 10**12); t3 = time.perf_counter()
    print(f"variant={variant:12s} T={T} set_state_to_times {(t1-t0)/T*1000:6.1f} ms/frame   get_batch_of_primitives {(t3-t2)/T*1000:6.1f} ms/frame")
