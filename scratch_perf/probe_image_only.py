import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ["ALGAN_RENDER_DEVICE"] = sys.argv[1] if len(sys.argv) > 1 else "cpu"
from algan import *
from algan.scene_manager import SceneManager
import torch, cProfile, pstats
fps = 10
with Off():
    x = ImageMob('benchmarks/performance/world_map.png').spawn()
with Sync(run_time=5):
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
sc = SceneManager.instance().current_scene
sc.set_video_settings(PREVIEW)
actors = [sc.camera, sc.camera.screen, *sc.light_sources, *sc.actors]
tl = sc.timeline_manager
T = 12
with sc.batch_prep_context():
    for i in range(3):
        times = torch.arange(0, T) / fps
        t0 = time.perf_counter(); tl.set_state_to_times(times, active_mobs=actors); torch.cuda.synchronize() if torch.cuda.is_available() else None; t1 = time.perf_counter()
        t2 = time.perf_counter(); prims, end, rs = sc.get_batch_of_primitives(0, T, actors, 10**12); torch.cuda.synchronize() if torch.cuda.is_available() else None; t3 = time.perf_counter()
        print(f"image-only T={T} set_state_to_times {(t1-t0)/T*1000:6.1f} ms/frame   get_batch_of_primitives {(t3-t2)/T*1000:6.1f} ms/frame  active_state device {tl.attr_to_timeline[x._color_texture_attr].active_state.device}")
    pr = cProfile.Profile(); pr.enable()
    tl.set_state_to_times(torch.arange(0, T) / fps, active_mobs=actors)
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(18)
