import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ["ALGAN_RENDER_DEVICE"] = "cpu"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
import torch, cProfile, pstats
SETTINGS.raytracing.set(shadows=True)
fps = 10
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
tl = sc.timeline_manager
with sc.batch_prep_context():
    for T in (3, 3, 6, 12, 24):
        times = torch.arange(0, T) / fps
        t0 = time.perf_counter()
        tl.set_state_to_times(times, active_mobs=actors)
        t1 = time.perf_counter()
        print(f"T={T:3d} set_state_to_times {t1-t0:7.3f}s  per-frame {(t1-t0)/T*1000:6.1f} ms")
    # per-attribute rematerialize timing at T=24
    times = torch.arange(0, 24) / fps
    tl._resolve_replay_windows()
    functions = tl.function_timeline.get_functions_for_times(times)
    updaters = tl.function_timeline.get_updaters_for_times(times)
    ids = tl._active_mob_ids(actors, functions, updaters)
    print("functions", len(functions), "updaters", len(updaters))
    for name, timeline in tl.attr_to_timeline.items():
        t0 = time.perf_counter(); timeline.rematerialize_state_at_times(times, ids); dt = time.perf_counter()-t0
        if dt > 0.005: print(f"  remat {name:30s} {dt*1000:7.1f} ms  active_state {tuple(timeline.active_state.shape)}")
    # full get_batch_of_primitives timing
    for T in (3, 12, 24):
        t0 = time.perf_counter()
        prims, end, rs = sc.get_batch_of_primitives(0, T, actors, 10**12)
        t1 = time.perf_counter()
        print(f"T={T:3d} get_batch_of_primitives {t1-t0:7.3f}s per-frame {(t1-t0)/T*1000:6.1f} ms  end={end} nprims={len(prims)}")
    pr = cProfile.Profile(); pr.enable()
    tl.set_state_to_times(torch.arange(0, 12) / fps, active_mobs=actors)
    pr.disable()
    st = pstats.Stats(pr); st.sort_stats("cumulative").print_stats(45)
    st.sort_stats("tottime").print_stats(25)
