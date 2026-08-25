import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
import torch, cProfile, pstats
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
T = int(sys.argv[1]) if len(sys.argv) > 1 else 21
with sc.batch_prep_context():
    for i in range(2):
        t0 = time.perf_counter(); prims, end, rs = sc.get_batch_of_primitives(0, T, actors, 10**12); torch.cuda.synchronize(); t1 = time.perf_counter()
        print(f"T={end} get_batch_of_primitives {(t1-t0)/end*1000:6.1f} ms/frame")
        del prims
    pr = cProfile.Profile(); pr.enable()
    prims, end, rs = sc.get_batch_of_primitives(0, T, actors, 10**12); torch.cuda.synchronize()
    pr.disable()
    st = pstats.Stats(pr); st.sort_stats("tottime").print_stats(30); st.sort_stats("cumulative").print_stats(40)
    st.dump_stats("scratch_perf/prep_T21.pstats")
    import io
    for pat in ("method 'to' of", "grid_to_triangle_vertices", "torch.cat", "expand_grid_to_verts"):
        s = io.StringIO(); stc = pstats.Stats("scratch_perf/prep_T21.pstats", stream=s); stc.sort_stats("tottime").print_callers(pat)
        print("CALLERS OF", pat); print("\n".join(l for l in s.getvalue().splitlines() if "/" in l or "{" in l)[:3000])
