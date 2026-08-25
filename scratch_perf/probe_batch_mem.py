import os, sys
os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ["ALGAN_RENDER_DEVICE"] = "cpu"
sys.path.insert(0, "benchmarks/performance")
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from collections import Counter
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=5):
    nn.move(UP)
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
    label.move(RIGHT*2)
scene = Scene
from algan.scene_manager import SceneManager
sc = SceneManager.instance().current_scene
tot = Counter(); n = Counter()
for a in sc.actors:
    if hasattr(a, "get_render_primitives"):
        m = a._get_memory_used_per_timestep()
        tot[type(a).__name__] += m; n[type(a).__name__] += 1
for k in tot: print(f"{k:30s} n={n[k]:4d} bytes/frame={tot[k]/1e6:9.2f} MB")
print("TOTAL MB/frame", sum(tot.values())/1e6, "actors", len(sc.actors))
print("budget MB", 0.15*2e9/1e6)
print("timelines:", len(sc.timeline_manager.attr_to_timeline))
for name, tl in sc.timeline_manager.attr_to_timeline.items():
    cs = tl.current_state
    print(f"  {name:40s} shape={tuple(cs.shape)} edits={len(tl.edits) if hasattr(tl,'edits') else '?'}")
