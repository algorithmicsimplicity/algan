import os, sys, hashlib, json
os.environ["ALGAN_USE_DAEMON"] = "0"
import torch
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
from algan.rendering.raytracing.scene_builder import _merge_scene
from algan.render_loop import _projection_anti_alias_level
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=0.5):
    nn.move(UP)
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
    label.move(RIGHT*2)
sc = SceneManager.instance().current_scene
sc.set_video_settings(UHD)
actors = [sc.camera, sc.camera.screen, *sc.light_sources, *sc.actors]
def h(t): return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
with sc.batch_prep_context():
    prims, end, rs = sc.get_batch_of_primitives(0, 3, actors, 10**12)
# project on the render thread exactly as the preflight does, then merge
from algan.utils.memory_utils import ManualMemory
sc.memory = ManualMemory(SETTINGS.computing.rendering_memory_fraction, managed=True)
sc._prewarm_render_batch(prims, rs)
merged, env = sc._prepare_merged_host_scene(prims)
out = {}
for k in sorted(merged.keys()):
    v = merged[k]
    if torch.is_tensor(v): out[k] = (str(v.dtype).replace("torch.", ""), list(v.shape), h(v))
    elif isinstance(v, (int, float, bool, str)): out[k] = v
    elif hasattr(v, "blocks"): out[k] = ("bvh", h(v.blocks), h(v.leaf_prim), h(v.leaf_tspan))
json.dump(out, open(sys.argv[1], "w"), indent=0, default=str)
print("wrote", sys.argv[1], len(out))
