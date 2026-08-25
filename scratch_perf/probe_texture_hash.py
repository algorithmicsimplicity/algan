import os, sys, hashlib, json
os.environ["ALGAN_USE_DAEMON"] = "0"
import torch
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager
from algan.rendering.raytracing.scene_builder import _merge_scene
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
def h(t): return hashlib.md5(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
with sc.batch_prep_context():
    prims, end, rs = sc.get_batch_of_primitives(0, 3, actors, 10**12)
tex_prims = [p for p in prims if getattr(p, "texture_map", None) is not None]
out = {"n_prims": len(prims), "n_textured": len(tex_prims), "end": end}
for i, p in enumerate(tex_prims):
    out[f"tex{i}.device"] = str(p.texture_map.device); out[f"tex{i}.shape"] = list(p.texture_map.shape); out[f"tex{i}.hash"] = h(p.texture_map)
    out[f"tex{i}.corners.hash"] = h(p.corners); out[f"tex{i}.uvs.hash"] = h(p.uvs) if p.uvs is not None else None
# a non-textured primitive too
others = [p for p in prims if getattr(p, "texture_map", None) is None]
out["other0.corners.hash"] = h(others[0].corners) if others else None
out["other0.colors.hash"] = h(others[0].colors) if others else None
print(json.dumps(out))
