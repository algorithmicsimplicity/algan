import os, sys, time
os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ["ALGAN_RENDER_DEVICE"] = "cpu"
from algan import *
from algan.scene_manager import SceneManager
import torch
with Off():
    x = ImageMob('benchmarks/performance/world_map.png').spawn()
with Sync(run_time=5):
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
sc = SceneManager.instance().current_scene
tl = sc.timeline_manager
name = x._color_texture_attr
t = tl.attr_to_timeline[name]
print("attr", name, "pointer(rows used)", t.pointer, "buffer", tuple(t.current_state.shape))
for mid, inds in t.mob_id_to_inds.items():
    print("  mob id", mid, "rows", inds.tolist())
print("x.id", x.id, "grid.id", x.grid.id, "children", [type(c).__name__ for c in x.children])
print("edits", [(e.indexes if not torch.is_tensor(e.indexes) else e.indexes.tolist(), e.time) for e in t.edits])
print("is texture attr in x.animatable_attrs?", name in x.animatable_attrs, "grid?", name in getattr(x.grid, 'animatable_attrs', []))
