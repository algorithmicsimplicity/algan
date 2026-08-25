import os, sys, hashlib, json
os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")
import torch
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
def h(t): return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:10]
core = nn.layers[0][0].core; syn = nn.layers[1][0].synapses[0]
out = {
  "label.location": label.location.reshape(-1)[:3].tolist(),
  "x.location": x.location.reshape(-1)[:3].tolist(),
  "core.grid": [core.grid_width, core.grid_height], "core.grid.hash": h(core.grid.location),
  "syn.grid": [syn.grid_width, syn.grid_height], "syn.grid.hash": h(syn.grid.location),
  "label.points.hash": h(label.location), "label.npoints": int(label.location.shape[-2]),
  "x.grid.hash": h(x.grid.location), "tex.hash": h(x._color_texture_uncopied()),
  "nn.bbox": [float(v) for v in torch.cat([nn.get_boundary_points().reshape(-1,3).amin(0), nn.get_boundary_points().reshape(-1,3).amax(0)])] if hasattr(nn, "get_boundary_points") else None,
}
print(json.dumps(out))
