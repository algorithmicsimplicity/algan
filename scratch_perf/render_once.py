"""Render the nn scene once with a fast x264 preset. usage: render_once.py PREVIEW|UHD <out.mp4>"""
import os, sys
os.environ["ALGAN_USE_DAEMON"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
preset = {"PREVIEW": PREVIEW, "UHD": UHD, "HD": HD}[sys.argv[1]]
run_time = 5 if sys.argv[1] == "PREVIEW" else 0.5
SETTINGS.raytracing.set(shadows=True)
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob('benchmarks/performance/world_map.png').move_next_to(nn, LEFT).spawn()
    label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()
with Sync(run_time=run_time):
    nn.move(UP)
    x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
    label.move(RIGHT*2)
r = Scene.save_video(sys.argv[2], preset, overwrite=True, ffmpeg_params=["-crf", "17", "-preset", "ultrafast"])
print("rendered", sys.argv[2], f"{r.duration_seconds:.1f}s")
