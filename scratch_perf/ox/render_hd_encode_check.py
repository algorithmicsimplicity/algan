import os
os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3

# Which encoding arm this run is; the output file is named after it.
arm = os.environ.get("ALGAN_VIDEO_ENCODER", "auto-default")
if os.environ.get("FFMPEG_BINARY"):
    arm += "_sysffmpeg"

def scene():
    run_time=0.5
    SETTINGS.raytracing.set(shadows=True)

    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob('world_map.png').move_next_to(nn, LEFT).spawn()
        label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()

    with Sync(run_time=run_time):
        nn.move(UP)
        x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
        label.move(RIGHT*2)

scene()
Scene.save_video(f"nn_HD_{arm}", HD)
