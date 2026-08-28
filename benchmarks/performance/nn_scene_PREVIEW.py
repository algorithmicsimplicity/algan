import os

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.utils.profiling_utils import profile_scene


def scene():
    run_time = 5
    SETTINGS.raytracing.set(shadows=True)

    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob("world_map.png").move_next_to(nn, LEFT).spawn()
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )

    with Sync(run_time=run_time):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


# The encoder is part of the measurement (the "video encode tail" stage). This
# benchmark box has 2 slow CPU cores, where libx264's default "slower" preset
# cannot keep up with the renderer at all; a fast preset keeps the rest of the
# profile representative of a machine with a full CPU (the output's quality is
# not what is being measured here).
profile_scene(
    scene,
    PREVIEW,
    "nn_PREVIEW",
    runs=2,
    kernel_profiler=False,
    save_video_kwargs=dict(ffmpeg_params=["-crf", "17", "-preset", "ultrafast"]),
)
