import os
os.environ['ALGAN_PREFETCH_BATCHES'] = '0'
from algan import *
from algan.utils.profiling_utils import profile_scene

rs = MD
one_mob = True

def empty_scene():
    if one_mob:
        with Off():
            Triangle().scale(0.1).spawn()
    Scene.wait(10)

profile_scene(empty_scene, video_settings=rs, runs=2, kernel_profiler=False)