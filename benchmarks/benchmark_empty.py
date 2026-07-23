from algan import *
from algan.utils.profiling_utils import profile_scene

rs = MD

def empty_scene():
    with Off():
        Triangle().scale(0).spawn() # small Mob cause scene won't render if there are 0 Mobs in it.
    Scene.wait(10)

profile_scene(empty_scene, render_settings=rs, runs=2, kernel_profiler=False)