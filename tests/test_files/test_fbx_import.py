from algan import *


def test_fbx_import():
    mob = (
        ThreeDModelMob("D:/algan/tests/dragon_mesh.glb", normalize=True)
        .scale(4)
        .spawn()
    )
    with Seq(run_time=1):
        mob.rotate(360, UP)


# render_all_funcs(__name__)
