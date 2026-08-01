import os

from algan import *
from algan.utils.algan_utils import render_all_funcs

# The image lives beside the tests package, not wherever pytest was launched.
WORLD_MAP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "world_map.jpg"
)


def images():
    # n = 4
    # m = math.isqrt(4 * n)
    # rgba_array = torch.stack((RED, BLUE, GREEN, PURPLE)).unsqueeze(0).expand(n, -1, -1).reshape(m, m, -1)
    # x = ImageMob(rgba_array).spawn()
    x = ImageMob(WORLD_MAP).spawn().despawn()
    # x.scale(3)


render_all_funcs(__name__)
