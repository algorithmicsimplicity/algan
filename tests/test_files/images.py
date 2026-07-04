from algan import *
from algan.utils.algan_utils import render_all_funcs


def images():
    #n = 4
    #m = math.isqrt(4 * n)
    #rgba_array = torch.stack((RED, BLUE, GREEN, PURPLE)).unsqueeze(0).expand(n, -1, -1).reshape(m, m, -1)
    #x = ImageMob(rgba_array).spawn()
    x = ImageMob('world_map.jpg').spawn().despawn()
    #x.scale(3)

render_all_funcs(__name__)
