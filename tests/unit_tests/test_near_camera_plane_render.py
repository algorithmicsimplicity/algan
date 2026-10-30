"""Geometry at the camera plane renders like geometry just across it.

Two defects of the sheet route's emission, each pinned by a scene that showed
it:

* A floor slab whose near edge is a hair IN FRONT of the camera plane projects
  its near vertices millions of pixels off-screen. The projected path's int64
  coverage lattice overflowed on them and took the wrong orientation, and the
  whole lower half of this frame came out black. The same slab nudged to
  straddle the plane was always fine (the ray cast draws it), so it is the
  reference.
* A triangle crossing the camera plane is ray-cast, and the ray cast never set
  the facing bit, so a straddler seen from behind read as front-facing while its
  projected neighbour read as back-facing. The band rule then split the shared
  diagonal of a two-triangle floor seen from below into two sheets, and it
  showed as a dashed dark seam (down to 0.70 of the surrounding brightness).
  Seen from the other side -- the same surface, wound the other way -- both
  triangles agree, so that is the reference.
"""

import numpy as np
import torch
from PIL import Image

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE
from algan.mobs.shapes_3d import Prism
from algan.mobs.surfaces.surface import Surface
from algan.scene import Scene
from algan.settings.video_settings import LD

W, H = 192, 108
VIDEO = LD.set(resolution=(W, H))


def _rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int32)


def test_a_slab_edge_just_in_front_of_the_camera_draws_like_one_across_it(tmp_path):
    frames = {}
    with Scene() as scene:
        eye = scene.get_camera().location.reshape(-1).clone()
        # 0.0005 in front: at this resolution (max|sx| + W)(max|sy| + H) is
        # about 2**43 for the slab's top face, far past the lattice bound.
        # -0.002 straddles the plane: the ray-cast reference.
        for name, delta in (("front", 0.0005), ("straddle", -0.002)):
            with Off():
                slab = Prism(width=60.0, height=0.05, depth=40, color=BLUE)
                slab.move_to(eye + torch.tensor([0.0, -1.025, -delta - 20.0]))
            slab.spawn(animate=False)
            path = tmp_path / f"{name}.png"
            scene.save_frame(str(path), video_settings=VIDEO)
            slab.despawn(animate=False)
            frames[name] = _rgb(path)
    front, reference = frames["front"], frames["straddle"]
    floor = reference.sum(-1) > 30
    # The reference shows the slab over the whole bottom of the frame.
    assert floor[H // 2 + 8 :].all()
    drawn = front.sum(-1) > 30
    assert int((drawn != floor).sum()) <= W, (
        f"{int((floor & ~drawn).sum())} slab pixels missing, "
        f"{int((drawn & ~floor).sum())} extra"
    )
    assert int((np.abs(front - reference).max(-1) > 8).sum()) <= W


def _floor(flip):
    def coords(uv):
        u, v = uv[..., :1], uv[..., 1:2]
        if flip:
            u, v = v, u
        return torch.cat((60 * u - 30, torch.zeros_like(u), 40 * v - 20), -1)

    surface = Surface(
        coord_function=coords,
        grid_width=2,
        grid_height=2,
        ignore_normals=True,
        color=BLUE,
    )
    surface.move_to(torch.tensor([0.0, -1.0, -2.0]))
    return surface


def test_an_open_surface_seen_from_behind_with_a_straddler_has_no_seam(tmp_path):
    frames = {}
    with Scene() as scene:
        with Off():
            # Turned so the floor's near right corner is behind the camera
            # plane: of its two triangles, one straddles and one is all-front.
            scene.get_camera().look_at(torch.tensor([-3.47, -2.0, 0.3]))
        for name, flip in (("behind", False), ("in_front", True)):
            with Off():
                surface = _floor(flip)
            surface.spawn(animate=False)
            path = tmp_path / f"{name}.png"
            scene.save_frame(str(path), video_settings=VIDEO)
            surface.despawn(animate=False)
            frames[name] = _rgb(path)
    behind, in_front = frames["behind"], frames["in_front"]
    assert (in_front.sum(-1) > 30).sum() > W * H // 2
    diff = np.abs(behind - in_front).max(-1)
    assert int(diff.max()) <= 3, (
        f"{int((diff > 3).sum())} pixels differ by more than 3 levels "
        f"(at most {int(diff.max())})"
    )
