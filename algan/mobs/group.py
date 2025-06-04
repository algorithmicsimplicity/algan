import torch

from algan.animation.animation_contexts import Sync
from algan.constants.spatial import RIGHT, DOWN
from algan.mobs.mob import Mob
from algan.defaults.style_defaults import DEFAULT_BUFFER
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import dot_product, broadcast_gather


class Group(Mob):
    r"""Combine a collection of Mobs into a single Mob.

    Specifically, creates an empty mob at the mid-point of the bounding box of the
    given mob collection and adds all of the mobs as children.

    Parameters
    ----------
    mobs
        The collection of mobs to group.
    args
        args for :class:`Mob`.
    kwargs
        kwargs for :class:`Mob`.

    Returns
    -------
    :class:`Group`
        The new mob which parents the provided mob collection.

    Examples
    --------
    .. algan:: Example1Group

        group = Group([Square() for _ in range(3)]).arrange_in_line(RIGHT).spawn()
        group.rotate(90, OUT)

        render_to_file()
    """
    def __init__(self, mobs, *args, **kwargs):
        def mean(x):
            x = [_ for _ in x if _ is not None]
            return torch.stack([_.mean(-2, keepdim=True) for _ in x], -1).mean(-1)

        def median(x):
            mn = torch.stack([_.amin(-2, keepdim=True) for _ in x], -1).amin(-1)
            mx = torch.stack([_.amax(-2, keepdim=True) for _ in x], -1).amax(-1)
            return (mn + mx) / 2
        super().__init__(median([mob.location for mob in mobs]), color=mean(list(traverse([mob.color for mob in mobs]))), init=False, *args, **kwargs)
        if all([_.data.spawn_time() >= 0 for _ in mobs]):
            self.spawn(animate=False)
        self.traversable = False
        self.add_children(mobs)
        self.mobs = mobs

    def __getitem__(self, item):
        return self.mobs[item]

    def __setitem__(self, item, value):
        self.mobs[item] = value

    def __iter__(self):
        return self.mobs.__iter__()

    def __len__(self):
        return len(self.mobs)

    def get_parts_as_mobs(self):
        return self.mobs

    def get_boundary_edge_point2(self, direction):
        points = torch.stack([(m.get_boundary_edge_point(direction)) for m in self.mobs])
        dots = dot_product(points, direction)
        furthest_ind = dots.argmax(0, keepdim=True)
        return broadcast_gather(points, 0, furthest_ind, keepdim=False)

    def arrange_in_line(self, direction=RIGHT, buffer=DEFAULT_BUFFER, start_at_first=False, equal_displacement=False, alignment_direction=None):
        mob_sizes = [(m.get_boundary_in_direction(direction) - m.get_boundary_in_direction(-direction)).norm(p=2,dim=-1, keepdim=True) for m in self.mobs]
        if alignment_direction is not None:
            alignment_dists = [(m.get_boundary_in_direction(alignment_direction) - m.get_center()).norm(p=2,dim=-1) for m in self.mobs]
            max_dist = max(alignment_dists)
            alignment_offsets = [max_dist - _ for _ in alignment_dists]
        if equal_displacement:
            max_size = max(mob_sizes)
            mob_sizes = [max_size for _ in range(len(mob_sizes))]
        total_size = sum(mob_sizes) + (buffer * (len(mob_sizes)-1))

        start = (self.mobs[0].location - direction * (mob_sizes[0]/2)) if start_at_first else (self.location - direction * total_size/2)
        with Sync():
            for i, mob in enumerate(self.mobs):
                start = start + direction * (mob_sizes[i] / 2)
                l = start
                if alignment_direction is not None:
                    l = l + alignment_offsets[i] * alignment_direction
                mob.location = l
                start = start + direction * (mob_sizes[i] / 2 + buffer)
        return self

    def arrange_between_points(self, start, end):
        dif = end-start
        with Sync():
            for i, mob in enumerate(self.mobs):
                mob.location = start + dif * ((i+1) / (len(self.mobs)+1))
        return self

    def arrange_in_grid(self, direction1=RIGHT, direction2=DOWN, num_rows=2, buffer=DEFAULT_BUFFER, buffer2=None):
        if buffer2 is None:
            buffer2 = buffer
        buf_dist1 = max([(m.get_boundary_edge_point(direction1) - m.get_boundary_edge_point(-direction1)).norm(p=2,dim=-1, keepdim=True) for m in self.mobs]) + buffer
        buf_dist2 = max([(m.get_boundary_edge_point(direction2) - m.get_boundary_edge_point(-direction2)).norm(p=2,dim=-1, keepdim=True) for m in self.mobs]) + buffer2
        num_cols = len(self.mobs) // num_rows
        start = self.location - (direction1 * buf_dist1 * (num_cols-1)/2 + direction2 * buf_dist2 * (num_rows-1)/2)
        with Sync():
            for i, mob in enumerate(self.mobs):
                mob.location = start + direction1 * buf_dist1 * (i%num_cols) + direction2 * buf_dist2 * (i//num_cols)
        return self

    def highlight(self):
        with Sync():
            for m in self.mobs:
                m.highlight()

    def highlight_off(self):
        with Sync():
            for m in self.mobs:
                m.highlight_off()
