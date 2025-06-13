import torch
import torch.nn.functional as F

from algan.animation.animation_contexts import Sync
from algan.constants.spatial import RIGHT, DOWN
from algan.mobs.mob import Mob
from algan.defaults.style_defaults import DEFAULT_BUFFER
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import dot_product, broadcast_gather


class Group(Mob):
    r"""Combine a collection of Mobs into a single Mob.

    Specifically, creates an empty mob at the mid-point of the bounding box of the
    given mob collection and adds the mobs as children.

    Parameters
    ----------
    mobs : Iterable[ :class:`~.Mob` ]
        The collection of mobs to group.
    *args, **kwargs
        Passed to :class:`~.Mob` .

    Returns
    -------
    :class:`~.Group`
        The new mob which parents the provided mob collection.

    Examples
    --------
    Arrange 3 mobs horizontally in a line, left to right.

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

    def arrange_in_line(self, direction:torch.Tensor=RIGHT, buffer:float=DEFAULT_BUFFER, start_at_first:bool=False,
                        equal_displacement:bool=False, alignment_direction:torch.Tensor|None=None):
        """Moves the grouped mobs so that they lie along a given line.

        Parameters
        ----------
        direction
            Vector in 3-D specifying the direction of the line. Defaults to RIGHT.
        buffer
            The amount of extra space added between the mobs. If 0, the mobs will be arranged edge-to-edge.
        start_at_first
            if True, the first mob's position will be unchanged, and the subsequent mobs will
            be arranged starting from the first mob's position.
            If False, the mobs will be arranged so that their center is equal to this Group's location.
        equal_displacement
            If True, the mobs will be arranged at evenly spaced intervals.
        alignment_direction
            If not None, the mobs will additionally be aligned on this direction.

        Returns
        -------
        :class:`~.Group`
            The Group instance itself, allowing for method chaining.

        """

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

    def arrange_in_grid(self, num_rows:int=2, row_direction:torch.Tensor=RIGHT, column_direction:torch.Tensor=DOWN,
                        row_buffer=DEFAULT_BUFFER, column_buffer=None):
        """Moves the grouped mobs so that they in a given grid.

        Parameters
        ----------
        num_rows
            The number of rows in the grid. The number of columns id then derived as len(mobs) // num_rows.
        row_direction
            Vector in 3-D specifying the direction along which rows are aligned.
            Defaults to RIGHT.
        column_direction
            Vector in 3-D specifying the direction along which columns are aligned.
            Defaults to DOWN.
        row_buffer
            The amount of extra space added between the mobs in the row direction.
        column_buffer
            The amount of extra space added between the mobs in the column direction. If None then
            it is set to `row_buffer`.

        Returns
        -------
        :class:`~.Group`
            The Group instance itself, allowing for method chaining.

        Examples
        --------

        Arrange mobs in a 3x3 grid slanted at a 45 degrees angle.

        .. algan:: Example1ArrangeInGrid

            group = Group([Square() for _ in range(9)]).scale(1/3).arrange_in_grid(3, RIGHT+UP, RIGHT+DOWN).spawn()
            group.rotate(90, OUT)

            render_to_file()

        """
        if column_buffer is None:
            column_buffer = row_buffer
        row_direction = F.normalize(row_direction, p=2, dim=-1)
        column_direction = F.normalize(column_direction, p=2, dim=-1)
        buf_dist1 = max([m.get_length_in_direction(row_direction) for m in self.mobs]) + row_buffer
        buf_dist2 = max([m.get_length_in_direction(column_direction) for m in self.mobs]) + column_buffer
        num_cols = len(self.mobs) // num_rows
        start = self.location - (row_direction * buf_dist1 * (num_cols-1)/2 + column_direction * buf_dist2 * (num_rows-1)/2)
        with Sync():
            for i, mob in enumerate(self.mobs):
                mob.location = start + row_direction * buf_dist1 * (i%num_cols) + column_direction * buf_dist2 * (i//num_cols)
        return self

    def highlight(self):
        with Sync():
            for m in self.mobs:
                m.highlight()

    def highlight_off(self):
        with Sync():
            for m in self.mobs:
                m.highlight_off()
