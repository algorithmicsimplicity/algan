import math

import torch
import torch.nn.functional as F

from algan.animation.animation_contexts import Sync, Off
from algan.constants.spatial import RIGHT, DOWN, ORIGIN
from algan.mobs.mob import Mob
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import dot_product, broadcast_gather


def midpoint(x):
    mn = torch.stack([_.amin(-2, keepdim=True) for _ in x], -1).amin(-1)
    mx = torch.stack([_.amax(-2, keepdim=True) for _ in x], -1).amax(-1)
    return (mn + mx) / 2


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

    def __init__(self, *mobs, **kwargs):
        mobs = list(traverse(mobs))
        self.mobs = mobs
        if len(mobs) == 0:
            super().__init__(ORIGIN, **kwargs)
            return

        def mean(x):
            x = [_ for _ in x if _ is not None]
            if len(x) == 0:
                return None
            return torch.stack([_.mean(-2, keepdim=True) for _ in x], -1).mean(-1)

        self.traversable = False
        super().__init__(
            self.get_mob_midpoint(),
            color=mean(list(traverse([mob.color for mob in mobs if hasattr(mob, 'color')]))),
            **kwargs,
        )

        self.traversable = False
        if all([mob.is_spawned() for mob in mobs]):
            self.spawn(animate=False)
        self.add_children(mobs)

    def get_mob_midpoint(self):
        return midpoint(
            list(
                traverse(
                    [[d.location for d in mob.get_descendants()] for mob in self.mobs]
                )
            )
        )

    def __getitem__(self, item):
        mobs = self.mobs[item]
        if isinstance(mobs, Mob):
            return mobs
        if len(mobs) == 0:
            return Mob(
                add_to_scene=False
            )  # return a null mob so we can still perform ops on the result, they just won't do anything.
        return Group(mobs)

    def __setitem__(self, item, value):
        self.mobs[item] = value

    def __iter__(self):
        return self.mobs.__iter__()

    def __len__(self):
        return len(self.mobs)

    def add(self, *mobs):
        """Add one or more Mobs and return this group.

        The variadic, chainable form matches Manim's ``Group.add`` while
        remaining backwards compatible with Algan's former single-Mob API.
        """
        mobs = list(traverse(mobs))
        if not mobs:
            return self
        self.add_children(mobs)
        self.mobs.extend(mobs)
        with Off():
            self.set_non_recursive(location=self.get_mob_midpoint())
        return self

    def get_parts_as_mobs(self):
        return self.mobs

    def get_boundary_edge_point2(self, direction):
        points = torch.stack(
            [(m.get_boundary_edge_point(direction)) for m in self.mobs]
        )
        dots = dot_product(points, direction)
        furthest_ind = dots.argmax(0, keepdim=True)
        return broadcast_gather(points, 0, furthest_ind, keepdim=False)

    def arrange_in_line(
        self,
        direction: torch.Tensor = RIGHT,
        buffer: float = STYLE_DEFAULTS.buffer,
        start_at_first: bool = False,
        equal_displacement: bool = False,
        alignment_direction: torch.Tensor | None = None,
    ):
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

        mob_sizes = [
            (
                m.get_boundary_in_direction(direction)
                - m.get_boundary_in_direction(-direction)
            ).norm(p=2, dim=-1, keepdim=True)
            for m in self.mobs
        ]
        if alignment_direction is not None:
            alignment_dists = [
                (
                    m.get_boundary_in_direction(alignment_direction) - m.get_center()
                ).norm(p=2, dim=-1)
                for m in self.mobs
            ]
            max_dist = max(alignment_dists)
            alignment_offsets = [max_dist - _ for _ in alignment_dists]
        if equal_displacement:
            max_size = max(mob_sizes)
            mob_sizes = [max_size for _ in range(len(mob_sizes))]
        total_size = sum(mob_sizes) + (buffer * (len(mob_sizes) - 1))

        start = (
            (self.mobs[0].location - direction * (mob_sizes[0] / 2))
            if start_at_first
            else (self.location - direction * total_size / 2)
        )
        with Sync():
            for i, mob in enumerate(self.mobs):
                start = start + direction * (mob_sizes[i] / 2)
                l = start
                if alignment_direction is not None:
                    l = l + alignment_offsets[i] * alignment_direction
                #loc + (disp_to_center) = l
                mob.location = l + (mob.location - mob.get_center())
                start = start + direction * (mob_sizes[i] / 2 + buffer)
        return self

    def arrange_between_points(self, start, end):
        dif = end - start
        with Sync():
            for i, mob in enumerate(self.mobs):
                mob.location = start + dif * ((i + 1) / (len(self.mobs) + 1))
        return self

    def arrange_in_grid(
        self,
        num_rows: int = None,
        row_direction: torch.Tensor = RIGHT,
        column_direction: torch.Tensor = DOWN,
        buffer=STYLE_DEFAULTS.buffer,
        column_buffer=None,
        tight_axis=None,
    ):
        """Moves the grouped mobs so that they in a given grid.

        Parameters
        ----------
        num_rows
            The number of rows in the grid. The number of columns id then derived as len(mobs) // num_rows.
            Defaults to sqrt(len(mobs)).
        row_direction
            Vector in 3-D specifying the direction along which rows are aligned.
            Defaults to RIGHT.
        column_direction
            Vector in 3-D specifying the direction along which columns are aligned.
            Defaults to DOWN.
        buffer
            The amount of extra space added between the mobs in the row direction.
        column_buffer
            The amount of extra space added between the mobs in the column direction. If None then
            it is set to `buffer`.

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
            column_buffer = buffer
        if num_rows is None:
            num_rows = math.isqrt(len(self.mobs))
        num_cols = len(self.mobs) // num_rows
        if num_rows * num_cols < len(self.mobs):
            num_cols += 1
        row_direction = F.normalize(row_direction, p=2, dim=-1)
        column_direction = F.normalize(column_direction, p=2, dim=-1)
        buf_dist1 = [
            max([m.get_length_in_direction(row_direction) for m in self.mobs]) + buffer
            for _ in range(num_cols)
        ]
        buf_dist2 = [
            max([m.get_length_in_direction(column_direction) for m in self.mobs])
            + column_buffer
            for _ in range(num_rows)
        ]
        if tight_axis is not None:
            if tight_axis == 0:
                buf_dist1 = [
                    max(
                        [
                            self.mobs[i + j * num_cols].get_length_in_direction(
                                row_direction
                            )
                            for j in range(num_rows)
                        ]
                    )
                    + column_buffer
                    for i in range(num_cols)
                ]
            elif tight_axis == 1:
                buf_dist2 = [
                    max(
                        [
                            self.mobs[i + j * num_cols].get_length_in_direction(
                                row_direction
                            )
                            for i in range(num_cols)
                        ]
                    )
                    + buffer
                    for j in range(num_rows)
                ]

        start = self.location - (
            row_direction * sum(buf_dist1) * 0.5
            + column_direction * sum(buf_dist2) * 0.5
        )
        with Sync():
            for i, mob in enumerate(self.mobs):
                x = i % num_cols
                y = i // num_cols
                x_dist = sum(buf_dist1[:x]) + buf_dist1[x] * 0.5
                y_dist = sum(buf_dist2[:y]) + buf_dist2[y] * 0.5
                mob.location = (
                    start + row_direction * x_dist + column_direction * y_dist
                ) + (mob.location - mob.get_center())
        return self
