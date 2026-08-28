"""Collecting Mobs so they can be treated as one.

:class:`Group` creates an invisible Mob at the centre of a collection and adds
everything to it as children, so every parent/child propagation rule applies:
move, rotate, scale or color the Group and the whole collection follows. Its
``mobs`` attribute is an alias for ``children``.

On top of that it adds layout: :meth:`Group.arrange_in_line` spreads members
along a direction and :meth:`Group.arrange_in_grid` lays them out in rows and
columns, both as ordinary animations, so members slide into place. Both use a
uniform cell size taken from the largest member.

Groups are indexable and iterable, and slicing returns a view rather than
registering new actors with the Scene.

See :doc:`/new_user_tutorials/child_mobs`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.spatial import DOWN, ORIGIN, RIGHT
from algan.errors import AlganConfigurationError
from algan.settings import SETTINGS
from algan.utils.python_utils import traverse
from algan.utils.tensor_utils import broadcast_gather, dot_product


def midpoint(x):
    """Internal: get the center of the box enclosing several point sets.

    Parameters
    ----------
    x
        Iterable of point tensors, each ``[..., N, 3]``.

    Returns
    -------
    torch.Tensor
        The midpoint of their combined extent, ``[..., 1, 3]``.
    """
    mn = torch.stack([_.amin(-2, keepdim=True) for _ in x], -1).amin(-1)
    mx = torch.stack([_.amax(-2, keepdim=True) for _ in x], -1).amax(-1)
    return (mn + mx) / 2


class Group(Mob):
    r"""Combine a collection of Mobs into a single Mob.

    Specifically, creates an empty mob at the mid-point of the bounding box of the
    given mob collection and adds the mobs as children.

    Parameters
    ----------
    mobs : Iterable[ :class:`~algan.animatable_base.mob.Mob` ]
        The collection of mobs to group.
    *args, **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob`.

    Returns
    -------
    :class:`~algan.mobs.group.Group`
        The new mob which parents the provided mob collection.

    Examples
    --------
    Arrange 3 mobs horizontally in a line, left to right.

    .. algan:: Example1Group

        from algan import *

        group = Group([Square() for _ in range(3)]).arrange_in_line(RIGHT).spawn()
        group.rotate(90, OUT)

        Scene.save_video()

    """

    def __init__(self, *mobs, _link_children=True, **kwargs):
        initial_mobs = list(traverse(mobs))
        self._link_children = bool(_link_children)
        if initial_mobs:
            scenes = {id(mob.scene): mob.scene for mob in initial_mobs}
            if len(scenes) != 1:
                raise AlganConfigurationError(
                    "A Group cannot contain Mobs from multiple Scenes"
                )
            child_scene = next(iter(scenes.values()))
            requested_scene = kwargs.setdefault("scene", child_scene)
            if requested_scene is not child_scene:
                raise AlganConfigurationError(
                    "A Group and its children must belong to the same Scene"
                )

        def mean(values):
            values = [value for value in values if value is not None]
            if not values:
                return None
            return torch.stack(
                [value.mean(-2, keepdim=True) for value in values], -1
            ).mean(-1)

        self.traversable = False
        super().__init__(
            self._midpoint_for(initial_mobs),
            color=mean(
                list(
                    traverse(
                        [mob.color for mob in initial_mobs if hasattr(mob, "color")]
                    )
                )
            ),
            **kwargs,
        )

        if self._link_children:
            self.add_children(initial_mobs)
        else:
            # Group slices are non-owning views: operations still recurse over
            # these children, but creating the view does not mutate parent
            # links. The bump is still owed -- ``super().__init__`` above writes
            # this Mob's location and color, and a recursive write caches the
            # descendant set it saw, which at that point was empty.
            self.children[:] = initial_mobs
            self._note_hierarchy_change()
        if (
            self._link_children
            and initial_mobs
            and all(mob.is_spawned() for mob in initial_mobs)
        ):
            self.spawn(animate=False)

    @property
    def mobs(self):
        """The Mobs in this Group -- an alias of
        :attr:`~algan.animatable_base.mob.Mob.children`.

        The live list, not a copy: mutating it changes the Group. Prefer
        :meth:`~algan.mobs.group.Group.add` and
        :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.remove_child`,
        which keep the Group's
        own location and parent links in step.
        """
        return self.children

    @staticmethod
    def _midpoint_for(mobs):
        if not mobs:
            return ORIGIN
        locations = list(
            traverse(
                [
                    [
                        descendant.location
                        for descendant in mob.get_descendants()
                        # Internal helper mobs (e.g. a bezier circuit's
                        # texture_points) sit at off-geometry locations;
                        # boundary queries already skip them, and including
                        # them here corrupts the midpoint.
                        if not descendant.exclude_from_boundary
                    ]
                    for mob in mobs
                ]
            )
        )
        return midpoint(locations) if locations else ORIGIN

    def get_mob_midpoint(self) -> torch.Tensor:
        """Get the middle of the Group's members' combined extent.

        The center of the box enclosing every member, which is where the Group
        places its own anchor. Internal helper geometry is excluded, so the answer
        matches what the viewer sees.

        Returns
        -------
        torch.Tensor
            The midpoint, shape ``(*, 1, 3)``, or ``ORIGIN`` for an empty Group.
        """
        return self._midpoint_for(self.children)

    def __getitem__(self, item):
        """Get a member by index, or a sub-Group by slice.

        ``group[0]`` is the member itself. A slice returns a **view**: a Group that
        recurses over the selected members without becoming their parent, so
        ``group[1:3].move(UP)`` moves those two members and nothing else. The view
        is not added to the scene and never needs spawning.

        Animation
        ---------
        Not animated: indexing only selects. Animate the result to move what it
        covers.

        Parameters
        ----------
        item
            Index of a single member, or a slice selecting several.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob` or
        :class:`~algan.mobs.group.Group`
            The member at an integer index, otherwise a non-owning Group view.
        """
        mobs = self.children[item]
        if isinstance(mobs, Mob):
            return mobs
        # Slicing is observational: the view is not an actor and does not add
        # itself as a parent of the selected children. Empty slices remain Group.
        return Group(
            *list(mobs),
            scene=self.scene,
            add_to_scene=False,
            _link_children=False,
        )

    def __setitem__(self, item, value):
        """Replace a member, or a slice of members, with other Mobs.

        The replaced Mobs are detached from the Group but stay in the scene; they are
        not despawned.

        Animation
        ---------
        Not animated: the membership change is immediate and affects only animation
        recorded from here on.

        Parameters
        ----------
        item
            Index or slice to replace.
        value
            Replacement Mob, or iterable of Mobs for a slice.
        """
        replacement = list(self.children)
        if isinstance(item, slice):
            replacement[item] = list(traverse((value,)))
        else:
            replacement[item] = value
        self.replace_children(replacement, link_parents=self._link_children)
        return self

    def __iter__(self):
        """Iterate over the Group's members, so ``for mob in group`` works.

        Returns
        -------
        Iterator[:class:`~algan.animatable_base.mob.Mob`]
            An iterator over the members, in order.
        """
        return iter(self.children)

    def __len__(self):
        """Get the number of members, so ``len(group)`` works.

        Returns
        -------
        int
            How many Mobs are in the Group.
        """
        return len(self.children)

    def add(self, *mobs):
        """Add Mobs to the Group.

        The added Mobs become children, so the Group's transforms carry them along,
        and the Group's anchor moves to the middle of the enlarged collection.
        Re-adding an existing member does nothing.

        Animation
        ---------
        Not animated: membership and the Group's re-centering both happen instantly
        (the re-centering is done inside ``Off()``, so it costs no video time and
        does not drag the existing members around).

        Parameters
        ----------
        *mobs
            Mobs to add. Nested iterables are flattened, so ``group.add([a, b])`` and
            ``group.add(a, b)`` are the same.

        Returns
        -------
        :class:`~algan.mobs.group.Group`
            This Group, so calls can be chained.

        Raises
        ------
        :class:`.HierarchyError`
            If a Mob is added twice, belongs to another Scene, or the change would
            create a cycle.
        """
        mobs = list(traverse(mobs))
        if not mobs:
            return self
        if self._link_children:
            self.add_children(mobs)
        else:
            # A non-owning view takes no parent links, but everything else has
            # to match ``add_children``: re-adding a member is a no-op rather
            # than a duplicate-child error, and the structure version has to be
            # bumped or ``get_descendants`` goes on serving the cache it filled
            # before the member arrived -- leaving the new member behind when
            # the view is moved, while ``len(view)`` says it is there.
            candidates = [
                mob
                for mob in mobs
                if not any(existing is mob for existing in self.children)
            ]
            if candidates:
                new_children = [*self.children, *candidates]
                self._validate_new_children(new_children)
                self.children[:] = new_children
                self._note_hierarchy_change()
        with Off(animation_manager=self.animation_manager):
            self.set_non_recursive(location=self.get_mob_midpoint())
        return self

    def get_parts_as_mobs(self):
        """Get the Group's members as a list.

        Returns
        -------
        list[:class:`~algan.animatable_base.mob.Mob`]
            The members. Unlike
            :meth:`~algan.animatable_base.mob.Mob.get_parts_as_mobs`,
            the Group itself is
            not included -- it carries no geometry of its own.
        """
        return self.mobs

    def get_boundary_edge_point2(self, direction: torch.Tensor) -> torch.Tensor:
        """Get the outermost point of any member along a direction.

        Parameters
        ----------
        direction
            Direction to search along, shape ``(*, 3)``.

        Returns
        -------
        torch.Tensor
            The extreme boundary point across all members, shape ``(*, 3)``.
        """
        points = torch.stack(
            [(m.get_boundary_edge_point(direction)) for m in self.mobs]
        )
        dots = dot_product(points, direction)
        furthest_ind = dots.argmax(0, keepdim=True)
        return broadcast_gather(points, 0, furthest_ind, keepdim=False)

    def arrange_in_line(
        self,
        direction: torch.Tensor = RIGHT,
        buffer: float | None = None,
        start_at_first: bool = False,
        equal_displacement: bool = False,
        alignment_direction: torch.Tensor | None = None,
    ):
        """Lay the members out in a line.

        Members are placed edge to edge with ``buffer`` between them, so differently
        sized Mobs still end up evenly gapped rather than evenly centred.

        Animation
        ---------
        Recorded as an animation: every member moves at once inside a
        :class:`~.Sync`, over the current context's duration (1 second by default).
        Call it before spawning to lay a Group out for free.

        Parameters
        ----------
        direction
            Direction the line runs in. Defaults to ``RIGHT``.
        buffer
            Gap between neighbouring members, in world units; ``0`` puts them edge to
            edge. Defaults to ``None``, meaning ``SETTINGS.style.buffer`` (``0.6``).
        start_at_first
            Whether to keep the first member where it is and build the line out from
            it. Defaults to False, which centers the line on the Group's location.
        equal_displacement
            Whether to space members by a constant pitch (that of the largest member)
            rather than by their own sizes. Defaults to False. True gives evenly
            spaced centers, which is what you want for a row of labelled cells.
        alignment_direction
            Direction along which to additionally align members, e.g. ``DOWN`` to sit
            them all on a shared baseline. Defaults to ``None``, meaning no secondary
            alignment.

        Returns
        -------
        :class:`~algan.mobs.group.Group`
            This Group, so calls can be chained.

        See Also
        --------
        :meth:`~algan.mobs.group.Group.arrange_in_grid`
            Lay members out in rows and columns.
        :meth:`~algan.mobs.group.Group.arrange_between_points`
            Space members evenly between two points.
        """
        if not self.children:
            return self
        if buffer is None:
            buffer = SETTINGS.style.buffer

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
        with Sync(animation_manager=self.animation_manager):
            for i, mob in enumerate(self.mobs):
                start = start + direction * (mob_sizes[i] / 2)
                location = start
                if alignment_direction is not None:
                    location = location + alignment_offsets[i] * alignment_direction
                # loc + (disp_to_center) = l
                mob.location = location + (mob.location - mob.get_center())
                start = start + direction * (mob_sizes[i] / 2 + buffer)
        return self

    def arrange_between_points(self, start: torch.Tensor, end: torch.Tensor):
        """Space the members evenly along the segment between two points.

        Members are placed at equal fractions of the way from ``start`` to ``end``,
        both endpoints excluded, so ``n`` members divide the segment into ``n + 1``
        equal steps. Sizes are ignored -- it is the centers that are evenly spaced.

        Animation
        ---------
        Recorded as an animation: every member moves at once inside a
        :class:`~.Sync`, over the current context's duration (1 second by default).

        Parameters
        ----------
        start
            Start of the segment, shape ``(*, 3)``.
        end
            End of the segment, shape ``(*, 3)``.

        Returns
        -------
        :class:`~algan.mobs.group.Group`
            This Group, so calls can be chained.
        """
        if not self.children:
            return self
        dif = end - start
        with Sync(animation_manager=self.animation_manager):
            for i, mob in enumerate(self.mobs):
                mob.location = start + dif * ((i + 1) / (len(self.mobs) + 1))
        return self

    def arrange_in_grid(
        self,
        num_rows: int = None,
        row_direction: torch.Tensor = RIGHT,
        column_direction: torch.Tensor = DOWN,
        buffer=None,
        column_buffer=None,
        tight_axis=None,
    ):
        """Lay the members out in a grid, filling row by row.

        Cells are sized to the largest member so the grid stays regular, and the whole
        grid is centered on the Group's location. Members fill along
        ``row_direction`` first, wrapping to the next line along ``column_direction``.

        Animation
        ---------
        Recorded as an animation: every member moves at once inside a
        :class:`~.Sync`, over the current context's duration (1 second by default).
        Call it before spawning to lay a Group out for free.

        Parameters
        ----------
        num_rows
            Number of rows; columns follow from the member count. Defaults to
            ``None``, meaning ``ceil(sqrt(len(mobs)))`` -- as square a grid as the
            count allows.
        row_direction
            Direction along which a row runs. Defaults to ``RIGHT``.
        column_direction
            Direction in which successive rows are stacked. Defaults to ``DOWN``.
        buffer
            Gap between members within a row, in world units. Defaults to ``None``,
            meaning ``SETTINGS.style.buffer`` (``0.6``).
        column_buffer
            Gap between rows, in world units. Defaults to ``None``, meaning use
            ``buffer``.
        tight_axis
            Which axis sizes its cells per row/column rather than uniformly across
            the whole grid: ``0`` for columns, ``1`` for rows. Defaults to ``None``,
            meaning every cell is the same size. Use it to close up the gaps when
            members vary a lot in size.

        Returns
        -------
        :class:`~algan.mobs.group.Group`
            This Group, so calls can be chained.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If ``num_rows`` is not a positive integer.
        ValueError
            If ``tight_axis`` is not ``0``, ``1`` or ``None``.

        Examples
        --------

        Arrange mobs in a 3x3 grid slanted at a 45 degrees angle.

        .. algan:: Example1ArrangeInGrid

            from algan import *

            group = Group([Square() for _ in range(9)]).scale(1/3).arrange_in_grid(3, RIGHT+UP, RIGHT+DOWN).spawn()
            group.rotate(90, OUT)

            Scene.save_video()

        """
        if not self.children:
            return self
        if buffer is None:
            buffer = SETTINGS.style.buffer
        if column_buffer is None:
            column_buffer = buffer
        if num_rows is None:
            num_rows = max(1, math.ceil(math.sqrt(len(self.children))))
        if not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows <= 0:
            raise AlganConfigurationError("num_rows must be a positive integer")
        num_cols = len(self.children) // num_rows
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
                        self.mobs[i + j * num_cols].get_length_in_direction(
                            row_direction
                        )
                        for j in range(num_rows)
                        if i + j * num_cols < len(self.mobs)
                    )
                    + buffer
                    for i in range(num_cols)
                ]
            elif tight_axis == 1:
                buf_dist2 = [
                    max(
                        self.mobs[i + j * num_cols].get_length_in_direction(
                            column_direction
                        )
                        for i in range(num_cols)
                        if i + j * num_cols < len(self.mobs)
                    )
                    + column_buffer
                    for j in range(num_rows)
                    if j * num_cols < len(self.mobs)
                ]
            else:
                raise ValueError("tight_axis must be 0, 1, or None")

        start = self.location - (
            row_direction * sum(buf_dist1) * 0.5
            + column_direction * sum(buf_dist2) * 0.5
        )
        with Sync(animation_manager=self.animation_manager):
            for i, mob in enumerate(self.mobs):
                x = i % num_cols
                y = i // num_cols
                x_dist = sum(buf_dist1[:x]) + buf_dist1[x] * 0.5
                y_dist = sum(buf_dist2[:y]) + buf_dist2[y] * 0.5
                mob.location = (
                    start + row_direction * x_dist + column_direction * y_dist
                ) + (mob.location - mob.get_center())
        return self
