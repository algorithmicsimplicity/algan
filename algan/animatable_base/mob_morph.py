"""Morphing (``become``) and batch-expansion machinery for :class:`~algan.animatable_base.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobMorphMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


def linear_sum_assignment(*args, **kwargs):
    """scipy.optimize.linear_sum_assignment, imported on first call
    (deferred: scipy costs ~0.4 s of ``import algan``).
    """
    from scipy.optimize import linear_sum_assignment as _lsa

    return _lsa(*args, **kwargs)


from algan.animation_timeline.animation_contexts import Off, Seq, Sync
from algan.animation_timeline.timeline import bump_hierarchy_version
from algan.utils.tensor_utils import cast_to_tensor, mid_point, squish, unsquish

if TYPE_CHECKING:
    from algan.animatable_base.mob import Mob


class MobMorphMixin:
    """``become`` morphing plus the ``expand_n_*`` batch-expansion helpers it
    uses to match source and target sub-mob counts.
    """

    def expand_n_list(self, lst, n: int) -> list:
        """Internal: pad a list of point tensors by duplicating entries.

        Used by
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` to give two
        Mobs the same number of paths
        before morphing. Duplicates are degenerate (collapsed onto their last
        point), so the padding is invisible.

        Parameters
        ----------
        lst
            Point tensors to pad.
        n
            How many entries to add.

        Returns
        -------
        list
            A list of ``len(lst) + n`` tensors.
        """
        current_children_count = len(lst)
        target_children_count = current_children_count + n
        # Determine how many times each existing child needs to be repeated/cloned
        repeat_indices = (
            torch.arange(target_children_count) * current_children_count
        ) // target_children_count
        split_factors = [
            (repeat_indices == i).sum() for i in range(current_children_count)
        ]

        new_submobs = []
        for submob, factor in zip(lst, split_factors):
            new_submobs.append(submob)  # Add the original child
            for _ in range(1, factor):
                new_submobs.append(
                    submob[-1, -1:, :].expand(
                        torch.Size(
                            [
                                *([-1 for _ in range(submob.dim() - 3)]),
                                submob.shape[-3],
                                self.num_points_per_object,
                                -1,
                            ]
                        )
                    )
                )
        return new_submobs

    def expand_n_children(self, n: int) -> Mob:
        """Internal: add ``n`` children by cloning the existing ones.

        Used by
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` so that a Mob
        with three parts can morph into
        one with five: the shortfall is made up with clones, which then morph
        into the extra target parts.

        Animation
        ---------
        Not animated. Structural, and only meaningful on a Mob with fresh history
        (see
        :meth:`~algan.animatable_base.mob.Mob.detach_history`).

        Parameters
        ----------
        n
            How many children to add.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        current_children_count = len(self.get_non_component_children())
        target_children_count = current_children_count + n
        # Determine how many times each existing child needs to be repeated/cloned
        repeat_indices = (
            torch.arange(target_children_count) * current_children_count
        ) // target_children_count
        split_factors = [
            (repeat_indices == i).sum() for i in range(current_children_count)
        ]

        new_submobs = []
        for submob, factor in zip(self.get_non_component_children(), split_factors):
            new_submobs.append(submob)  # Add the original child
            for _ in range(1, factor):
                new_submobs.append(submob.clone())  # Add clones
        self.children = new_submobs + [_ for _ in self.children if _ in self.components]
        bump_hierarchy_version()
        return self

    def expand_n_tensor(self, value: torch.Tensor, n: int) -> torch.Tensor:
        """Internal: pad a batched point tensor by duplicating sub-objects.

        Used by
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` to align the
        segment counts of two curves.
        Added entries are degenerate (collapsed onto their last point).

        Parameters
        ----------
        value
            Batched point data to pad.
        n
            How many sub-objects to add.

        Returns
        -------
        torch.Tensor
            The padded tensor.
        """
        current_batch_size = value.shape[-3]
        target_batch_size = current_batch_size + n
        if value.shape[-3] == 1:
            # Already a singleton batch, no per-element expansion needed.
            return value.expand(target_batch_size, -1, -1)

        # Determine how many times each existing batch element needs to be repeated
        repeat_indices = (
            torch.arange(target_batch_size) * current_batch_size
        ) // target_batch_size
        split_factors = [(repeat_indices == i).sum() for i in range(current_batch_size)]

        new_batched_values = []
        for sub_object_data, factor in zip(value, split_factors):
            new_batched_values.append(sub_object_data)  # Add original sub-object data
            for _ in range(1, factor):
                # Clone the last point of the sub-object data to expand
                new_batched_values.append(
                    sub_object_data[..., -1:, :].expand(
                        torch.Size(
                            [
                                *([-1 for _ in range(sub_object_data.dim() - 2)]),
                                self.num_points_per_object,
                                -1,
                            ]
                        )
                    )
                )
        return torch.stack(new_batched_values, -3)

    def expand_n_batch(self, n: int) -> Mob:
        """Internal: grow this Mob's batch by ``n`` objects, cloning to fill.

        Used by
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` to match the
        number of primitives on both
        sides of a morph. Every animatable attribute is re-batched together, so
        colour and opacity keep lining up with the new geometry.

        Animation
        ---------
        Not animated, and **not recorded**: the writes go through
        ``setattr_and_rebatch_without_record``, which re-allocates this Mob's
        timeline rows. Recorded history stays with the old rows, so this is only
        valid on a Mob with fresh history (see
        :meth:`~algan.animatable_base.mob.Mob.detach_history`).

        Parameters
        ----------
        n
            How many batch objects to add.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        RuntimeError
            If the Mob's batch-boundary metadata does not describe its current
            batch, which would leave views and render primitives disagreeing.
        """
        # Current number of logical objects in the batch (points / points_per_object)
        current_batch_size = self.location.shape[-2] // self.num_points_per_object
        target_batch_size = current_batch_size + n
        # Determine how many times each existing batch element needs to be repeated
        repeat_indices = (
            torch.arange(target_batch_size) * current_batch_size
        ) // target_batch_size
        split_factors = [(repeat_indices == i).sum() for i in range(current_batch_size)]

        # Keep the repeat mapping so non-animatable batch-boundary metadata can
        # be expanded alongside the timeline-backed attributes below.
        repeat_indices = repeat_indices.to(torch.long)

        # Iterate over animatable attributes and expand their batch dimensions
        for attr in self.animatable_attrs:
            if not hasattr(self, attr):
                # Attr has no rows in the global attribute timeline yet.
                continue
            value = cast_to_tensor(self.__getattribute__(attr))[
                0
            ]  # Get the current value (first time step)
            if (
                value.shape[-2] == 1
            ):  # If already a singleton batch, no expansion needed
                continue

            # Unsquish to separate individual objects in the batch if needed
            value_per_object = unsquish(value, -2, self.num_points_per_object)
            new_batched_values = []
            for sub_object_data, factor in zip(value_per_object, split_factors):
                new_batched_values.append(
                    sub_object_data
                )  # Add original sub-object data
                for _ in range(1, factor):
                    # Clone the last point of the sub-object data to expand
                    new_batched_values.append(
                        sub_object_data[..., -1:, :].expand(
                            torch.Size(
                                [
                                    *([-1 for _ in range(sub_object_data.dim() - 2)]),
                                    self.num_points_per_object,
                                    -1,
                                ]
                            )
                        )
                    )
            # Stack the new batched values and write them back to the global
            # attribute timeline, re-allocating this mob's rows for the new size.
            self.setattr_and_rebatch_without_record(
                attr, squish(torch.stack(new_batched_values, -3), -3, -2).unsqueeze(0)
            )

        # ``parent_batch_sizes`` is structural metadata rather than an
        # animatable attribute, so setattr/rebatch cannot update it for us.
        # Leaving it at the old size makes indexed views and render primitives
        # disagree with the newly-expanded tensors.
        if self.parent_batch_sizes is not None:
            parent_batch_sizes = self.parent_batch_sizes
            points_per_object = self.num_points_per_object
            if self.singleton_batch_indexing and len(parent_batch_sizes) == 1:
                # A singleton wrapper stores the total number of child objects
                # in its sole entry (BezierCircuitCubic.from_batches uses this
                # for the packed character batch).
                self.parent_batch_sizes = torch.tensor(
                    (target_batch_size * points_per_object,),
                    dtype=parent_batch_sizes.dtype,
                    device=parent_batch_sizes.device,
                )
            else:
                objects_per_parent = parent_batch_sizes // points_per_object
                if (
                    bool((parent_batch_sizes % points_per_object != 0).any())
                    or int(objects_per_parent.sum()) != current_batch_size
                ):
                    raise RuntimeError(
                        "parent_batch_sizes does not describe the Mob's current batch"
                    )

                repeat_indices_on_device = repeat_indices.to(parent_batch_sizes.device)
                if bool((objects_per_parent == 1).all()):
                    # Every batch object is independently indexable. Repeating
                    # an object therefore repeats its parent entry as well.
                    self.parent_batch_sizes = parent_batch_sizes.index_select(
                        0, repeat_indices_on_device
                    )
                else:
                    # Multiple batch objects belong to each parent. Preserve
                    # those parent groups while counting repeated objects into
                    # the group they came from.
                    parent_of_object = torch.repeat_interleave(
                        torch.arange(
                            len(parent_batch_sizes),
                            device=parent_batch_sizes.device,
                        ),
                        objects_per_parent,
                    )
                    expanded_parent_of_object = parent_of_object.index_select(
                        0, repeat_indices_on_device
                    )
                    self.parent_batch_sizes = (
                        torch.bincount(
                            expanded_parent_of_object,
                            minlength=len(parent_batch_sizes),
                        )
                        * points_per_object
                    ).to(parent_batch_sizes.dtype)
        return self

    def reorder_batch_to_minimize_movement(self, target: Mob) -> Mob:
        """Re-pair this Mob's parts with the nearest parts of another Mob.

        Objects are matched by optimal assignment on their centers, so each part
        morphs into the *closest* target part rather than the one that happens to
        share its index. For text and other heavily batched Mobs this is what turns
        a morph from glyph fragments flying across the screen into a smooth local
        deformation. The assignment is costly for Mobs with very many parts.

        This Mob and ``target`` must already have the same batch size (e.g. after
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.expand_n_batch`);
        mismatched or single-object batches are left
        untouched.

        Animation
        ---------
        Not animated, and **not recorded** -- it re-batches attributes in place, so
        it is only valid on a Mob with fresh history.
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` calls it
        for you when ``minimize_movement=True``.

        Parameters
        ----------
        target
            The Mob whose parts this Mob's parts are matched against.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        num_points_per_object = self.num_points_per_object
        my_points = unsquish(
            cast_to_tensor(self.location)[0], -2, num_points_per_object
        )  # [num_objects, num_points_per_object, 3]
        target_points = unsquish(
            cast_to_tensor(target.location)[0], -2, num_points_per_object
        )
        num_objects = my_points.shape[-3]
        # Nothing to reorder for a single (or mismatched) object batch.
        if num_objects <= 1 or target_points.shape[-3] != num_objects:
            return self

        # Pair objects by their centers using optimal (minimum total distance) assignment.
        my_centers = my_points.mean(-2)
        target_centers = target_points.mean(-2)
        distance_matrix = torch.cdist(target_centers, my_centers)
        target_inds, my_inds = linear_sum_assignment(distance_matrix.cpu().numpy())
        # Build the permutation that sends my object `my_inds[k]` to slot `target_inds[k]`,
        # so that afterwards self's k-th object is the one matched to target's k-th object.
        permutation = torch.empty(num_objects, dtype=torch.long)
        permutation[torch.as_tensor(target_inds, dtype=torch.long)] = torch.as_tensor(
            my_inds, dtype=torch.long
        )
        permutation = permutation.to(my_points.device)

        # Apply the same object permutation to every (non-broadcast) batched attribute so
        # all of this Mob's data stays consistent.
        for attr in self.animatable_attrs:
            if not hasattr(self, attr):
                # Attr has no rows in the global attribute timeline yet.
                continue
            value = cast_to_tensor(self.__getattribute__(attr))[0]
            if (value.shape[-2] == 1) or (value.shape[-2] % num_points_per_object != 0):
                continue
            value_per_object = unsquish(value, -2, num_points_per_object)
            if value_per_object.shape[-3] != num_objects:
                continue
            value_per_object = value_per_object.index_select(-3, permutation)
            self.setattr_and_rebatch_without_record(
                attr, squish(value_per_object, -3, -2).unsqueeze(0)
            )
        return self

    def get_non_component_children(self) -> list[Mob]:
        """Get the children you added, excluding the Mob's own structural parts.

        A shape's components are the pieces it builds itself from; this returns
        only the Mobs added on top of them, which is the set
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become`
        pairs up when morphing.

        Returns
        -------
        list[:class:`~algan.animatable_base.mob.Mob`]
            Direct children that are not components of this Mob.
        """
        return [_ for _ in self.children if _ not in self.components]

    def become(
        self,
        other_mob: Mob,
        detach_history: bool = True,
        minimize_movement=False,
    ) -> Mob:
        """Morph this Mob into another one.

        Location, colour, opacity, orientation and geometry all animate across to
        match ``other_mob``. Where the two Mobs are made of different numbers of
        parts, the smaller side is padded with degenerate copies first, so a
        three-letter word can morph smoothly into a five-letter one.

        Behaves like Manim's ``Transform``: this Mob ends up wearing
        ``other_mob``'s appearance and is the single Mob left in the scene.
        ``other_mob`` is only read for its target values -- it is never spawned, and
        the caller's copy is not mutated.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). The structural padding happens instantly inside ``Off()`` before
        the morph, so the viewer sees only the transformation.

        Parameters
        ----------
        other_mob
            The Mob to morph into. It must be built from the same primitive type
            as this Mob (same ``num_points_per_object`` and the same number of
            components).
        detach_history
            Whether to start this Mob on a fresh animation history first, via
            :meth:`~algan.animatable_base.mob.Mob.detach_history`.
            Defaults to True, which is what allows a
            morph that changes the number of parts. Pass False only for a morph
            recorded inside another morph (``become`` uses it that way for
            children).
        minimize_movement
            Whether to pair each part with the *closest* part of ``other_mob`` by
            optimal assignment, rather than in index order. Defaults to False,
            matching Manim; True gives a smooth local deformation instead of parts
            flying across the screen, at the cost of an assignment solve that is
            expensive for Mobs with very many parts.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            The morphed Mob, wearing ``other_mob``'s appearance. When
            ``detach_history`` is True this is a **different object** from the one
            you called the method on, so use the returned Mob for any later
            animation.

        Raises
        ------
        NotImplementedError
            If the two Mobs are built from different primitive types, e.g. morphing
            a triangle mesh into a bezier circuit.

        Examples
        --------
        .. algan:: Example1MobBecome

            from algan import *

            square = Square().spawn()
            square = square.become(Circle(color=RED))

            Scene.save_video()
        """
        if (other_mob.num_points_per_object != self.num_points_per_object) or (
            len(other_mob.components) != len(self.components)
        ):
            raise NotImplementedError(
                "You are trying to change an object of one primitive type (e.g., triangle) "
                "to another type (e.g., cubic bezier circuit). This is not supported. "
                "When using become(), the target mob must be of the same primitive type as the original."
            )

        with Off(animation_manager=self.animation_manager):
            new_self = self
            if detach_history:
                # Detach this mob's history so that the (potentially shape-changing)
                # transformation is recorded in a fresh, internally-consistent history.
                # detach_history() freezes and despawns the current mob and returns a
                # spawned clone that takes its place seamlessly.
                new_self = self.detach_history()
                # Clone the target purely as a source of data to morph towards, so we
                # never mutate (or spawn) the mob the caller passed in.
                other_mob = other_mob.clone(add_to_scene=False)

            # Adjust child counts to match for smooth transitions
            child_difference = len(other_mob.get_non_component_children()) - len(
                new_self.get_non_component_children()
            )
            if child_difference > 0:
                new_self.expand_n_children(child_difference)
            elif child_difference < 0:
                other_mob.expand_n_children(-child_difference)

        my_children = new_self.get_non_component_children()
        other_children = other_mob.get_non_component_children()
        with Seq(animation_manager=self.animation_manager):
            with Sync(animation_manager=self.animation_manager):
                if len(new_self.get_non_component_children()) > 0:
                    # Recursively apply 'become' to children to handle nested transformations
                    if minimize_movement:
                        child_locs = torch.stack(
                            [mid_point(c.location, -2).squeeze() for c in my_children]
                        )
                        other_child_locs = torch.stack(
                            [
                                mid_point(c.location, -2).squeeze()
                                for c in other_children
                            ]
                        )
                        distance_matrix = torch.cdist(child_locs, other_child_locs)
                        row_ind, col_ind = linear_sum_assignment(
                            distance_matrix.cpu().numpy()
                        )

                        for i, j in zip(row_ind, col_ind):
                            my_children[i].become(
                                other_children[j],
                                detach_history=False,
                                minimize_movement=minimize_movement,
                            )
                    else:
                        for my_child, other_child in zip(my_children, other_children):
                            my_child.become(
                                other_child,
                                detach_history=False,
                                minimize_movement=minimize_movement,
                            )  # Children do not detach their history
                for my_component, other_component in zip(
                    new_self.components, other_mob.components
                ):
                    my_component.become(
                        other_component,
                        detach_history=False,
                        minimize_movement=minimize_movement,
                    )

                # Adjust batch size (number of points per object) for smooth transitions
                if new_self.num_points_per_object == 4:

                    def get_sub_circuits(x):
                        """Split a [segments, 4, xyz] tensor at path breaks."""
                        start_inds = (
                            (
                                (x[..., 0, :] - x.roll(1, -3)[..., -1, :]).abs().sum(-1)
                                > 1e-6
                            )
                            .nonzero(as_tuple=False)
                            .flatten()
                        )
                        if start_inds.numel() == 0:
                            return [x]
                        return [
                            x[
                                start_inds[i] : start_inds[i + 1]
                                if (i + 1) < len(start_inds)
                                else x.shape[-3]
                            ]
                            for i in range(len(start_inds))
                        ]

                    def get_parent_circuits(mob):
                        """Split cubic segments using packed-object boundaries."""
                        segments = unsquish(mob.location, -2, 4).squeeze(0)
                        parent_batch_sizes = mob.parent_batch_sizes
                        if parent_batch_sizes is None:
                            return [segments]
                        if (
                            bool((parent_batch_sizes % 4 != 0).any())
                            or int(parent_batch_sizes.sum()) != mob.location.shape[-2]
                        ):
                            raise RuntimeError(
                                "parent_batch_sizes does not match cubic control points"
                            )
                        return list(
                            segments.split((parent_batch_sizes // 4).tolist(), dim=-3)
                        )

                    had_parent_batches = (
                        new_self.parent_batch_sizes is not None
                        or other_mob.parent_batch_sizes is not None
                    )
                    my_parent_circuits = get_parent_circuits(new_self)
                    other_parent_circuits = get_parent_circuits(other_mob)

                    # Packed text stores one parent batch per glyph. Equalize
                    # those parent batches first, then equalize the disconnected
                    # paths and cubic segments *within each glyph*. Flattening
                    # all paths globally loses the glyph boundaries required to
                    # pair control points with per-glyph transforms at render time.
                    parent_difference = len(other_parent_circuits) - len(
                        my_parent_circuits
                    )
                    if parent_difference > 0:
                        my_parent_circuits = new_self.expand_n_list(
                            my_parent_circuits, parent_difference
                        )
                    elif parent_difference < 0:
                        other_parent_circuits = other_mob.expand_n_list(
                            other_parent_circuits, -parent_difference
                        )

                    my_parent_batches = []
                    other_parent_batches = []
                    parent_batch_sizes = []
                    for my_parent, other_parent in zip(
                        my_parent_circuits, other_parent_circuits
                    ):
                        my_control_points = get_sub_circuits(my_parent)
                        other_control_points = get_sub_circuits(other_parent)

                        circuit_difference = len(other_control_points) - len(
                            my_control_points
                        )
                        if circuit_difference > 0:
                            my_control_points = new_self.expand_n_list(
                                my_control_points, circuit_difference
                            )
                        elif circuit_difference < 0:
                            other_control_points = other_mob.expand_n_list(
                                other_control_points, -circuit_difference
                            )

                        my_cs = []
                        other_cs = []
                        for my_c, other_c in zip(
                            my_control_points, other_control_points
                        ):
                            segment_difference = other_c.shape[-3] - my_c.shape[-3]
                            if segment_difference > 0:
                                my_c = new_self.expand_n_tensor(
                                    my_c, segment_difference
                                )
                            elif segment_difference < 0:
                                other_c = other_mob.expand_n_tensor(
                                    other_c, -segment_difference
                                )
                            my_cs.append(my_c)
                            other_cs.append(other_c)

                        my_parent_batch = torch.cat(my_cs, -3)
                        other_parent_batch = torch.cat(other_cs, -3)
                        my_parent_batches.append(my_parent_batch)
                        other_parent_batches.append(other_parent_batch)
                        parent_batch_sizes.append(my_parent_batch.shape[-3] * 4)

                    # The cubic alignment above can insert degenerate segments.  Any
                    # non-singleton point-wise attributes (for example opacity after
                    # ``wave_color``) must be structurally expanded at the same time.
                    # Otherwise location owns the new number of timeline rows while
                    # color/opacity retain rows for the pre-morph geometry.
                    aligned_segment_count = sum(parent_batch_sizes) // 4
                    my_segment_count = new_self.location.shape[-2] // 4
                    other_segment_count = other_mob.location.shape[-2] // 4
                    if aligned_segment_count > my_segment_count:
                        new_self.expand_n_batch(
                            aligned_segment_count - my_segment_count
                        )
                    if aligned_segment_count > other_segment_count:
                        other_mob.expand_n_batch(
                            aligned_segment_count - other_segment_count
                        )

                    new_self.setattr_and_rebatch_without_record(
                        "location",
                        squish(torch.cat(my_parent_batches, -3), -3, -2).unsqueeze(0),
                    )
                    other_mob.setattr_and_rebatch_without_record(
                        "location",
                        squish(torch.cat(other_parent_batches, -3), -3, -2).unsqueeze(
                            0
                        ),
                    )
                    if had_parent_batches:
                        metadata_device = (
                            new_self.parent_batch_sizes.device
                            if new_self.parent_batch_sizes is not None
                            else other_mob.parent_batch_sizes.device
                        )
                        metadata_dtype = (
                            new_self.parent_batch_sizes.dtype
                            if new_self.parent_batch_sizes is not None
                            else other_mob.parent_batch_sizes.dtype
                        )
                        expanded_parent_batch_sizes = torch.tensor(
                            parent_batch_sizes,
                            dtype=metadata_dtype,
                            device=metadata_device,
                        )
                        new_self.parent_batch_sizes = expanded_parent_batch_sizes
                        other_mob.parent_batch_sizes = (
                            expanded_parent_batch_sizes.clone()
                        )
                else:
                    batch_difference = (
                        other_mob.location.shape[-2] - new_self.location.shape[-2]
                    ) // new_self.num_points_per_object
                    if batch_difference > 0:
                        new_self.expand_n_batch(batch_difference)
                    elif batch_difference < 0:
                        other_mob.expand_n_batch(-batch_difference)

                    if minimize_movement:
                        # Re-pair the (now equally-sized) batches so each source object
                        # morphs into the closest target object, minimizing total movement.
                        other_mob.reorder_batch_to_minimize_movement(new_self)

                # Set all animatable attributes non-recursively to match the target mob's values
                for attr_name in new_self.animatable_attrs:
                    if not hasattr(new_self, attr_name) or not hasattr(
                        other_mob, attr_name
                    ):
                        continue
                    # Use getattr to safely access attributes, as not all mobs may have all listed attributes
                    new_self.set_non_recursive(
                        **{attr_name: getattr(other_mob, attr_name)}
                    )

            # Like Manim's Transform, the (single) transforming mob is left in the
            # scene holding the target's appearance; the target mob itself is never
            # added to the scene. We return the transformed mob so that it can be
            # further animated / transformed by the caller.
            return new_self
