"""Parent/child hierarchy management for :class:`~algan.animatable_base.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobHierarchyMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from algan.animatable_base.animatable import Animatable
from algan.animation_timeline.timeline import (
    HIERARCHY_VERSION,
    _opt_disabled,
    bump_hierarchy_version,
)
from algan.errors import HierarchyError
from algan.utils.python_utils import traverse

if TYPE_CHECKING:
    from algan.animatable_base.mob import Mob


class MobHierarchyMixin:
    """Parent, child and descendant management, mixed into
    :class:`~algan.animatable_base.mob.Mob`.

    A Mob's children follow its transforms: move, rotate, scale or recolour a
    parent and the change propagates down. Use
    :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_children` to
    build a hierarchy, or a :class:`~algan.mobs.group.Group` when you just want to
    handle several Mobs
    as one.

    The hierarchy is a graph, not a tree: a Mob may have several parents and
    then accumulates all of their changes, which is what lets two overlapping
    Groups each arrange the same member. It is read when an animation is
    *recorded*, not when it plays, so re-parenting between two recorded
    animations leaves the first one alone -- see :doc:`/new_user_tutorials/child_mobs`.
    """

    def _link_parent(self, other_mob: Mob) -> None:
        """Record the upward half of a parent/child link, and nothing else.

        The downward half, validation, ``anchor_priority`` and the version bump
        belong to whoever calls this. Both public entry points --
        :meth:`add_parent` and :meth:`add_children` -- go through one of those,
        so a link the public API creates always has both halves.
        """
        if not any(parent is other_mob for parent in self.parents):
            self.parents.append(other_mob)

    def _unlink_parent(self, other_mob: Mob) -> None:
        """Drop the upward half of a parent/child link, and nothing else."""
        self.parents[:] = [parent for parent in self.parents if parent is not other_mob]

    def _drop_child(self, mob: Mob) -> bool:
        """Detach ``mob`` from this Mob, both halves, if it is a child here.

        Returns whether anything was detached, which is what lets
        :meth:`remove_child` raise on a non-child while :meth:`remove_parent`
        stays silent.
        """
        for index, child in enumerate(self.children):
            if child is mob:
                del self.children[index]
                child._unlink_parent(self)
                self.anchor_priority = max(
                    (1 + item.anchor_priority for item in self.children),
                    default=0,
                )
                bump_hierarchy_version()
                return True
        return False

    def add_parent(self, other_mob: Mob) -> Mob:
        """Attach this Mob to another as one of its children.

        The mirror image of
        :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_children`
        called from the child's side: ``child.add_parent(parent)`` and
        ``parent.add_children(child)`` build the same link, so this Mob follows
        ``other_mob``'s transforms from here on. A Mob may have several parents,
        and then accumulates every one of their changes.
        Re-adding an existing parent does nothing.

        Animation
        ---------
        Not animated: the hierarchy changes immediately, and only affects
        animations recorded from here on.

        Parameters
        ----------
        other_mob
            The Mob to become a parent of this one.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        TypeError
            If ``other_mob`` is not an
            :class:`~algan.animatable_base.animatable.Animatable`.
        :class:`.HierarchyError`
            If ``other_mob`` is this Mob, belongs to another Scene, or is
            already somewhere below it -- any of which would make the graph a
            cycle.
        """
        if not isinstance(other_mob, Animatable):
            raise TypeError(
                f"A parent must be an Animatable instance, "
                f"got {type(other_mob).__name__}"
            )
        if other_mob is self:
            raise HierarchyError("A Mob cannot be its own parent")
        # Checked here as well as inside ``add_children`` so the message is
        # phrased from the side the caller is standing on. ``Group`` has always
        # rejected a Mob that is its own child or listed twice; this side
        # accepted both a self-parent and a two-Mob loop in silence, leaving a
        # graph nothing can traverse.
        if self._contains_in_hierarchy(self, other_mob):
            raise HierarchyError(
                f"{type(other_mob).__name__} is already below "
                f"{type(self).__name__}, so making it the parent "
                f"would create a cycle"
            )
        other_mob.add_children(self)
        return self

    def remove_parent(self, other_mob: Mob) -> Mob:
        """Detach this Mob from one of its parents, so it stops following that
        Mob's transforms.

        The reverse of
        :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.add_parent`,
        and drops both halves of the link -- this Mob leaves ``other_mob``'s
        children as well. Any *other* parents it has are untouched, and it stays
        in the scene with its own state.
        Removing a Mob that is not a parent does nothing.

        Animation
        ---------
        Not animated: the hierarchy changes immediately, and only affects
        animations recorded from here on.

        Parameters
        ----------
        other_mob
            The Mob to stop treating as a parent.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        if not other_mob._drop_child(self):
            # Reached for a link recorded on one side only, which the public
            # API no longer creates but ``replace_children(link_parents=False)``
            # still can. Clear whatever half is actually there.
            self._unlink_parent(other_mob)
        return self

    def get_children(
        self, generation: int = 0, include_components: bool = True
    ) -> list[Mob]:
        """Get this Mob's children, optionally reaching further down the hierarchy.

        Parameters
        ----------
        generation
            How many levels down to collect: ``0`` for direct children, ``1`` for
            grandchildren, and so on -- each level replaces the one above rather
            than adding to it. Defaults to ``0``.
        include_components
            Whether to include children that are structural components of the Mob
            (the parts a shape builds itself from) as opposed to Mobs you added.
            Defaults to True.

        Returns
        -------
        list[:class:`~algan.animatable_base.mob.Mob`]
            The children at that generation. The Mobs are live, not copies.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_hierarchy.MobHierarchyMixin.get_descendants`
            Every level at once, flattened.
        """
        children = self.children
        if not include_components:
            children = [_ for _ in children if _ not in self.components]
        if generation <= 0:
            return children
        children = [_.get_children(generation - 1) for _ in children]
        return [x for level_children in children for x in level_children]

    def get_descendants(self, include_self: bool = True) -> list[Mob]:
        """Get every Mob at or below this one, flattened into one list.

        Parameters
        ----------
        include_self
            Whether this Mob is the first element of the list. Defaults to True.

        Returns
        -------
        list[:class:`~algan.animatable_base.mob.Mob`]
            This Mob (unless excluded) followed by its children, their children,
            and so on. The Mobs are live, not copies.
        """
        cache = getattr(self, "_descendants_cache", None)
        if (
            cache is not None
            and cache[0] == HIERARCHY_VERSION[0]
            and not _opt_disabled("desccache")
        ):
            descendants = cache[1]
        else:
            descendants = list(
                traverse(
                    [
                        self,
                        [c.get_descendants() for c in self.children]
                        if hasattr(self, "children")
                        else [],
                    ]
                )
            )
            object.__setattr__(
                self, "_descendants_cache", (HIERARCHY_VERSION[0], descendants)
            )
        return list(descendants) if include_self else descendants[1:]

    @staticmethod
    def _contains_in_hierarchy(root, target):
        stack = [root]
        visited = set()
        while stack:
            current = stack.pop()
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            if current is target:
                return True
            stack.extend(getattr(current, "children", ()))
        return False

    def _validate_new_children(self, mobs):
        seen = set()
        for mob in mobs:
            if not isinstance(mob, Animatable):
                raise TypeError(
                    f"Children must be Animatable instances, got {type(mob).__name__}"
                )
            identity = id(mob)
            if identity in seen:
                raise HierarchyError("A child cannot occur more than once")
            seen.add(identity)
            if mob is self:
                raise HierarchyError("A Mob cannot be its own child")
            if mob.scene is not self.scene:
                raise HierarchyError("A Mob hierarchy cannot span multiple Scenes")
            if self._contains_in_hierarchy(mob, self):
                raise HierarchyError("This hierarchy mutation would create a cycle")

    def replace_children(self, mobs, *, link_parents: bool = True) -> Mob:
        """Swap out this Mob's children for a different set.

        Children that are not in the new set have their link to this Mob dropped,
        so they stop following its transforms; they are not despawned and remain
        in the scene on their own.

        Animation
        ---------
        Not animated: the hierarchy changes immediately, and only affects
        animations recorded from here on. Already-recorded animation is unchanged.

        Parameters
        ----------
        mobs
            The new children. Nested iterables are flattened, so a list of Groups
            is accepted.
        link_parents
            Whether to maintain the children's upward links to this Mob. Defaults
            to True; pass False only when the caller manages those links itself.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        TypeError
            If any item is not an
            :class:`~algan.animatable_base.animatable.Animatable`.
        :class:`~algan.errors.HierarchyError`
            If a Mob appears twice, is its own child, belongs to another Scene, or
            the change would create a cycle.
        """
        new_children = list(traverse(mobs))
        self._validate_new_children(new_children)
        old_children = list(self.children)

        for child in old_children:
            if link_parents and not any(child is item for item in new_children):
                # The half-link helpers, not remove_parent/add_parent: the
                # child list is rebuilt wholesale just below, so the downward
                # half must not be edited (or version-bumped) child by child.
                child._unlink_parent(self)

        self.children[:] = new_children
        if link_parents:
            for child in new_children:
                child._link_parent(self)
        self.anchor_priority = max(
            (1 + child.anchor_priority for child in new_children),
            default=0,
        )
        bump_hierarchy_version()
        return self

    def add_children(self, *mobs) -> Mob:
        """Attach Mobs as children, so they follow this Mob's transforms.

        Once attached, moving, rotating, scaling or recolouring this Mob carries
        the children along. Adding a Mob that is already a child does nothing.

        Animation
        ---------
        Not animated: the hierarchy changes immediately, and only affects
        animations recorded from here on. Children keep their own spawn state --
        attaching an unspawned Mob does not spawn it.

        Parameters
        ----------
        *mobs
            Mobs to attach. Nested iterables are flattened, so
            ``mob.add_children([a, b])`` and ``mob.add_children(a, b)`` are the
            same.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        TypeError
            If any item is not an
            :class:`~algan.animatable_base.animatable.Animatable`.
        :class:`~algan.errors.HierarchyError`
            If a Mob appears twice, is its own child, belongs to another Scene, or
            the change would create a cycle.
        """
        candidates = list(traverse(mobs))
        # Re-adding an existing child is idempotent, matching Group.add.
        candidates = [
            mob
            for mob in candidates
            if not any(existing is mob for existing in self.children)
        ]
        if not candidates:
            return self
        self._validate_new_children([*self.children, *candidates])
        for mob in candidates:
            self.children.append(mob)
            mob._link_parent(self)
            self.anchor_priority = max(self.anchor_priority, 1 + mob.anchor_priority)
        bump_hierarchy_version()
        return self

    def remove_child(self, mob: Mob) -> Mob:
        """Detach a child so it stops following this Mob's transforms.

        The child stays in the scene and keeps its own state; it is simply no
        longer driven by this Mob. Any *other* parents it has are untouched, so
        it goes on following those.

        Animation
        ---------
        Not animated: the hierarchy changes immediately, and only affects
        animations recorded from here on.

        Parameters
        ----------
        mob
            The child to detach.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        ValueError
            If ``mob`` is not a child of this Mob.
        """
        if not self._drop_child(mob):
            raise ValueError("The requested Mob is not a child of this Mob")
        return self
