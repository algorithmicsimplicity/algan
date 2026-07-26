"""Screen-relative layout and bounding-box queries for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobLayoutMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

from algan.animatable_base.animatable import Animatable
from algan.animation_timeline.timeline import HIERARCHY_VERSION
from algan.animation_timeline.timeline import (
    _opt_disabled,
    bump_hierarchy_version,
)
from algan.utils.python_utils import traverse
from algan.errors import HierarchyError


class MobHierarchyMixin:
    """All methods related to managing to mob hierarchy, i.e. parent/child/descendant relationships. """

    def set_parent_to(self, other_mob):
        if not any(parent is other_mob for parent in self.parents):
            self.parents.append(other_mob)
        return self

    def remove_parent(self, other_mob):
        self.parents[:] = [
            parent for parent in self.parents if parent is not other_mob
        ]
        return self

    def get_children(self, generation=0, include_components=True):
        children = self.children
        if not include_components:
            children = [_ for _ in children if _ not in self.components]
        if generation <= 0:
            return children
        children = [_.get_children(generation - 1) for _ in children]
        return [x for l in children for x in l]

    def get_descendants(self, include_self: bool = True) -> list[Mob]:
        """Retrieves a list all descendant Mobs in the hierarchy, optionally including itself.

        The traversal is cached against the global structure version (bumped
        by any hierarchy change), because recorded-function replay re-reads it
        for every event of every frame batch.

        Parameters
        ----------
        include_self
            If True, the current Mob instance
            is included in the returned list.

        Returns
        -------
        list[Mob]
            A flat list containing the Mob and all its children,
            grandchildren, and so on.

        """
        cache = getattr(self, "_descendants_cache", None)
        if (cache is not None and cache[0] == HIERARCHY_VERSION[0]
                and not _opt_disabled("desccache")):
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
            if self._contains_in_hierarchy(mob, self):
                raise HierarchyError("This hierarchy mutation would create a cycle")

    def replace_children(self, mobs, *, link_parents=True):
        """Replace the canonical child list while preserving hierarchy links."""
        new_children = list(traverse(mobs))
        self._validate_new_children(new_children)
        old_children = list(self.children)

        for child in old_children:
            if link_parents and not any(child is item for item in new_children):
                child.remove_parent(self)

        self.children[:] = new_children
        if link_parents:
            for child in new_children:
                child.set_parent_to(self)
        self.anchor_priority = max(
            (1 + child.anchor_priority for child in new_children),
            default=0,
        )
        bump_hierarchy_version()
        return self

    def add_children(self, *mobs):
        """Add children while rejecting cycles and duplicate relationships."""
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
            mob.set_parent_to(self)
            self.anchor_priority = max(
                self.anchor_priority, 1 + mob.anchor_priority
            )
        bump_hierarchy_version()
        return self

    def remove_child(self, mob):
        for index, child in enumerate(self.children):
            if child is mob:
                del self.children[index]
                child.remove_parent(self)
                self.anchor_priority = max(
                    (1 + item.anchor_priority for item in self.children),
                    default=0,
                )
                bump_hierarchy_version()
                return self
        raise ValueError("The requested Mob is not a child of this Mob")
