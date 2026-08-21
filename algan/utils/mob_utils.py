"""Batched views over collections of Mobs.

Algan packs many like Mobs into a single tensor batch -- every glyph of a
:class:`~algan.mobs.text.Text`, every sphere of a point cloud -- so that one Mob,
one Scene actor and one render primitive cover all of them.

There are two ways to build such a pack. A class that can construct its geometry
for many objects at once offers a ``from_batches`` constructor
(:meth:`~algan.mobs.bezier_circuit.BezierCircuitCubic.from_batches`,
:meth:`~algan.mobs.surfaces.surface.Surface.from_batches`), which never creates
the per-object Mobs at all; ``pack_animatable_rows`` does the timeline
bookkeeping those constructors share. ``batch_mobs`` is the generic fallback: it
packs Mobs that already exist, so it saves render-time cost but not construction
cost.

:class:`BatchedMobViewSequence` presents the result as an ordinary indexable
sequence: ``text.character_mobs[3]`` returns a lazily-constructed view onto row 3
of the batch, which behaves like a Mob but owns no storage of its own. A view
shares its source's id, and therefore its timeline rows and its lifespan -- so
members of a pack cannot spawn or despawn independently, and stagger their
entrances through opacity instead (this is what ``Tex.write()`` does).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from algan.animation_timeline.animation_contexts import *
from algan.utils.python_utils import traverse


class BatchedMobViewSequence(Sequence):
    """Sequence of lazy views into a mob's batch dimension.

    Creating the views eagerly would recreate the Python object graph that
    batching is intended to avoid.  A view is therefore cloned only when a
    caller indexes or iterates over that element, and is cached so repeated
    indexing has the same object identity as an ordinary list of mobs.  The
    clone shares the packed mob's timeline rows; only ``data_sub_inds`` differs.
    """

    def __init__(self, mob, size):
        self.mob = mob
        self.size = int(size)
        self._views = {}

    def __len__(self):
        return self.size

    def __deepcopy__(self, memo):
        """Copy the packed owner while rebuilding indexed views lazily.

        ``_views`` is a derived cache whose entries share the owner's timeline
        rows. Deep-copying those entries independently gives them fresh owner
        rows while retaining their old global ``data_sub_inds``, which can make
        the copied view index beyond its new local timeline allocation.
        """
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        clone.mob = copy.deepcopy(self.mob, memo)
        clone.size = self.size
        clone._views = {}
        return clone

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(self.size))]
        if not isinstance(item, int):
            raise TypeError(
                f"batch indices must be integers or slices, not {type(item).__name__}"
            )
        if item < 0:
            item += self.size
        if item < 0 or item >= self.size:
            raise IndexError("batched mob view index out of range")
        if item not in self._views:
            self._views[item] = self.mob[item]
        return self._views[item]


def _widen_packed_attrs(mob, count, rows_per_member, overrides):
    """Rewrite every animatable attribute of ``mob`` to cover ``count`` members."""
    total = count * rows_per_member
    for attr in mob.animatable_attrs:
        try:
            value = getattr(mob, attr)
        except AttributeError:
            continue
        if attr in overrides:
            value = overrides[attr]
        elif value.shape[-2] == 1:
            # One shared row: every member reads the same value.
            value = value.expand(*value.shape[:-2], total, value.shape[-1]).contiguous()
        elif rows_per_member > 1 and value.shape[-2] == rows_per_member and count > 1:
            # One member's worth of rows: give every member its own copy.
            value = value.repeat(*([1] * (value.dim() - 2)), count, 1).contiguous()
        mob._setattr_and_rebatch_without_record(attr, value)


def pack_animatable_rows(mob, count, overrides=None):
    """Widen a Mob's attribute rows to one per logical object, and declare the batch.

    A packed Mob stands for ``count`` logical objects held in one tensor batch:
    every animatable attribute carries one row per object instead of a single
    shared row, and ``parent_batch_sizes`` records the boundaries that
    :meth:`~algan.animatable_base.mob.Mob.__getitem__` slices on.  Every
    construction-time packer needs those same steps, so they live here instead
    of being copied into each one.  Use :func:`pack_member_rows` for a component
    whose members own several rows each, such as a surface's vertex grid.

    The writes go through
    :meth:`~algan.animatable_base.animatable.Animatable._setattr_and_rebatch_without_record`,
    which re-allocates the Mob's timeline rows and leaves any recorded history
    behind on the old ones, so this is only valid on a Mob whose history is
    fresh -- which at construction time it always is.

    Parameters
    ----------
    mob
        The Mob to pack.  Modified in place.
    count
        Number of logical objects the batch holds.
    overrides
        Values to write instead of the widened current ones, as
        ``{attr_name: tensor}``; each must already carry ``count`` rows.
        Defaults to None, which widens every attribute.

    Returns
    -------
    :class:`~algan.animatable_base.mob.Mob`
        The packed Mob, so calls can be chained.
    """
    with Off(
        record_funcs=False,
        record_attr_modifications=False,
        animation_manager=mob.animation_manager,
    ):
        _widen_packed_attrs(mob, count, 1, {} if overrides is None else overrides)
        # The compressed encoding: one entry holding the member count, which
        # Mob._set_data_sub_inds expands back to one entry per member on the
        # first index.
        mob.parent_batch_sizes = torch.tensor((count,), dtype=torch.long)
        mob.singleton_batch_indexing = True
    return mob


def pack_member_rows(mob, count, rows_per_member, overrides=None):
    """Pack a component whose every member owns ``rows_per_member`` rows.

    The counterpart to :func:`pack_animatable_rows` for the level below a pack:
    a surface's vertex grid, or a circuit's texture grid, where one logical
    object is a block of rows rather than a single row.  Attributes already
    holding one member's worth of rows are repeated per member; single shared
    rows are widened across the whole batch.

    Parameters
    ----------
    mob
        The component Mob to pack.  Modified in place.
    count
        Number of logical objects the parent batch holds.
    rows_per_member
        Rows this component owns for each of those objects.
    overrides
        Values to write instead of the widened current ones, as
        ``{attr_name: tensor}``; each must already carry
        ``count * rows_per_member`` rows.  Defaults to None.

    Returns
    -------
    :class:`~algan.animatable_base.mob.Mob`
        The packed component, so calls can be chained.
    """
    with Off(
        record_funcs=False,
        record_attr_modifications=False,
        animation_manager=mob.animation_manager,
    ):
        mob.parent_batch_sizes = torch.full((count,), rows_per_member, dtype=torch.long)
        _widen_packed_attrs(
            mob, count, rows_per_member, {} if overrides is None else overrides
        )
    return mob


def batch_mobs(mobs, parent_batch_sizes=None, add_to_scene=True):
    """Pack existing Mobs into one Mob holding all of their rows.

    The generic counterpart to a class's own ``from_batches``: it works for any
    Mob class, but the per-object Mobs have to exist first, so it saves
    render-time cost and Scene-actor count rather than construction cost.

    Parameters
    ----------
    mobs
        The Mobs to pack, in any nested iterable.  They must all belong to one
        Scene.
    parent_batch_sizes
        How many of ``mobs`` belong to each row of the pack's parent, used when
        recursing into components and children.  Defaults to None, which builds
        a root pack whose ``parent_batch_sizes`` instead carries the member
        boundaries that indexing slices on.
    add_to_scene
        Whether the pack is registered as a Scene actor.  Defaults to True.

    Returns
    -------
    :class:`~algan.animatable_base.mob.Mob` or None
        The packed Mob, or None when ``mobs`` is empty.
    """
    mobs = list(traverse(mobs))
    if len(mobs) == 0:
        return None
    scene = mobs[0].scene
    if any(mob.scene is not scene for mob in mobs[1:]):
        raise ValueError("Cannot batch Mobs from multiple Scenes")
    with Off(
        record_funcs=False,
        record_attr_modifications=False,
        animation_manager=scene.animation_manager,
    ):
        # Cloning with add_to_scene would register every component clone as its
        # own actor, with its own timeline rows -- and the batched components
        # built below replace them, so each pack would leak one dead actor per
        # component. Register the pack itself instead.
        batch_mob = mobs[0].clone(recursive=False, clone_data=True, add_to_scene=False)
        if add_to_scene:
            scene.add_actor(batch_mob)
            mobs[0].animation_manager.context.add_mob(batch_mob)
        for attr in batch_mob.animatable_attrs:
            if not all(hasattr(mob, attr) for mob in mobs):
                continue
            batch_mob._setattr_and_rebatch_without_record(
                attr,
                torch.cat(
                    [
                        mob.__getattribute__(attr).expand(
                            -1, mob.location.shape[-2], -1
                        )
                        for mob in mobs
                    ],
                    -2,
                ),
            )

        batch_sizes = [torch.tensor((_.location.shape[-2],)).view(-1) for _ in mobs]
        member_sizes = torch.cat(batch_sizes, -1)
        pbs = []
        is_root_pack = parent_batch_sizes is None
        if parent_batch_sizes is None:
            parent_batch_sizes = torch.tensor((len(mobs),), dtype=torch.long)
        i = 0
        for j in range(len(parent_batch_sizes)):
            pbs.append(
                sum(batch_sizes[i : i + parent_batch_sizes[j]]).view(-1)
                if parent_batch_sizes[j] > 0
                else torch.zeros((1,), dtype=torch.long)
            )

            i += parent_batch_sizes[j]

        child_pbs = torch.cat(pbs, -1)
        if is_root_pack:
            # A root pack has no parent whose edits need expanding, so its
            # parent_batch_sizes carries the member boundaries that
            # Mob.__getitem__ slices on instead. Summing them into one entry
            # (the old behaviour) made pack[i] raise IndexError for every i but
            # 0, because one entry describes one member.
            if bool((member_sizes == 1).all()):
                # The compressed form the direct packers use, which
                # _set_data_sub_inds expands back into member_sizes on the
                # first index. Kept identical to them so batch_mobs stays
                # usable as their reference implementation.
                batch_mob.parent_batch_sizes = torch.tensor(
                    (len(mobs),), dtype=torch.long
                )
                batch_mob.singleton_batch_indexing = True
            else:
                # Members owning several rows each, which the compressed form
                # cannot express.
                batch_mob.parent_batch_sizes = member_sizes
        else:
            batch_mob.parent_batch_sizes = child_pbs

        components = []
        for i in range(len(batch_mob.components)):
            components.append(
                batch_mobs(
                    [m.components[i] for m in mobs],
                    torch.ones((child_pbs.sum(),), dtype=torch.long),
                    add_to_scene=add_to_scene,
                )
            )

        # A copy: the batched children appended below are children, not
        # components, and aliasing the two lists made get_children(
        # include_components=False) report a packed subtree as structural.
        batch_mob.components = list(components)
        for i, c in enumerate(mobs[0].components):
            for attr in mobs[0].__dict__:
                if mobs[0].__getattribute__(attr) is c:
                    batch_mob.__setattr__(attr, components[i])

        children = [m.get_children(0, include_components=False) for m in mobs]
        child = batch_mobs(
            children,
            torch.tensor([len(_) for _ in children]),
            add_to_scene=add_to_scene,
        )
        if child is not None:
            components.append(child)

        batch_mob.add_children(components)

        # The pack was cloned from ``mobs[0]``, so anything a class derived from
        # its own geometry at construction describes ONE member and now has to
        # be redone against all of them. ``from_batches`` needs no such hook --
        # it hands the constructor every member's geometry to begin with.
        batch_mob._after_repack()

        return batch_mob
