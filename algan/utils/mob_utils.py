from __future__ import annotations

import copy
from collections.abc import Sequence

import torch

from algan.animation.animation_contexts import *
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
                "batch indices must be integers or slices, "
                f"not {type(item).__name__}"
            )
        if item < 0:
            item += self.size
        if item < 0 or item >= self.size:
            raise IndexError("batched mob view index out of range")
        if item not in self._views:
            self._views[item] = self.mob[item]
        return self._views[item]


def batch_mobs(mobs, parent_batch_sizes=None, add_to_scene=True):
    mobs = list(traverse(mobs))
    if len(mobs) == 0:
        return None
    with Off(record_funcs=False, record_attr_modifications=False):
        batch_mob = mobs[0].clone(recursive=False, clone_data=True, add_to_scene=add_to_scene)
        for attr in batch_mob.animatable_attrs:
            if not all(hasattr(mob, attr) for mob in mobs):
                continue
            batch_mob.setattr_and_rebatch_without_record(
                attr,
                torch.cat(
                    [
                        mob.__getattribute__(attr).expand(-1, mob.location.shape[-2], -1)
                        for mob in mobs
                    ],
                    -2,
                ),
            )

        batch_sizes = [torch.tensor((_.location.shape[-2],)).view(-1) for _ in mobs]
        pbs = []
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

        batch_mob.components = components
        for i, c in enumerate(mobs[0].components):
            for attr in mobs[0].__dict__:
                if mobs[0].__getattribute__(attr) is c:
                    batch_mob.__setattr__(attr, components[i])

        children = [m.get_children(0, include_components=False) for m in mobs]
        child = batch_mobs(children, torch.tensor([len(_) for _ in children]), add_to_scene=add_to_scene)
        if child is not None:
            components.append(child)

        batch_mob.add_children(components)

        return batch_mob
