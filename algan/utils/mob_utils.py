import torch

from algan.utils.python_utils import traverse
from algan.animation.animation_contexts import *


def batch_mobs(mobs, parent_batch_sizes = None):
    mobs = list(traverse(mobs))
    if len(mobs) == 0:
        return None
    orig_parent_batch_sizes = parent_batch_sizes
    with Off(record_funcs=False, record_attr_modifications=False):
        batch_mob = mobs[0].clone(reset_history=True, recursive=False, clone_data=True)
        for attr in batch_mob.animatable_attrs:
            batch_mob.data.data_dict[attr] = torch.cat([_.__getattribute__(attr).expand(-1,_.location.shape[-2], -1) for _ in mobs], -2)

        batch_sizes = [torch.tensor((_.location.shape[-2],)).view(-1) for _ in mobs]
        pbs = []
        if parent_batch_sizes is None:
            parent_batch_sizes = torch.tensor((len(mobs),), dtype=torch.long)
        i = 0
        for j in range(len(parent_batch_sizes)):
            try:
                pbs.append(sum(batch_sizes[i:i+parent_batch_sizes[j]]).view(-1) if parent_batch_sizes[j] > 0 else torch.zeros((1,), dtype=torch.long))
            except AttributeError:
                pbs.append(sum(batch_sizes[i:i+parent_batch_sizes[j]]).view(-1) if parent_batch_sizes[j] > 0 else torch.zeros((1,), dtype=torch.long))

            i += parent_batch_sizes[j]

        child_pbs = torch.cat(pbs, -1)
        batch_mob.parent_batch_sizes = child_pbs

        components = []
        for i in range(len(batch_mob.components)):
            try:
                components.append(batch_mobs([m.components[i] for m in mobs], torch.ones((child_pbs.sum(),), dtype=torch.long)))
            except:
                components.append(batch_mobs([m.components[i] for m in mobs], torch.ones((child_pbs.sum(),), dtype=torch.long)))

        batch_mob.components = components
        for i, c in enumerate(mobs[0].components):
            for attr in mobs[0].__dict__.keys():
                if mobs[0].__getattribute__(attr) is c:
                    batch_mob.__setattr__(attr, components[i])

        children = [m.get_children(0, include_components=False) for m in mobs]
        child = batch_mobs(children, torch.tensor([len(_) for _ in children]))
        if child is not None:
            components.append(child)

        batch_mob.add_children(components)
        for c in batch_mob.children:
            if batch_mob.location.shape[-2] != len(c.parent_batch_sizes) or c.parent_batch_sizes.sum() != c.location.shape[-2]:
                print(' ')

        return batch_mob

