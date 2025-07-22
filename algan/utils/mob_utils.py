import torch

from algan.utils.python_utils import traverse
from algan.animation.animation_contexts import *

def batch_mobs(mobs):
    mobs = list(traverse(mobs))
    i = 0
    while True:
        mobs.get_children(include_self=True)
        i += 1
    mob = _batch_mobs(mobs)

def _batch_mobs(mobs):
    if len(mobs) == 0:
        return None
    with Off(record_funcs=False, record_attr_modifications=False):
        batch_mob = mobs[0].clone(reset_history=True, recursive=False, clone_data=True)
        for attr in batch_mob.animatable_attrs:
            batch_mob.data.data_dict[attr] = torch.cat([_.__getattribute__(attr).expand(-1,_.location.shape[-2], -1) for _ in mobs], -2)
        #one = torch.ones((1,))
        batch_mob.parent_batch_sizes = torch.cat([torch.tensor((_.location.shape[-2],)).view(-1) for _ in mobs], -1)

        components = []
        for i in range(len(batch_mob.components)):
            components.append(batch_mobs([m.components[i] for m in mobs]))

        batch_mob.components = components
        child = batch_mobs([m.get_children(0, include_components=False) for m in mobs])
        if child is not None:
            components.append(child)

        batch_mob.add_children(components)
        for i, c in enumerate(mobs[0].components):
            for attr in mobs[0].__dict__.keys():
                if mobs[0].__getattribute__(attr) is c:
                    batch_mob.__setattr__(attr, components[i])
        return batch_mob

