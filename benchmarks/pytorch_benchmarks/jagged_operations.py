import torch
import gc

from torch.nested._internal.nested_tensor import nested_from_padded

dev = torch.device('cuda')
torch.set_default_device(dev)

batch_size = 20
num_channels = 3
num_fragments = int(0.3e9) // (batch_size * num_channels * 4)
num_objects = 100
num_fragments = num_fragments - (num_fragments % num_objects)
num_fragments_per_object = num_fragments // num_objects

repeat_inds = torch.tensor([num_fragments_per_object for _ in range(num_objects)], device=dev)
gather_inds = torch.repeat_interleave(torch.arange(len(repeat_inds), device=dev), repeat_inds)
lengths = repeat_inds


def broadcast_to_jagged(t, nt):
    min_seqlen = nt._maybe_min_seqlen
    max_seqlen = nt._maybe_max_seqlen
    padded_max_S = max_seqlen
    total_L = nt._values.shape[nt._ragged_idx - 1]
    if padded_max_S is None:
        # use upper bound on max seqlen if it's not present
        padded_max_S = total_L

    # convert dense tensor -> jagged
    t = t.expand(
        [x if i != nt._ragged_idx else padded_max_S for i, x in enumerate(t.shape)]
    )
    t_as_nt = nested_from_padded(
        t,
        offsets=nt._offsets,
        ragged_idx=nt._ragged_idx,
        sum_S=total_L,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )
    return t_as_nt

@torch.compiler.disable(recursive=True)
def get_jagged(fragments):
    #return torch.nested.nested_tensor_from_jagged(squish(fragments, 1, 2), lengths=torch.tensor([num_objects for _ in range(num_fragments_per_object)]), jagged_dim=2)
    out = torch.nested.nested_tensor_from_jagged(fragments.view([*fragments.shape[:-2], -1]),
                                                 lengths=lengths, jagged_dim=3, max_seqlen=num_fragments_per_object)
    torch.cuda.synchronize()
    return out


fragments = torch.full((batch_size, num_channels, num_objects, num_fragments_per_object), 0)
objects = torch.full((batch_size, num_channels, num_objects, 1), 1)
objects_j = objects.permute(2, 0, 1, 3)
objects_broadcast = broadcast_to_jagged(objects_j, get_jagged(fragments))

num_iterations = 300
import time
import cProfile
import pstats

#import torch_tensorrt


compiled = lambda f: f#torch.compile(f, dynamic=False, fullgraph=False)#, backend='cudagraphs')
                                                    #, backend="tensorrt",
                                                    #options={"min_block_size": 1},)
                                                    #         "use_python_runtime": True, }
                                                    #)#"onnxrt")# mode="reduce-overhead")#backend='cudagraphs')


def profile_func(func):
    pr = cProfile.Profile()
    start = time.time()
    pr.enable()
    out = func()
    pr.disable()
    end = time.time()

    with open('profiler_dump.txt', 'w') as f:
        ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    print(f'took {end - start} seconds.')
    return out


def get_repeat_interleaved(objects):
    out = torch.repeat_interleave(objects.squeeze(-1), repeat_inds, -1).view([*objects.shape[:-1], -1])
    torch.cuda.synchronize()
    return out


def get_gather_expanded(objects):
    out = torch.gather(objects.squeeze(-1), -1, gather_inds.expand(objects.shape[0], objects.shape[1], -1)).view([*objects.shape[:-1], -1]).contiguous()
    torch.cuda.synchronize()
    return out


def op(x, y):
    x += y
    #torch.cuda.synchronize()
    return x


def rep(f, i=num_iterations):
    for _ in range(i):
        f()
        #gc.collect()
        #torch.cuda.empty_cache()


@compiled
def _normal_broadcast(fragments, objects):
    op(fragments, objects)
    return fragments


def normal_broadcast(fragments, objects):
    out = _normal_broadcast(fragments, objects)
    torch.cuda.synchronize()
    return out

@compiled
def jagged_broadcast(fragments, objects):
    fragments_n = get_jagged(fragments)
    op(fragments_n, objects_j)
    torch.cuda.synchronize()
    return fragments_n


@compiled
def _gather_expanded(fragments, objects):
    objects_n = get_gather_expanded(objects)
    op(fragments, objects_n)
    return fragments


def gather_expanded(fragments, objects):
    out = _gather_expanded(fragments, objects)
    torch.cuda.synchronize()
    return out


def scatter_add(fragments, objects):
    pass
    #torch.scatter_add(fragments.flatten(0,1), 0, ind, objects)


@compiled
def repeat_interleaved(fragments, objects):
    objects_n = get_repeat_interleaved(objects)
    op(fragments, objects_n)


if __name__ == '__main__':
    #x = torch.nested.nested_tensor_from_jagged(torch.tensor([1000, ]), lengths=torch.tensor([400, 300, 200, 100])).cpu()
    #y = torch.ones((4,1)).cpu()
    #print(x + y)

    with torch.no_grad():
        ops = [normal_broadcast, gather_expanded, repeat_interleaved, jagged_broadcast]
        def run(i=num_iterations):
            for f in ops:
                gc.collect()
                torch.cuda.empty_cache()
                rep(lambda: f(fragments, objects), i)
        run(1)
        profile_func(run)
