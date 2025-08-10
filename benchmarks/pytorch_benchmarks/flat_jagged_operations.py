import torch
import gc

batch_size = 20
num_channels = 3
num_fragments = int(0.3e9) // (4)
num_objects = 100
num_fragments = num_fragments - (num_fragments % num_objects)
num_fragments_per_object = num_fragments // num_objects

dev = torch.device('cuda')
torch.set_default_device(dev)

objects = torch.full((num_objects,), 1)
fragments = torch.full((num_fragments_per_object, num_objects), 0)

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


@torch.compiler.disable(recursive=True)
def get_jagged(fragments):
    #return torch.nested.nested_tensor_from_jagged(squish(fragments, 1, 2), lengths=torch.tensor([num_objects for _ in range(num_fragments_per_object)]), jagged_dim=2)
    out = torch.nested.nested_tensor_from_jagged(fragments.view([-1]), lengths=torch.tensor([num_fragments_per_object for _ in range(num_objects)]), jagged_dim=1, max_seqlen=num_fragments_per_object)
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

#@compiled
def jagged_broadcast(fragments, objects):
    fragments_n = get_jagged(fragments)
    op(fragments_n, objects.unsqueeze(1))
    torch.cuda.synchronize()
    return fragments_n


if __name__ == '__main__':
    with torch.no_grad():
        ops = [jagged_broadcast]
        def run(i=num_iterations):
            for f in ops:
                gc.collect()
                torch.cuda.empty_cache()
                rep(lambda: f(fragments, objects), i)
        run(1)
        profile_func(run)
