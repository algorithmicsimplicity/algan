import gc
import time
import cProfile
import pstats
from functools import wraps

import torch
import math

n = int(1e9 * 0.2)
k = int(math.sqrt(n))
x = torch.randn((n // k, k), device="cuda")
#y1 = x.clone()
#y2 = torch.randint(0, 1, (n,)).cuda()
def synced(f):
    @wraps(f)
    def _sync(*args, **kwargs):
        torch.cuda.synchronize()
        return f(*args, **kwargs)
    return _sync

@synced
def sort(x):
    x *= x
    #torch.cuda.synchronize()
    return x
    #x.sort(0)#out=(y1, y2))

num_iterations = 500

def rep(f, i=num_iterations):
    for _ in range(i):
        f()

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

with torch.no_grad():
    ops = [sort]#[normal_broadcast, gather_expanded, repeat_interleaved, jagged_broadcast]
    def run(i=num_iterations):
        for f in ops:
            gc.collect()
            torch.cuda.empty_cache()
            rep(lambda: f(x), i)
    run(1)
    profile_func(run)