"""Why a memory-bound Taichi CPU kernel can be *slower* than the torch call it replaces.

"Memory-bound" explains why a kernel is not faster. It does not explain why one
is **slower** -- a loop that streams the same bytes should at worst tie. This
measures where the difference actually comes from, because the answer decided
whether two of the three kernels in P13 were worth shipping (they were not, and
one of them was slower only because of how its loop was written).

Run on a CPU arch, with the machine otherwise idle::

    ALGAN_RENDER_DEVICE=cpu ALGAN_USE_DAEMON=0 uv run python benchmarks/_taichi_loop_shapes_taichi.py

Measured on a 4-core x64 box, ``[950, 1518, 3]`` f32 (35 MB moved), all arms
writing into a preallocated buffer:

============================  ==========  ========================================
form                          GB/s        note
============================  ==========  ========================================
``torch.Tensor.copy_``        30-59       vectorized memcpy; the target to beat
ti flat 1-D loop              14-22       Taichi's ceiling, 1.4-4x off torch
ti nested plain loops, 3-D    7.3-7.5     multi-dimensional addressing costs ~2-3x
ti ``ndrange(B, L)`` + static 4.3-4.4     ndrange decomposition on top of that
ti ``ndrange(B, L, C)``       1.7-1.9     **8-12x off the flat loop, same bytes**
============================  ==========  ========================================

Absolute throughput swings run to run on a shared 4-core box; the *ordering* and
the ratios between the Taichi forms are stable, and those are what the findings
rest on.

Three findings, in order of how much they cost:

1. **``ti.ndrange`` over several dimensions is expensive.** Taichi flattens it
   into one parallel loop and recovers every index per iteration, and that
   arithmetic can dominate a copy whose useful work is one load and one store.
   Rewriting P13's gather from ``ndrange(B, L)`` to a flat loop took it from
   0.68x of torch to ~1.1x -- the kernel was not slow because gathers are slow.
2. **Multi-dimensional ndarray indexing costs again**, about 3x against flat
   offsets, even with plain nested loops and no ndrange.
3. **Launch overhead is ~80 us per call**, which only matters for small work.

And one that is genuinely structural: even Taichi's best form streams at ~0.75x
of torch's vectorized ``copy_`` at best, and less when torch's copy is
warm. That is the floor a pure-copy kernel cannot get
under, and it is why P13's colour bake stays off however it is written.

``advanced_optimization`` is **not** the explanation, which is worth recording
because it is the obvious suspect (Algan ships it off; see
``taichi_runtime.taichi_init_kwargs``). Re-running every row with
``ALGAN_ADV_OPT=1`` changes nothing outside noise.

**Measurement trap, hit twice while writing this.** If the torch arm allocates
its result and the kernel arm writes into a preallocated buffer, the comparison
is charging torch an allocation the kernel never pays -- worth more than the
effect at these sizes, and it inverted one conclusion. Every arm here allocates
or every arm reuses, never a mix.
"""

import statistics
import time

import taichi as ti
import torch

from algan.rendering.taichi_runtime import init_taichi, taichi_arch_is_cpu

init_taichi()


@ti.kernel
def copy_flat(x: ti.types.ndarray(dtype=ti.f32, ndim=1),
              out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    """The ceiling: one flat parallel loop, flat offsets, no decomposition."""
    for i in range(x.shape[0]):
        out[i] = x[i]


@ti.kernel
def copy_ndrange3(x: ti.types.ndarray(dtype=ti.f32, ndim=3),
                  out: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    """One ndrange per element -- the shape to avoid."""
    for b, i, c in ti.ndrange(x.shape[0], x.shape[1], x.shape[2]):
        out[b, i, c] = x[b, i, c]


@ti.kernel
def copy_ndrange2_static(x: ti.types.ndarray(dtype=ti.f32, ndim=3),
                         out: ti.types.ndarray(dtype=ti.f32, ndim=3),
                         channels: ti.template()):
    """ndrange over the outer two dims, channel loop unrolled at compile time."""
    for b, i in ti.ndrange(x.shape[0], x.shape[1]):
        for c in ti.static(range(channels)):
            out[b, i, c] = x[b, i, c]


@ti.kernel
def copy_nested(x: ti.types.ndarray(dtype=ti.f32, ndim=3),
                out: ti.types.ndarray(dtype=ti.f32, ndim=3),
                channels: ti.template()):
    """Plain nested loops: no ndrange, but still multi-dimensional addressing."""
    for b in range(x.shape[0]):
        for i in range(x.shape[1]):
            for c in ti.static(range(channels)):
                out[b, i, c] = x[b, i, c]


def bench(fn, iters=20, rounds=5):
    for _ in range(3):
        fn()
    runs = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        runs.append((time.perf_counter() - start) / iters)
    return statistics.median(runs)


def main():
    if not taichi_arch_is_cpu():
        print("Taichi is not on a CPU arch; this measures the CPU backend.")
        return 1

    config = ti.lang.impl.get_runtime().prog.config()
    print(
        f"threads: torch {torch.get_num_threads()}, "
        f"taichi {getattr(config, 'cpu_max_num_threads', '?')}; "
        f"advanced_optimization={bool(getattr(config, 'advanced_optimization', False))}"
    )

    B, L, C = 950, 1518, 3
    source = torch.rand(B, L, C)
    destination = torch.empty_like(source)
    flat_source, flat_destination = source.view(-1), destination.view(-1)
    moved = 2 * source.numel() * 4
    print(f"\ncopying [{B}, {L}, {C}] f32 ({moved / 1e6:.0f} MB moved), preallocated:")
    for name, run in (
        ("torch copy_", lambda: destination.copy_(source)),
        ("ti flat 1-D", lambda: copy_flat(flat_source, flat_destination)),
        ("ti nested plain loops", lambda: copy_nested(source, destination, C)),
        ("ti ndrange(B, L) + static", lambda: copy_ndrange2_static(source, destination, C)),
        ("ti ndrange(B, L, C)", lambda: copy_ndrange3(source, destination)),
    ):
        seconds = bench(run)
        print(f"  {name:<28}{seconds * 1e3:>8.2f} ms{moved / seconds / 1e9:>9.1f} GB/s")

    print("\nlaunch overhead (1 element, 2000 calls):")
    tiny_in, tiny_out = torch.rand(1), torch.empty(1)
    start = time.perf_counter()
    for _ in range(2000):
        copy_flat(tiny_in, tiny_out)
    print(f"  {(time.perf_counter() - start) / 2000 * 1e6:.1f} us/launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
