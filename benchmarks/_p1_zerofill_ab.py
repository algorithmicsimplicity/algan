"""A/B for the ``[T, N, D]`` rematerialization buffer's zero-fill (P1).

``generate_array_states`` builds ``torch.zeros((T, N, D))`` where ``N`` is every
row the scene ever allocated, then scatters the compact query for the ~31% of
rows that are actually live this window into it. The other ~69% exist only to
be read back as zero.

``torch.zeros`` on CPU is ``empty`` + an explicit ``memset`` over the whole
buffer. ``np.zeros`` is ``calloc``, and a request this large goes straight to
the OS page allocator, which hands back pages that are *already* zero and
charges only for the ones actually touched. If that holds on Windows, the fill
becomes proportional to the live rows rather than to ``N`` -- byte-identical,
since both produce zeros.

This measures the fill on its own, then the fill + a representative scatter, at
the shapes ``_p1_probe_s05.py`` recorded off the reference scene.

    .venv/Scripts/python.exe benchmarks/_p1_zerofill_ab.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _memory_cap import cap_process_memory  # noqa: E402

# Shapes here come from parameters, not from a live scene, so cap before torch
# is imported (a mis-sized generator has blue-screened this machine before).
cap_process_memory(float(os.environ.get("ALGAN_BENCH_MEM_GB", "6")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# (T, N, D) triples: the reference scene's window sizes against its ~505k rows,
# at the channel widths its attributes actually use.
CASES = [
    (50, 505_407, 1),
    (50, 505_407, 3),
    (50, 505_407, 4),
    (50, 505_407, 9),
    (25, 505_407, 4),
]
LIVE_FRACTION = 0.31
REPEATS = 5


def _timed(fn, repeats=REPEATS):
    """Median of ``repeats``.

    The buffer is freed between runs on purpose -- a real call always allocates
    fresh, and reusing one would measure a warm allocation this code never gets.
    """
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
        del out
    samples.sort()
    return samples[len(samples) // 2]


def torch_zeros(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)


def numpy_zeros(shape, dtype=torch.float32):
    np_dtype = torch.empty((), dtype=dtype).numpy().dtype
    return torch.from_numpy(np.zeros(shape, dtype=np_dtype))


def main():
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}")
    print(
        f"{'case':<22}{'torch.zeros':>13}{'np.zeros':>12}{'speedup':>9}"
        f"{'   +scatter(torch)':>19}{'+scatter(np)':>14}{'speedup':>9}"
    )
    for T, N, D in CASES:
        shape = (T, N, D)
        mb = T * N * D * 4 / 2**20
        R = int(N * LIVE_FRACTION)
        rows = torch.linspace(0, N - 1, R).to(torch.int64)
        src = torch.ones((T, R, D), dtype=torch.float32)

        def only_fill(alloc, shape=shape):
            return lambda: alloc(shape)

        def with_scatter(alloc, shape=shape, rows=rows, src=src):
            def run():
                out = alloc(shape)
                out.index_copy_(1, rows, src)
                return out

            return run

        t_fill = _timed(only_fill(torch_zeros))
        n_fill = _timed(only_fill(numpy_zeros))
        t_all = _timed(with_scatter(torch_zeros))
        n_all = _timed(with_scatter(numpy_zeros))

        print(
            f"T={T} N={N} D={D}".ljust(22)
            + f"{t_fill * 1e3:10.1f}ms{n_fill * 1e3:10.1f}ms{t_fill / max(n_fill, 1e-9):8.2f}x"
            + f"{t_all * 1e3:15.1f}ms{n_all * 1e3:12.1f}ms{t_all / max(n_all, 1e-9):8.2f}x"
            + f"   ({mb:.0f} MB)"
        )

    # The claim under test is that untouched pages are never charged. Verify it
    # directly: fill only, then fill + touch every row.
    T, N, D = 50, 505_407, 4
    all_rows = torch.arange(N, dtype=torch.int64)
    src_all = torch.ones((T, N, D), dtype=torch.float32)
    full = _timed(
        lambda: numpy_zeros((T, N, D)).index_copy_(1, all_rows, src_all), repeats=3
    )
    part = _timed(lambda: numpy_zeros((T, N, D)), repeats=3)
    print(
        f"\nnp.zeros untouched {part * 1e3:.1f}ms vs fully touched {full * 1e3:.1f}ms"
        " -- a large gap means the pages are genuinely lazy."
    )


if __name__ == "__main__":
    main()
