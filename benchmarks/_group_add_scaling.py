"""Time growing a Group one ``add`` at a time against building it in one call.

Authoring only, nothing is rendered. For each size N this spawns N Dots, adds
them to an empty spawned Group one by one, and separately builds the same
collection with one ``Group(mobs)`` call, each in a fresh Scene. Incremental
adds should cost about N times one add, not N squared: before the fix 2,000
adds took four minutes on a CPU container, against 0.4 s for ``Group(mobs)``.

    <venv-python> benchmarks/_group_add_scaling.py [--sizes 500 1000 2000]

Dot construction is not counted in the first two columns. ``--interleaved``
adds a column for the loop as a script writes it, ``group.add(Dot().spawn())``,
which does count construction (5-7 ms a Dot). Authoring a Scene it never renders,
the script ends with Algan's "finished without rendering" note.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Benchmarks must never be measured inside a warm daemon (see benchmark_empty.py).
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from algan import Dot, Group, Scene  # noqa: E402


def _dots(n):
    return [Dot().spawn() for _ in range(n)]


def time_incremental(n):
    with Scene():
        dots = _dots(n)
        group = Group().spawn()
        start = time.perf_counter()
        for dot in dots:
            group.add(dot)
        return time.perf_counter() - start


def time_interleaved(n):
    """The loop as a script writes it: build, spawn and add one at a time.

    Times the whole loop, Dot construction included, so it is not comparable
    with the other two columns; compare it across versions.
    """
    with Scene():
        group = Group().spawn()
        start = time.perf_counter()
        for _ in range(n):
            group.add(Dot().spawn())
        return time.perf_counter() - start


def time_at_once(n):
    with Scene():
        dots = _dots(n)
        start = time.perf_counter()
        Group(dots)
        return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument(
        "--interleaved",
        action="store_true",
        help="also time `group.add(Dot().spawn())` in a loop, construction included",
    )
    args = parser.parse_args()
    print(
        f"{'N':>6} {'N adds (s)':>12} {'per add (ms)':>13} {'Group(mobs) (s)':>16} "
        f"{'build+add loop (s)':>19}"
    )
    for n in args.sizes:
        incremental = time_incremental(n)
        at_once = time_at_once(n)
        interleaved = time_interleaved(n) if args.interleaved else float("nan")
        print(
            f"{n:>6} {incremental:>12.2f} {1000 * incremental / n:>13.2f} "
            f"{at_once:>16.2f} {interleaved:>19.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
