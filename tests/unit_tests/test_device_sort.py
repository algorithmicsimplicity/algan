"""What decides whether a sort leaves torch, and what the kernel is handed.

The sort itself cannot be run here: ``qd.algorithms.sort`` is built from
``block.sync`` and ``block.radix_rank_match_atomic_or``, which the CPU arch
does not implement, so the only box that can execute it is one with a GPU
(``benchmarks/_device_sort_probe.py`` is that measurement, and it checks the
permutation against ``torch.argsort`` on real hardware). What *is* testable
anywhere is everything around it: the gate that decides the arm, and the
arguments the wrapper builds -- the seeding mode, the pass count and the scan
depth are compile-time template values, so getting one wrong compiles a
different sort rather than raising.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import device_sort

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def recorded(monkeypatch):
    """Make the kernel arm reachable on any box, and record what it was called with."""
    calls = []

    def fake_kernel(*args):
        calls.append(args)

    monkeypatch.setattr(device_sort, "radix_sort_available", lambda keys: True)
    monkeypatch.setitem(
        __import__("sys").modules,
        "algan.rendering.raytracing.radix_sort_taichi",
        type(
            "module",
            (),
            {"argsort_pairs": staticmethod(fake_kernel), "__spec__": None},
        ),
    )
    return calls


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def test_the_cpu_arch_never_takes_the_kernel(monkeypatch):
    """A CPU box has no device sort, whatever the mode says.

    ``block.sync`` is unimplemented there, so the kernel would not compile --
    and the CPU's own ``argsort`` is a parallel merge sort that has nothing to
    gain anyway.
    """
    monkeypatch.setenv("ALGAN_DEVICE_RADIX_SORT", "1")
    keys = torch.arange(1 << 17, dtype=torch.int32)
    assert device_sort.radix_sort_enabled() is True
    assert device_sort.radix_sort_available(keys) is False
    assert device_sort.stable_argsort(keys) is None


def test_the_mode_follows_mps_friendly_and_the_variable_overrides_it(monkeypatch):
    """The default is the mode whose premise is "torch's ops are the slow ones"."""
    monkeypatch.setattr(device_sort, "mps_friendly", lambda: True)
    assert device_sort.radix_sort_enabled() is True
    monkeypatch.setattr(device_sort, "mps_friendly", lambda: False)
    assert device_sort.radix_sort_enabled() is False

    monkeypatch.setenv("ALGAN_DEVICE_RADIX_SORT", "1")
    assert device_sort.radix_sort_enabled() is True
    monkeypatch.setenv("ALGAN_DEVICE_RADIX_SORT", "0")
    monkeypatch.setattr(device_sort, "mps_friendly", lambda: True)
    assert device_sort.radix_sort_enabled() is False


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.int16, torch.uint8, torch.bool, torch.float16]
)
def test_a_dtype_the_sort_cannot_order_declines(monkeypatch, dtype):
    monkeypatch.setenv("ALGAN_DEVICE_RADIX_SORT", "1")
    keys = torch.zeros(1 << 17, dtype=dtype)
    assert device_sort.radix_sort_available(keys) is False


def test_a_small_input_stays_on_torch(monkeypatch):
    """The emitted launch chain has a fixed length; under a floor it dominates."""
    monkeypatch.setenv("ALGAN_DEVICE_RADIX_SORT", "1")
    monkeypatch.setenv("ALGAN_DEVICE_RADIX_SORT_MIN", "1024")
    assert (
        device_sort.radix_sort_available(torch.zeros(1023, dtype=torch.int32)) is False
    )
    # Above the floor only the arch is left to refuse it, and here it does.
    assert device_sort._minimum_elements() == 1024


# ---------------------------------------------------------------------------
# what the kernel is handed
# ---------------------------------------------------------------------------


def _unpack(call):
    (
        src_key,
        perm_in,
        work_keys,
        tmp_keys,
        values,
        tmp_values,
        scratch,
        count_buf,
        count,
        key_dtype,
        end_bit,
        depth,
        mode,
    ) = call
    return locals()


def test_a_fresh_argsort_seeds_mode_zero(recorded):
    keys = torch.arange(1 << 17, dtype=torch.int32)
    order = device_sort.stable_argsort(keys)
    (call,) = recorded
    got = _unpack(call)
    assert got["mode"] == 0
    assert got["end_bit"] == 32
    assert got["count"] == keys.numel()
    assert order is got["values"]
    assert order.dtype is torch.int32
    # With nothing to compose, the unused permutation argument aliases the
    # output rather than costing an allocation of its own.
    assert got["perm_in"] is order
    assert got["work_keys"].dtype is torch.int32
    assert got["count_buf"].shape == ()


def test_composing_with_an_existing_order_picks_the_seeding_by_key_space(recorded):
    keys = torch.arange(1 << 17, dtype=torch.int64)
    perm = torch.arange(1 << 17, dtype=torch.int32)

    device_sort.stable_argsort(keys, perm=perm, keys_are_permuted=True)
    assert _unpack(recorded[-1])["mode"] == 1

    device_sort.stable_argsort(keys, perm=perm, keys_are_permuted=False)
    assert _unpack(recorded[-1])["mode"] == 2
    # An int64 key is eight passes where an int32 key is four, which is the
    # whole reason ``sheets._narrow_sort_key`` exists.
    assert _unpack(recorded[-1])["end_bit"] == 64


def test_a_wider_permutation_is_narrowed_for_the_kernel(recorded):
    keys = torch.arange(1 << 17, dtype=torch.int32)
    perm = torch.arange(1 << 17, dtype=torch.int64)
    device_sort.stable_argsort(keys, perm=perm, keys_are_permuted=True)
    handed = _unpack(recorded[-1])["perm_in"]
    assert handed.dtype is torch.int32
    assert torch.equal(handed.to(torch.int64), perm)


@pytest.mark.parametrize(
    ("n", "depth"), [(1 << 16, 2), ((1 << 16) + 1, 3), (1 << 24, 3), ((1 << 24) + 1, 4)]
)
def test_the_scan_depth_covers_the_count_from_a_small_fixed_set(n, depth):
    """``log256_max_n`` is a specialization key: one per input size would compile
    a fresh sort for every chunk. Three values cover everything that fits.
    """
    assert device_sort._scan_depth(n) == depth
    assert 256**depth >= n


def test_a_lexsort_composes_least_significant_key_first(recorded):
    pixel = torch.arange(1 << 17, dtype=torch.int64)
    group = torch.arange(1 << 17, dtype=torch.int64)
    depth = torch.rand(1 << 17)
    device_sort.stable_lexsort(pixel, group, depth)
    modes = [_unpack(call)["mode"] for call in recorded]
    keys = [_unpack(call)["src_key"] for call in recorded]
    # Depth first with nothing to compose, then each more significant key
    # gathered through the order built so far -- no torch index_select between.
    assert modes == [0, 2, 2]
    assert keys[0] is depth
    assert keys[1] is group
    assert keys[2] is pixel


def test_a_lexsort_is_all_or_nothing(monkeypatch):
    """A half-device chain would pay the very gathers it exists to remove."""
    monkeypatch.setattr(
        device_sort, "radix_sort_available", lambda keys: keys.dtype is torch.int64
    )
    wide = torch.arange(1 << 17, dtype=torch.int64)
    narrow = torch.arange(1 << 17, dtype=torch.int32)
    assert device_sort.stable_lexsort(wide, narrow) is None
    assert device_sort.stable_lexsort() is None
