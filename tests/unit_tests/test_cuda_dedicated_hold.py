"""A Windows CUDA render holds torch's allocator inside dedicated memory.

A WDDM driver backs what the card cannot hold with shared system memory
instead of refusing it, and torch's caching allocator asks the driver for
more while freed blocks sit in its cache. A render job therefore limits the
allocator to a fixed share of the device (``memory_utils._CUDA_DEDICATED_HOLD``):
torch then frees its cache before it refuses, and refuses with the ordinary
out-of-memory error the batch loop answers by shrinking its window.

The limit itself is torch's, so these tests pin what Algan decides: where the
hold applies, what it asks torch for, that the out-of-arena budgets keep
reading the whole device, and that the job's end gives the device back.

No device is touched: ``torch.cuda.set_per_process_memory_fraction`` and the
render device are stand-ins.
"""

import sys

import pytest
import torch

from algan.render_loop import (
    _one_frame_does_not_fit_message,
    _render_device_pool_bytes,
)
from algan.settings import SETTINGS
from algan.utils import memory_utils

TOTAL = 16 << 30


class _Device:
    total_memory = TOTAL


@pytest.fixture
def cuda_render(monkeypatch):
    """A Windows process whose render device is the current CUDA device,
    named without an index as ``render_device()`` names it, with nothing
    live on it; yields the calls made to torch's limit.
    """
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _Device)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _index=None: 8 << 20)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("ALGAN_CUDA_DEDICATED_HOLD", raising=False)
    monkeypatch.setattr(memory_utils, "render_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device=None: calls.append((fraction, device)),
    )
    monkeypatch.setattr(memory_utils, "_CUDA_ALLOCATOR_HELD", None)
    return calls


def test_a_render_job_holds_the_allocator_and_gives_it_back(cuda_render):
    assert memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert cuda_render == [(memory_utils._CUDA_DEDICATED_HOLD, 0)]
    # A nested job changes nothing.
    assert memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert len(cuda_render) == 1

    memory_utils.release_cuda_allocator_hold()
    # torch has no "unset": the whole device is its default.
    assert cuda_render[-1] == (1.0, 0)
    memory_utils.release_cuda_allocator_hold()
    assert len(cuda_render) == 2


def test_a_process_holding_other_memory_is_not_held(cuda_render, monkeypatch):
    # The limit presumes the process's CUDA memory is the render's. Six
    # one-frame renders whose arenas a test kept alive reached 13.3 GiB live;
    # the seventh's arena, sized from what the driver had free, fit the card
    # and was refused by the limit. Such a render runs as it did unheld.
    share = memory_utils._CUDA_HOLD_FOREIGN_SHARE
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda _index=None: int(TOTAL * share) + 1
    )

    assert not memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert cuda_render == []
    assert not memory_utils.cuda_allocator_is_held()

    # At the share itself the render is still held.
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda _index=None: int(TOTAL * share)
    )
    assert memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert memory_utils.cuda_allocator_is_held()


def test_a_frame_that_does_not_fit_names_the_switch_while_held(cuda_render):
    # The one thing the hold can take away is a frame the driver would have
    # paged through; the error for it says how to get that back.
    assert "ALGAN_CUDA_DEDICATED_HOLD" not in _one_frame_does_not_fit_message()

    memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert "ALGAN_CUDA_DEDICATED_HOLD=0" in _one_frame_does_not_fit_message()

    memory_utils.release_cuda_allocator_hold()
    assert "ALGAN_CUDA_DEDICATED_HOLD" not in _one_frame_does_not_fit_message()


def test_an_indexed_render_device_is_held_on_its_own_index(cuda_render, monkeypatch):
    monkeypatch.setattr(memory_utils, "render_device", lambda: torch.device("cuda", 1))

    assert memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert cuda_render == [(memory_utils._CUDA_DEDICATED_HOLD, 1)]


def test_the_out_of_arena_pool_stays_the_whole_device(cuda_render, monkeypatch):
    # The merge's headroom and the preparation budget are shares of the pool,
    # and they size frame windows. Read off the limit they shrank every
    # memory-bound window, and 11 of a 30-scene video's clips differed from
    # the unheld render; the limit is the allocator's alone.
    total = TOTAL
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: (3 << 30, total), raising=False
    )
    monkeypatch.setattr(SETTINGS.computing, "available_memory_override", None)

    memory_utils.hold_cuda_allocator_to_dedicated_memory()

    assert _render_device_pool_bytes(torch.device("cuda", 0)) == total
    assert _render_device_pool_bytes(torch.device("cuda")) == total


def test_the_share_is_of_the_device_not_of_what_is_free(cuda_render):
    # A limit read off the memory free when the job starts would differ from
    # run to run with another process's use of the card.
    assert memory_utils.cuda_dedicated_hold_fraction() == pytest.approx(
        memory_utils._CUDA_DEDICATED_HOLD
    )
    assert 0.5 < memory_utils._CUDA_DEDICATED_HOLD < 1.0


@pytest.mark.parametrize("value", ["0", "1", "1.5", "-0.2"])
def test_the_switch_turns_the_hold_off(cuda_render, monkeypatch, value):
    monkeypatch.setenv("ALGAN_CUDA_DEDICATED_HOLD", value)

    assert not memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert cuda_render == []
    # Nothing was held, so there is nothing to give back.
    memory_utils.release_cuda_allocator_hold()
    assert cuda_render == []


def test_the_switch_sets_the_share(cuda_render, monkeypatch):
    monkeypatch.setenv("ALGAN_CUDA_DEDICATED_HOLD", "0.5")

    assert memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert cuda_render == [(0.5, 0)]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_a_driver_that_refuses_is_left_alone(cuda_render, monkeypatch, platform):
    # Off Windows the driver itself refuses what the card cannot hold.
    monkeypatch.setattr(sys, "platform", platform)

    assert not memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert cuda_render == []


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_only_a_cuda_render_is_held(cuda_render, monkeypatch, device):
    monkeypatch.setattr(memory_utils, "render_device", lambda: torch.device(device))

    assert not memory_utils.hold_cuda_allocator_to_dedicated_memory()
    assert cuda_render == []
