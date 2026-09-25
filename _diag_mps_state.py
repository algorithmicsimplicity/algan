"""Temporary MPS suite-lifetime telemetry for CI diagnosis."""
from __future__ import annotations

import dataclasses
import os

import psutil
import pytest
import torch

TARGET = "tests/unit_tests/test_updater_mob_creation.py::test_numeric_display_counts_inside_an_updater"


def _tensor_bytes(value, seen=None):
    if seen is None:
        seen = set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(_tensor_bytes(getattr(value, field.name), seen) for field in dataclasses.fields(value))
    if isinstance(value, dict):
        return sum(_tensor_bytes(k, seen) + _tensor_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(_tensor_bytes(item, seen) for item in value)
    return 0


def _emit(label):
    from algan.scene_manager import SceneManager
    from algan.mobs.surfaces import surface as surface_mod
    from algan.rendering import logical_pn
    from algan.rendering import mps_zero_copy
    from algan.rendering.raytracing import primitives, raster_pipeline

    proc = psutil.Process()
    manager = SceneManager.instance()
    stack = manager.scene_stack
    actors = sum(len(getattr(scene, "actors", ())) for scene in stack)
    effects = sum(len(getattr(scene, "effects", ())) for scene in stack)
    rss = proc.memory_info().rss
    avail = psutil.virtual_memory().available
    mps_alloc = torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else -1
    mps_driver = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else -1
    imports = mps_zero_copy.cache_stats()
    surf_cache = surface_mod._grid_triangle_indices_cache
    dice_cache = logical_pn._DICE_PATTERN_CACHE
    sample_cache = primitives._SAMPLE_TENSOR_CACHE
    edge_cache = raster_pipeline._EDGE_CACHE
    try:
        from algan.taichi_compat import kernel_specializations, submodule
        runtime = submodule("lang.impl").get_runtime()
        kernels = len(runtime.kernels)
        specs = sum(len(kernel_specializations(kernel)) for kernel in runtime.kernels)
    except Exception as exc:
        kernels = specs = -1
        print(f"DIAG_RUNTIME_ERROR {type(exc).__name__}: {exc}", flush=True)
    print(
        "DIAG_STATE"
        f" label={label}"
        f" rss={rss}"
        f" host_available={avail}"
        f" mps_alloc={mps_alloc}"
        f" mps_driver={mps_driver}"
        f" scene_stack={len(stack)}"
        f" scene_actors={actors}"
        f" scene_effects={effects}"
        f" imports_entries={imports[0]}"
        f" imports_storages={imports[1]}"
        f" imports_bytes={imports[2]}"
        f" surface_entries={len(surf_cache)}"
        f" surface_bytes={_tensor_bytes(surf_cache)}"
        f" dice_entries={len(dice_cache)}"
        f" dice_bytes={_tensor_bytes(dice_cache)}"
        f" sample_entries={len(sample_cache)}"
        f" sample_bytes={_tensor_bytes(sample_cache)}"
        f" edge_entries={len(edge_cache)}"
        f" edge_bytes={_tensor_bytes(edge_cache)}"
        f" kernels={kernels}"
        f" specializations={specs}",
        flush=True,
    )


def pytest_collection_modifyitems(config, items):
    limit = int(os.environ["DIAG_PREFIX_LIMIT"])
    target = next(item for item in items if item.nodeid == TARGET)
    kept = list(items[:limit])
    if target not in kept:
        kept.append(target)
    items[:] = kept
    print(f"DIAG_COLLECTION limit={limit} selected={len(items)} target={target.nodeid}", flush=True)


def pytest_runtest_setup(item):
    if item.nodeid == TARGET:
        _emit("before_target")


def pytest_sessionfinish(session, exitstatus):
    _emit(f"session_end_status_{exitstatus}")
