"""Select an execution prefix and record state after pytest fixture teardown."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

TARGETS = (
    "test_lifted_path_tracer_features_render",
    "test_capture_is_off_unless_armed",
    "test_a_ray_exactly_on_a_shared_edge_hits_exactly_one_neighbour",
    "test_ray_reorder_preserves_other_views_of_the_same_arena",
)
_previous = None


def pytest_collection_modifyitems(config, items):
    start = os.environ.get("PR144_START", "test_a")
    end = os.environ.get("PR144_END", "test_s")
    selected = [
        item for item in items
        if ("/unit_tests/" in str(item.fspath)
            and start <= Path(str(item.fspath)).name < end)
        or any(target in item.nodeid for target in TARGETS)
    ]
    config.hook.pytest_deselected(items=[item for item in items if item not in selected])
    items[:] = selected


def snapshot():
    from algan.taichi_compat import program
    from algan.settings import SETTINGS

    prog = program()
    cfg = prog.config() if prog is not None else None
    values = {
        name: repr(getattr(cfg, name)) for name in dir(cfg)
        if not name.startswith("_") and not callable(getattr(cfg, name))
    }
    values["env"] = repr(sorted(
        (name, value) for name, value in os.environ.items()
        if name.startswith(("ALGAN_", "QD_", "TI_"))
    ))
    values["settings"] = repr(SETTINGS.raytracing.to_dict())
    return values


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _previous
    yield
    try:
        values = snapshot()
        if values != _previous:
            diff = values if _previous is None else {
                name: (value, _previous.get(name)) for name, value in values.items()
                if value != _previous.get(name)
            }
            output = Path("algan_outputs")
            output.mkdir(exist_ok=True)
            with (output / "state-trace.jsonl").open("a") as stream:
                stream.write(json.dumps({"after": item.nodeid, "changes": diff}) + "\n")
            _previous = values
    except Exception as error:
        print("DIAGNOSTIC TELEMETRY ERROR", item.nodeid, repr(error), flush=True)
