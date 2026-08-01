"""The shipped memory tables must still describe what the engine does.

These are the staleness guards. If someone adds, removes or resizes an arena
allocation and does not re-run the calibration generator, the shipped table
silently under-predicts and the failure surfaces much later as an
out-of-memory error in somebody's render. Re-deriving the measurement here
turns that into a failing test.

Only the post-processing scopes are covered at this tier, because they can be
driven without a renderer or a GPU. Scopes that need a real render are guarded
by the calibration generator's own verification (``--verify``).
"""

import pytest
import torch

from algan.rendering import mem_usage
from algan.rendering.mem_usage_lookup import (
    density_seed,
    get_post_process_memory_required,
    postprocess_key,
    unit_bytes,
)
from algan.rendering.mem_usage_runtime import peak_bytes
from algan.rendering.post_processing.post_process import post_process_frames
from algan.rendering.raytracing import settings as rt_settings
from algan.utils.calibrate_memory import (
    CalibrationError,
    build_tables,
    schema_fingerprint,
)
from algan.utils.memory_utils import ManualMemory

_DTYPES = {
    "torch.float32": torch.float32,
    "torch.uint8": torch.uint8,
    "torch.float16": torch.float16,
}


def _arena(num_bytes=256 << 20):
    return ManualMemory(
        0, device=torch.device("cpu"), managed=True, num_bytes=num_bytes)


def _plain_keys(limit=4):
    """Shipped post-process keys with an empty chain, smallest first.

    An empty chain means the configuration can be reconstructed from the key
    alone -- no user callable to rebuild -- so the test adapts automatically as
    the corpus changes rather than hard-coding a configuration.
    """
    keys = [
        key for key in mem_usage.TRACES.get("postprocess", {})
        if dict(key).get("chain") == ""
    ]
    keys.sort(key=lambda key: dict(key)["height"] * dict(key)["width"])
    return keys[:limit]


def _frames_for(key, count):
    fields = dict(key)
    dtype = _DTYPES[fields["dtype"]]
    shape = (count, fields["height"], fields["width"], fields["channels"])
    if dtype == torch.uint8:
        frames = torch.full(shape, 128, dtype=dtype)
        frames[..., 3] = 100
    else:
        frames = torch.full(shape, 0.5, dtype=dtype)
        frames[..., 3] = 0.7
    return frames


def _apply_key_settings(key, monkeypatch):
    fields = dict(key)
    monkeypatch.setattr(
        rt_settings, "POST_PROCESS_TONEMAP", bool(fields["tonemap"]))
    monkeypatch.setattr(
        rt_settings, "TONEMAPPING", bool(fields["tonemapping"]))
    monkeypatch.setattr(
        rt_settings, "TONEMAP_METHOD", fields["tonemap_method"])
    return fields


@pytest.mark.skipif(not _plain_keys(), reason="no chainless keys shipped")
def test_shipped_traces_predict_the_arena_exactly(monkeypatch):
    # The strongest form of the guard: replay the *shipped* trace and compare
    # against what the real pipeline actually allocates.
    for key in _plain_keys():
        fields = _apply_key_settings(key, monkeypatch)
        trace = mem_usage.TRACES["postprocess"][key]
        for count in (1, 2):
            frames = _frames_for(key, count)
            memory = _arena()
            post_process_frames(
                memory, frames, fields["anti_alias_level"],
                post_processes=(), apply_fxaa=bool(fields["fxaa"]))
            assert memory.max_pointer == peak_bytes(trace, count), (
                f"shipped trace disagrees with the arena for {key} at "
                f"{count} frames -- re-run "
                f"'python -m algan.utils.calibrate_memory'")


@pytest.mark.skipif(not _plain_keys(), reason="no chainless keys shipped")
def test_lookup_resolves_shipped_keys_without_probing(monkeypatch):
    for key in _plain_keys(2):
        fields = _apply_key_settings(key, monkeypatch)
        shape = (3, fields["height"], fields["width"], fields["channels"])
        # memory=None means a probe is impossible, so a non-None answer can
        # only have come from the table.
        predicted = get_post_process_memory_required(
            shape, _DTYPES[fields["dtype"]], fields["anti_alias_level"], (),
            bool(fields["fxaa"]), device=torch.device("cpu"), memory=None,
            tonemap_enabled=bool(fields["tonemap"]),
            tonemapping=bool(fields["tonemapping"]),
            tonemap_method=fields["tonemap_method"],
            tonemap_kernel=bool(fields["tonemap_kernel"]))
        assert predicted is not None


def test_unseen_configuration_is_probed_not_refused():
    # A resolution no corpus covers must still be sizable; this is what
    # replaced the old "attach an algan_memory_planner" hard failure.
    shape = (2, 37, 53, 4)
    key = postprocess_key(
        frame_shape=shape, frame_dtype=torch.float32, anti_alias_level=1,
        post_processes=(), apply_fxaa=False, tonemap_enabled=False,
        tonemapping=True, tonemap_method="neutral", tonemap_kernel=False)
    assert key not in mem_usage.TRACES.get("postprocess", {})
    predicted = get_post_process_memory_required(
        shape, torch.float32, 1, (), False,
        device=torch.device("cpu"), memory=_arena(),
        tonemap_enabled=False, tonemapping=True, tonemap_method="neutral",
        tonemap_kernel=False)
    assert predicted is not None and predicted >= 0


def test_unit_bytes_sums_coefficients_fixed_and_alignment_slack():
    entry = mem_usage.UNITS["wavefront_state"]
    key = dict(next(iter(entry)))
    coefficients = next(iter(entry.values()))
    predicted = unit_bytes("wavefront_state", key, pool=10, primary=4)
    assert predicted == (
        coefficients["pool"] * 10
        + coefficients["primary"] * 4
        + coefficients["fixed"]
        + coefficients["align_slack"])


def test_unit_bytes_returns_none_for_an_unknown_route():
    assert unit_bytes("wavefront_state", {"global_hits": 99}) is None


def test_density_seed_is_raised_by_the_safety_factor():
    seeded = density_seed("sparse_discovery")
    raw = mem_usage.DENSITIES["sparse_discovery"]["density"]
    assert seeded == pytest.approx(raw * mem_usage.DENSITY_SAFETY)
    # Seeded above zero is the point: the learner used to start at 0.0, so the
    # first chunk of every render over-committed and relied on OOM halving.
    assert seeded > 0.0


def test_shipped_fingerprint_matches_the_shipped_tables():
    # Catches a table edited by hand, or a partial regeneration.
    rebuilt = {
        "traces": mem_usage.TRACES,
        "units": {
            scope: {key: {"coefficients": coefficients,
                          "regressors": [name for name in coefficients
                                         if name != "align_slack"]}
                    for key, coefficients in entries.items()}
            for scope, entries in mem_usage.UNITS.items()
        },
        "densities": {
            scope: {
                "density_of": entry["density_of"],
                "density_per": entry["density_per"],
                "structural": {
                    key: {"coefficients": coefficients,
                          "regressors": [name for name in coefficients
                                         if name != "align_slack"]}
                    for key, coefficients in entry["structural"].items()
                },
            }
            for scope, entry in mem_usage.DENSITIES.items()
        },
    }
    assert schema_fingerprint(rebuilt) == mem_usage.SCHEMA_FINGERPRINT


def test_generator_rejects_a_scope_it_has_no_model_for():
    # A newly annotated scope must be given a model deliberately; it cannot be
    # silently dropped from the tables.
    class _Fake:
        scope = "brand_new_scope"
        params = {}
        events = []
        peak_forward = 0
        peak_reverse = 0
        alloc_count = 0
        entry_forward = 0
        source = "test"

    with pytest.raises(CalibrationError, match="unmodelled scope"):
        build_tables([_Fake()], [])
