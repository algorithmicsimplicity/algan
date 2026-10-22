"""The batching loop's device syncs and its per-chunk background upload.

On CUDA the loop no longer drains the devices at every batch boundary
(``_batch_boundary_sync``; ``ALGAN_OPT_DISABLE=loopsync`` restores it), and a
tensor background is uploaded and prepared once per render job rather than once
per chunk (``_prepare_background_for_chunk``'s ``cache``;
``ALGAN_OPT_DISABLE=bgupload`` restores the per-chunk upload). Both are
scheduling changes, so every arm must hand the renderer the same values.

The CUDA arms are reached here on the CPU by faking the device the loop asks
about; the one test that needs a real device skips without one.
"""

from __future__ import annotations

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
import algan.render_loop as render_loop_module
from algan.render_loop import (
    _background_reuse_active,
    _background_to_device,
    _batch_boundary_sync,
    _prepare_background_for_chunk,
)
from algan.rendering.raytracing.scene_builder import _StaticImageBackground
from tests.unit_tests.test_render_batch_sizing import _make_preflight_scene


def _disable(monkeypatch, *names):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset(names))


def _count_syncs(monkeypatch):
    calls = []
    monkeypatch.setattr(render_loop_module, "_sync_devices", lambda: calls.append(1))
    return calls


@pytest.mark.parametrize(
    ("device", "disabled", "expected"),
    [
        ("cuda", (), 0),
        ("cuda", ("loopsync",), 1),
        ("cpu", (), 1),
        ("mps", (), 1),
    ],
)
def test_batch_boundary_sync_is_skipped_on_cuda_only(
    monkeypatch, device, disabled, expected
):
    _disable(monkeypatch, *disabled)
    monkeypatch.setattr(
        render_loop_module, "render_device", lambda: torch.device(device)
    )
    calls = _count_syncs(monkeypatch)

    _batch_boundary_sync()

    assert len(calls) == expected


@pytest.mark.parametrize(
    ("device", "disabled", "expected"),
    [
        ("cuda", (), 0),
        ("cuda", ("loopsync",), 9),
        ("cpu", (), 9),
    ],
)
def test_the_batching_loop_syncs_only_where_the_device_needs_it(
    monkeypatch, device, disabled, expected
):
    # Three batches of two frames. Each batch used to sync at the top of its
    # iteration, after its fetch and after its render.
    rendered = []
    scene = _make_preflight_scene(monkeypatch, lambda *_args: True, rendered)
    original = scene._get_batch_of_primitives
    scene._get_batch_of_primitives = lambda start, end, *args: original(
        start, min(end, start + 2), *args
    )
    _disable(monkeypatch, *disabled)
    monkeypatch.setattr(
        render_loop_module, "render_device", lambda: torch.device(device)
    )
    calls = _count_syncs(monkeypatch)

    frames = list(scene.get_frames(0, 6, post_processes=(), manual_memory=False))

    assert rendered == [2, 2, 2]
    assert [len(batch) for batch in frames] == rendered
    assert len(calls) == expected


def test_background_reuse_is_cuda_only_and_has_a_kill_switch(monkeypatch):
    _disable(monkeypatch)
    assert _background_reuse_active(torch.device("cuda"))
    assert _background_reuse_active("cuda:0")
    assert not _background_reuse_active(torch.device("cpu"))
    assert not _background_reuse_active(torch.device("mps"))
    _disable(monkeypatch, "bgupload")
    assert not _background_reuse_active(torch.device("cuda"))


def test_a_cpu_target_is_never_staged(monkeypatch):
    _disable(monkeypatch)
    source = torch.rand(5)
    moved = _background_to_device(source, torch.device("cpu"))
    assert moved is source


def _image(frames=1):
    generator = torch.Generator().manual_seed(7)
    image = torch.rand(frames, 4, 6, 5, generator=generator)
    image[0, 0, 0] = 0.0
    image[0, -1, -1] = 1.0
    return image


BACKGROUNDS = {
    "color": lambda: torch.tensor([0.1, 0.2, 0.3, 0.0, 1.0]),
    "image": _image,
    "rows": lambda: _image().reshape(-1, 5),
}


def _fake_cuda(monkeypatch, *, reuse=True):
    """Take the CUDA arm on the CPU and count the uploads it makes.

    Each upload is a new tensor, as a copy to another device is.
    """
    uploads = []

    def upload(background, device):
        uploads.append(background)
        return background.to(device, copy=True)

    monkeypatch.setattr(
        render_loop_module, "_background_reuse_active", lambda _d: reuse
    )
    monkeypatch.setattr(render_loop_module, "_background_to_device", upload)
    return uploads


def _chunk(background, start, end, cache):
    return _prepare_background_for_chunk(
        background,
        screen_width=6,
        screen_height=4,
        anti_alias_level=1,
        current_ind=start,
        new_ind=end,
        frames_per_second=1,
        device=torch.device("cpu"),
        cache=cache,
    )


def _unwrap(prepared):
    if isinstance(prepared, _StaticImageBackground):
        return prepared.rows, prepared.num_frames
    return prepared, None


CHUNKS = ((0, 3), (3, 4), (4, 9))


@pytest.mark.parametrize("kind", sorted(BACKGROUNDS))
def test_a_tensor_background_is_uploaded_once_per_job(monkeypatch, kind):
    _disable(monkeypatch)
    background = BACKGROUNDS[kind]()
    expected = [_unwrap(_chunk(background, *chunk, None)) for chunk in CHUNKS]
    uploads = _fake_cuda(monkeypatch)
    cache = {}

    actual = [_unwrap(_chunk(background, *chunk, cache)) for chunk in CHUNKS]

    assert len(uploads) == 1
    for (rows, frames), (expected_rows, expected_frames) in zip(actual, expected):
        assert torch.equal(rows, expected_rows)
        assert rows.dtype == expected_rows.dtype
        assert frames == expected_frames
    # A still image keeps its one frame's rows, re-wrapped per chunk length.
    if kind == "image":
        assert [frames for _rows, frames in actual] == [3, 1, 5]
        assert actual[0][0] is actual[2][0]


def test_a_background_written_in_place_is_prepared_afresh(monkeypatch):
    _disable(monkeypatch)
    uploads = _fake_cuda(monkeypatch)
    cache = {}
    background = torch.tensor([0.1, 0.2, 0.3, 0.0, 1.0])
    first = _chunk(background, 0, 2, cache)
    assert _chunk(background, 2, 4, cache) is first

    background[0] = 0.9
    after = _chunk(background, 4, 6, cache)

    assert len(uploads) == 2
    assert torch.equal(after, background)
    assert float(first[0]) == pytest.approx(0.1)
    assert _chunk(background, 6, 8, cache) is after


def test_a_different_background_object_misses(monkeypatch):
    _disable(monkeypatch)
    uploads = _fake_cuda(monkeypatch)
    cache = {}
    _chunk(torch.tensor([0.1, 0.2, 0.3]), 0, 2, cache)
    _chunk(torch.tensor([0.1, 0.2, 0.3]), 2, 4, cache)
    assert len(uploads) == 2


@pytest.mark.parametrize(
    "arm", ["no cache", "reuse off", "staticbg off", "over the cap", "inference"]
)
def test_arms_that_prepare_every_chunk(monkeypatch, arm):
    _disable(monkeypatch, *(("staticbg",) if arm == "staticbg off" else ()))
    if arm == "over the cap":
        monkeypatch.setattr(render_loop_module, "_BACKGROUND_CACHE_MAX_BYTES", 16)
    background = _image()
    if arm == "inference":
        with torch.inference_mode():
            background = _image()
    expected = [_unwrap(_chunk(background, *chunk, None)) for chunk in CHUNKS]
    uploads = _fake_cuda(monkeypatch, reuse=arm != "reuse off")
    cache = None if arm == "no cache" else {}

    actual = [_unwrap(_chunk(background, *chunk, cache)) for chunk in CHUNKS]

    assert len(uploads) == len(CHUNKS)
    assert not cache
    for (rows, frames), (expected_rows, expected_frames) in zip(actual, expected):
        assert torch.equal(rows, expected_rows)
        assert frames == expected_frames
    if arm == "staticbg off":
        # The per-frame copies are as long as their chunk.
        assert [rows.shape[0] for rows, _ in actual] == [
            1 + (end - start) * 24 for start, end in CHUNKS
        ]


def test_a_callable_background_is_never_cached(monkeypatch):
    _disable(monkeypatch)
    uploads = _fake_cuda(monkeypatch)
    cache = {}
    _chunk(lambda x, y, time: x + y + time, 0, 2, cache)
    assert uploads == []
    assert cache == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("kind", sorted(BACKGROUNDS))
def test_staged_uploads_match_the_blocking_copy_on_cuda(monkeypatch, kind):
    _disable(monkeypatch)
    device = torch.device("cuda")
    background = BACKGROUNDS[kind]()
    staged = _background_to_device(background, device)
    assert staged.device.type == "cuda"
    assert staged.dtype == background.dtype
    assert staged.stride() == background.to(device).stride()
    assert torch.equal(staged.cpu(), background)

    cache = {}
    prepared = [_unwrap(_chunk_on(background, *chunk, cache)) for chunk in CHUNKS]
    _disable(monkeypatch, "bgupload")
    blocking = [_unwrap(_chunk_on(background, *chunk, None)) for chunk in CHUNKS]
    for (rows, frames), (expected_rows, expected_frames) in zip(prepared, blocking):
        assert torch.equal(rows.cpu(), expected_rows.cpu())
        assert frames == expected_frames


def _chunk_on(background, start, end, cache):
    return _prepare_background_for_chunk(
        background,
        screen_width=6,
        screen_height=4,
        anti_alias_level=1,
        current_ind=start,
        new_ind=end,
        frames_per_second=1,
        device=torch.device("cuda"),
        cache=cache,
    )


@pytest.mark.parametrize("exit_path", ["complete", "error", "close"])
def test_the_job_owns_the_background_cache(monkeypatch, exit_path):
    scene = _make_preflight_scene(monkeypatch, lambda *_args: True, [])
    monkeypatch.setattr(render_loop_module, "ensure_taichi_for_render", lambda: None)
    seen = []
    render = scene._render_primitive_batch

    def record(*args, **kwargs):
        seen.append(scene.__dict__.get("_chunk_background_cache"))
        if exit_path == "error":
            raise ValueError("render failed")
        yield from render(*args, **kwargs)

    monkeypatch.setattr(scene, "_render_primitive_batch", record)
    frames = scene.get_frames(0, 3, post_processes=(), manual_memory=False)
    if exit_path == "error":
        with pytest.raises(ValueError, match="render failed"):
            list(frames)
    elif exit_path == "close":
        next(frames)
        frames.close()
    else:
        list(frames)
    assert seen
    assert all(cache == {} for cache in seen)
    assert "_chunk_background_cache" not in scene.__dict__


def test_background_only_frames_match_with_the_cache(monkeypatch):
    # A Scene with nothing spawned renders through _render_background_batch.
    # One frame per chunk, so the cached arm reuses its upload four times.
    from algan import Scene
    from algan.rendering import memory_model

    class OneFramePerChunk(memory_model.ChunkMemoryModel):
        def plan(self, signature, remaining, bytes_remaining):
            return 1

    monkeypatch.setattr(render_loop_module, "ChunkMemoryModel", OneFramePerChunk)
    _disable(monkeypatch)

    def render(reuse):
        with Scene() as scene:
            scene.set_background(torch.tensor([0.2, 0.4, 0.6]))
            with monkeypatch.context() as patched:
                uploads = _fake_cuda(patched, reuse=reuse)
                frames = torch.cat(
                    [
                        batch.clone()
                        for batch in scene.get_frames(0, 5, post_processes=())
                    ]
                )
        return frames, len(uploads)

    cached, cached_uploads = render(True)
    plain, plain_uploads = render(False)

    assert cached_uploads == 1
    assert plain_uploads == 5
    assert torch.equal(cached, plain)
