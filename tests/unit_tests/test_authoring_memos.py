"""Authoring-time memos must reproduce the uncached construction exactly.

Covers the Tex/Text glyph memo, the circuit-frame memo, the offset-based rows
of a batched Mob's view, and the lazily opened speech clip.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

import algan.mobs.bezier_circuit as bezier_circuit
import algan.mobs.text as text_module
from algan import BLUE, GREEN, RED, DecimalNumber, SceneManager, Tex, Text
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.text import mn


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _state(mob):
    """Every animatable tensor and segment table of ``mob``'s subtree."""
    rows = []
    for part in [mob, *mob.get_descendants()]:
        for attr in sorted(part.animatable_attrs):
            try:
                value = getattr(part, attr)
            except AttributeError:
                continue
            if torch.is_tensor(value):
                rows.append((type(part).__name__, attr, value.detach().clone()))
        for extra in ("num_mobs_per_segment", "_matching_tex_keys"):
            if hasattr(part, extra):
                rows.append((type(part).__name__, extra, getattr(part, extra)))
    return rows


def _assert_same_state(a, b):
    assert len(a) == len(b)
    for (kind_a, attr_a, value_a), (kind_b, attr_b, value_b) in zip(a, b):
        assert (kind_a, attr_a) == (kind_b, attr_b)
        if torch.is_tensor(value_a):
            assert value_a.shape == value_b.shape, attr_a
            assert torch.equal(value_a, value_b), attr_a
        else:
            assert value_a == value_b, attr_a


def _builders():
    builders = [
        lambda: Tex(r"w_{\rm new}", "=", r"w_{\rm old}", "-", r"\eta", color=RED),
        lambda: Tex("{{a}} + {{b}}"),
        lambda: DecimalNumber(-12.25, decimal_places=3),
    ]
    if hasattr(mn, "Text"):
        builders += [
            lambda: Text("for (int i = 0; i < n; ++i)", color=BLUE),
            lambda: Text("red green", color_map={"green": GREEN}, weight="BOLD"),
        ]
    return builders


@pytest.mark.parametrize("index", range(5))
def test_memoized_text_matches_a_fresh_typeset(monkeypatch, index):
    builders = _builders()
    if index >= len(builders):
        pytest.skip("needs manimpango")
    build = builders[index]
    text_module._TEX_GLYPH_MEMO.clear()
    bezier_circuit._FRAME_MEMO.clear()
    with monkeypatch.context() as patch:
        patch.setattr(text_module, "_tex_glyph_memo_key", lambda *a, **k: None)
        patch.setattr(bezier_circuit, "_frame_memo_key", lambda points: None)
        fresh = _state(build())
    build()  # fills both memos
    assert text_module._TEX_GLYPH_MEMO
    _assert_same_state(fresh, _state(build()))


def test_tex_memo_key_follows_the_template():
    template = mn.config["tex_template"]
    before = text_module._tex_glyph_memo_key(("x",), " ", None, True, None, {})
    original = template.post_doc_commands
    try:
        template.post_doc_commands = original + r"\boldmath"
        after = text_module._tex_glyph_memo_key(("x",), " ", None, True, None, {})
    finally:
        template.post_doc_commands = original
    assert before != after
    assert text_module._tex_glyph_memo_key(("x",), " ", None, True, None, {}) == before
    # An explicit template is never memoized.
    assert (
        text_module._tex_glyph_memo_key(
            ("x",), " ", None, True, None, {"tex_template": template}
        )
        is None
    )


def test_circuit_frame_memo_hands_out_copies():
    bezier_circuit._FRAME_MEMO.clear()
    square = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    points = torch.cat(
        [
            torch.stack([a * (1 - t) + b * t for t in torch.linspace(0, 1, 4)])
            for a, b in zip(square, square.roll(-1, 0))
        ]
    )
    locations, bases, synthesized = bezier_circuit._circuit_frames([points])
    reference = (locations[0], bases[0], bool(synthesized[0]))
    first = bezier_circuit._circuit_location_and_basis(points)
    first[0].add_(100.0)
    first[1].add_(100.0)
    second = bezier_circuit._circuit_location_and_basis(points.clone())
    for expected, actual in zip(reference, second):
        if torch.is_tensor(expected):
            assert torch.equal(expected, actual)
        else:
            assert expected == actual


def test_batched_view_rows_match_a_split_of_the_whole_batch():
    counts = [4, 8, 4, 12, 4, 8]
    batches = [
        torch.randn(count, 3, generator=torch.Generator().manual_seed(i))
        for i, count in enumerate(counts)
    ]
    pack = BezierCircuitCubic.from_batches(batches, add_to_scene=False)
    sizes = pack.control_points.parent_batch_sizes
    total = int(sizes.sum())
    reference = torch.arange(total).split(sizes.tolist())
    for item in (0, 3, -1, slice(1, 4), slice(None, None, 2), slice(-3, None)):
        view = pack.control_points.clone(
            add_to_scene=False, clone_data=False, recursive=True
        )
        view._set_data_sub_inds([item] if isinstance(item, int) else item)
        chosen = reference[item] if isinstance(item, slice) else [reference[item]]
        assert torch.equal(view.data_sub_inds, torch.cat(list(chosen)))


def test_speech_clip_reads_its_duration_from_the_sidecar(tmp_path):
    moviepy = pytest.importorskip("moviepy")
    import numpy as np

    from algan.utils import audio_utils

    path = tmp_path / "tone.wav"

    def write(seconds):
        clip = moviepy.AudioArrayClip(
            np.sin(np.linspace(0, 400 * math.pi, int(44100 * seconds)))
            .reshape(-1, 1)
            .repeat(2, 1)
            * 0.2,
            fps=44100,
        )
        clip.write_audiofile(str(path), fps=44100, logger=None)

    write(3.0)
    probed = audio_utils._cached_audio_file_clip(str(path))
    lazy = audio_utils._cached_audio_file_clip(str(path))
    assert type(lazy).__name__ == "LazyAudioFileClip"
    assert lazy._reader is None
    assert lazy.duration == probed.duration
    assert lazy.nchannels == probed.nchannels
    assert np.array_equal(
        lazy.to_soundarray(fps=44100), probed.to_soundarray(fps=44100)
    )
    lazy.close()
    probed.close()

    # A rewritten file is probed again rather than trusting the old sidecar.
    write(4.0)
    os.utime(path, ns=(os.stat(path).st_atime_ns, os.stat(path).st_mtime_ns + 10**9))
    reprobed = audio_utils._cached_audio_file_clip(str(path))
    assert type(reprobed).__name__ != "LazyAudioFileClip"
    assert reprobed.duration > probed.duration
    reprobed.close()
