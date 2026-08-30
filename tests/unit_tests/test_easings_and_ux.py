"""Unit tests for newly added easings, color parsing, exception hierarchy, CLI, and RenderResult reprs."""

import tempfile
from pathlib import Path

import pytest
import torch

from algan.cli import main as cli_main
from algan.constants import easings
from algan.constants.color import Color
from algan.errors import (
    AlganError,
    AudioTranscriptMismatchError,
    InvalidColorError,
    ModifiedProtectedAttributeError,
)
from algan.utils.algan_utils import RenderResult

# ---------------------------------------------------------------------------
# Rate Functions Tests
# ---------------------------------------------------------------------------


def test_easings_endpoints():
    """All easing curves should map 0 -> ~0 and 1 -> ~1 (within floating point precision)."""
    funcs = [
        easings.identity,
        easings.linear,
        easings.smooth,
        easings.ease_in_sine,
        easings.ease_out_sine,
        easings.ease_in_out_sine,
        easings.ease_in_quad,
        easings.ease_out_quad,
        easings.ease_in_out_quad,
        easings.ease_in_cubic,
        easings.ease_out_cubic,
        easings.ease_in_out_cubic,
        easings.ease_in_quart,
        easings.ease_out_quart,
        easings.ease_in_out_quart,
        easings.ease_in_quint,
        easings.ease_out_quint,
        easings.ease_out_quintic,
        easings.ease_in_out_quint,
        easings.ease_in_expo,
        easings.ease_out_expo,
        easings.ease_in_circ,
        easings.ease_out_circ,
        easings.ease_in_out_circ,
        easings.ease_in_back,
        easings.ease_out_back,
        easings.ease_in_out_back,
        easings.ease_in_elastic,
        easings.ease_out_elastic,
        easings.ease_in_out_elastic,
        easings.ease_in_bounce,
        easings.ease_out_bounce,
        easings.ease_in_out_bounce,
    ]

    t_0 = torch.tensor(0.0)
    t_1 = torch.tensor(1.0)

    for fn in funcs:
        y0 = float(fn(t_0))
        y1 = float(fn(t_1))
        assert abs(y0) < 1e-4, f"{fn.__name__}(0) was {y0}, expected ~0"
        assert abs(y1 - 1.0) < 1e-4, f"{fn.__name__}(1) was {y1}, expected ~1"


def test_easings_tensor_broadcast():
    """Rate functions should seamlessly process 1D/2D tensor inputs."""
    t = torch.linspace(0, 1, 11)
    for name, fn in inspect_funcs(easings):
        out = fn(t)
        assert isinstance(out, torch.Tensor), f"{name} did not return a Tensor"
        assert out.shape == t.shape, f"{name} shape mismatch"


def inspect_funcs(module):
    return [
        (name, getattr(module, name))
        for name in dir(module)
        if callable(getattr(module, name))
        and not name.startswith("_")
        and name not in ("inversed", "tan", "delay_fade", "pulse_fade")
    ]


# ---------------------------------------------------------------------------
# Color Parsing Tests
# ---------------------------------------------------------------------------


def test_color_hex_variations():
    # 6-hex
    c6 = Color("#FF0000")
    assert torch.allclose(c6[:3], torch.tensor([1.0, 0.0, 0.0]))
    assert c6.opacity.item() == 1.0

    # 3-hex (#RGB -> #RRGGBB)
    c3 = Color("#F00")
    assert torch.allclose(c3[:3], torch.tensor([1.0, 0.0, 0.0]))
    assert c3.opacity.item() == 1.0

    # 8-hex (#RRGGBBAA)
    c8 = Color("#FF000080")
    assert torch.allclose(c8[:3], torch.tensor([1.0, 0.0, 0.0]))
    assert abs(c8.opacity.item() - 0x80 / 255.0) < 1e-5

    # 4-hex (#RGBA -> #RRGGBBAA)
    c4 = Color("#F008")
    assert torch.allclose(c4[:3], torch.tensor([1.0, 0.0, 0.0]))
    assert abs(c4.opacity.item() - 0x88 / 255.0) < 1e-5

    # CSS color names
    assert torch.allclose(Color("red")[:3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(Color("white")[:3], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(Color("black")[:3], torch.tensor([0.0, 0.0, 0.0]))
    assert Color("transparent").opacity.item() == 0.0


def test_invalid_color_raises_actionable_error():
    with pytest.raises(InvalidColorError) as exc_info:
        Color("invalid_color_123")
    assert "Invalid color string" in str(exc_info.value)

    with pytest.raises(InvalidColorError):
        Color("#12345")  # invalid 5-digit hex


# ---------------------------------------------------------------------------
# Exception Taxonomy Tests
# ---------------------------------------------------------------------------


def test_exception_taxonomy():
    assert issubclass(ModifiedProtectedAttributeError, AlganError)
    assert issubclass(AudioTranscriptMismatchError, AlganError)
    assert issubclass(InvalidColorError, AlganError)


# ---------------------------------------------------------------------------
# RenderResult Jupyter Repr Tests
# ---------------------------------------------------------------------------


def test_render_result_repr():
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4_path = Path(tmpdir) / "test.mp4"
        mp4_path.write_bytes(b"dummy video bytes")

        res = RenderResult(
            status="rendered", output_path=mp4_path, duration_seconds=5.0
        )
        html = res._repr_html_()
        assert "<video controls" in html
        assert "test.mp4" in html

        png_path = Path(tmpdir) / "test.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake png data")
        res_png = RenderResult(status="rendered", output_path=png_path)
        assert "<img src=" in res_png._repr_html_()
        assert res_png._repr_png_() == b"\x89PNG\r\n\x1a\nfake png data"


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


def test_cli_commands():
    # Check command returns 0
    assert cli_main(["check"]) == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        new_file = Path(tmpdir) / "test_scene.py"
        assert cli_main(["new", str(new_file)]) == 0
        assert new_file.exists()
        assert "from algan import *" in new_file.read_text(encoding="utf-8")
