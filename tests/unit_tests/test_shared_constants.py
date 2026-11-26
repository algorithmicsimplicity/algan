"""A script must not be able to change a shared constant for code that runs after it.

``BLUE`` and ``ORIGIN`` are one object each for the whole process, so an
in-place edit through an alias used to reach every later Mob -- and, on the
warm daemon, every later script: ``c = BLUE; c.glow = 2`` (which ``Color``'s
own docstring recommended) made every later ``BLUE`` glow, and
``o = ORIGIN; o += 1.0`` moved the origin. The two spellings are
``probes/lead/daemon_constant_leak_a.py`` from the 2026-11-26 audit.
"""

from __future__ import annotations

import pytest
import torch

import algan
from algan.constants import _shared, color, spatial
from algan.errors import AlganConfigurationError

# The two halves of the daemon probe: script A edits the constants through
# aliases, script B (the next script the same process serves) reads them.
_SCRIPT_A = """
from algan import *
c = BLUE
try:
    c.glow = 2
except AlganConfigurationError:
    pass
o = ORIGIN
o += 1.0
x = RED
x *= 0.5
"""

_SCRIPT_B = """
from algan import *
seen = (BLUE.clone(), ORIGIN.clone(), RED.clone())
"""


@pytest.fixture(autouse=True)
def _restore_constants():
    yield
    _shared.restore_shared_constants()


@pytest.mark.parametrize("channel", ["glow", "opacity", "rgb"])
def test_a_channel_setter_refuses_a_palette_constant(channel):
    before = color.BLUE.clone()
    c = color.BLUE
    value = (0.1, 0.2, 0.3) if channel == "rgb" else 0.5
    with pytest.raises(AlganConfigurationError, match=f"set_{channel}"):
        setattr(c, channel, value)
    assert torch.equal(color.BLUE, before)


def test_a_view_of_a_palette_constant_is_refused_too():
    alias = color.RED.as_subclass(color.Color)
    with pytest.raises(AlganConfigurationError):
        alias.glow = 2
    assert float(color.RED.glow) == 0.0


def test_the_documented_spellings_still_work_on_copies():
    glowing = color.BLUE.set_glow(2)
    assert float(glowing.glow) == 2.0
    assert float(color.BLUE.glow) == 0.0
    own = color.Color(color.BLUE)
    own.glow = 2
    own.opacity = 0.25
    assert float(own.glow) == 2.0
    assert float(own.opacity) == 0.25
    assert float(color.BLUE.glow) == 0.0
    assert float(color.BLUE.opacity) == 1.0


def test_restore_undoes_in_place_arithmetic_through_an_alias():
    blue, origin = color.BLUE.clone(), spatial.ORIGIN.clone()
    origin_object = spatial.ORIGIN
    o = spatial.ORIGIN
    o += 1.0
    x = color.BLUE
    x *= 0.5
    assert not torch.equal(spatial.ORIGIN, origin)
    _shared.restore_shared_constants()
    assert torch.equal(spatial.ORIGIN, origin)
    assert torch.equal(color.BLUE, blue)
    # In place: every module that bound the object sees the restored value.
    assert spatial.ORIGIN is origin_object
    assert algan.ORIGIN is origin_object


def test_restore_survives_a_replaced_storage():
    white = color.WHITE.clone()
    color.WHITE.data = torch.zeros(3)
    _shared.restore_shared_constants()
    assert torch.equal(color.WHITE, white)
    with pytest.raises(AlganConfigurationError):
        color.WHITE.glow = 1


def test_a_script_does_not_leak_constant_edits_into_the_next_one():
    """What the warm daemon sees: two scripts, one process, restore between."""
    exec(compile(_SCRIPT_A, "script_a", "exec"), {})
    _shared.restore_shared_constants()
    namespace = {}
    exec(compile(_SCRIPT_B, "script_b", "exec"), namespace)
    blue, origin, red = namespace["seen"]
    assert float(blue.glow) == 0.0
    assert torch.equal(origin, torch.zeros(3))
    assert torch.equal(red, color.Color("#FC6255"))


def test_every_direction_and_palette_entry_is_registered():
    registered = {id(live) for live, _ in _shared._PRISTINE}
    for module in (color, spatial):
        for name, value in vars(module).items():
            if name.isupper() and isinstance(value, torch.Tensor):
                assert id(value) in registered, name
