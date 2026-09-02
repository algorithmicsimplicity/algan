"""The ``torch.compile`` switch and the decorator every pipeline function uses.

The wrapper is exercised against Dynamo's ``eager`` backend -- tracing with no
code generation -- so these tests cost milliseconds rather than an Inductor
build, and against a backend that raises, so the fallback contract is pinned:
a compile failure warns once and runs the function eagerly, an error the
function itself raises propagates untouched, and the switch is read live.
"""

from __future__ import annotations

import sys
import warnings

import pytest
import torch

from algan import SETTINGS, AlganConfigurationError, AlganWarning
from algan.utils import torch_compile as tc

# In the fast suite: the setting is read at every call of every decorated
# pipeline function, so a change to how the section validates or restores it
# reaches the whole render path at once.
pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def _eager_backend(monkeypatch):
    """Dynamo tracing only, and a clean slate for every test."""
    monkeypatch.setattr(tc, "_BACKEND", "eager")
    tc.reset_compiled_functions()
    yield
    tc.reset_compiled_functions()


def _wrapped_state(fn):
    return fn._algan_compiled


def test_the_setting_is_a_tristate_that_rejects_anything_else():
    assert SETTINGS.computing.torch_compile == "auto"
    for spelling, expected in (
        ("1", True),
        ("off", False),
        (" Auto ", "auto"),
        (True, True),
        (False, False),
    ):
        SETTINGS.computing.set(torch_compile=spelling)
        assert SETTINGS.computing.torch_compile == expected
    with pytest.raises(AlganConfigurationError, match="torch_compile"):
        SETTINGS.computing.set(torch_compile="sometimes")
    with pytest.raises(AlganConfigurationError, match="torch_compile"):
        SETTINGS.computing.torch_compile = 2


def test_auto_follows_platform_support(monkeypatch):
    monkeypatch.setattr(tc, "_SUPPORT", (True, ""))
    SETTINGS.computing.set(torch_compile="auto")
    assert tc.torch_compile_enabled()
    monkeypatch.setattr(tc, "_SUPPORT", (False, "no"))
    assert not tc.torch_compile_enabled()
    # Explicit values decide for themselves, whatever the platform says.
    SETTINGS.computing.set(torch_compile=True)
    assert tc.torch_compile_enabled()
    monkeypatch.setattr(tc, "_SUPPORT", (True, ""))
    SETTINGS.computing.set(torch_compile=False)
    assert not tc.torch_compile_enabled()


def test_windows_is_unsupported_by_default(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    supported, reason = tc._probe_support()
    assert not supported
    assert "Windows" in reason


def test_this_platform_reports_a_reason_only_when_unsupported():
    supported, reason = tc.torch_compile_support()
    assert supported == (reason == "")
    # Memoized: the same object comes back.
    assert tc.torch_compile_support() is tc.torch_compile_support()


def test_the_environment_variable_overrides_the_setting(monkeypatch):
    SETTINGS.computing.set(torch_compile=True)
    monkeypatch.setenv("ALGAN_TORCH_COMPILE", "0")
    assert not tc.torch_compile_enabled()
    SETTINGS.computing.set(torch_compile=False)
    monkeypatch.setenv("ALGAN_TORCH_COMPILE", "1")
    assert tc.torch_compile_enabled()


def test_the_switch_is_read_at_every_call():
    calls = []

    @tc.compiled
    def f(x):
        calls.append(1)
        return x * 2 + 1

    x = torch.arange(4.0)
    SETTINGS.computing.set(torch_compile=False)
    assert torch.equal(f(x), x * 2 + 1)
    assert _wrapped_state(f).compiled is None, "off means no compile was built"

    SETTINGS.computing.set(torch_compile=True)
    assert torch.equal(f(x), x * 2 + 1)
    assert _wrapped_state(f).compiled is not None
    name, state = tc.compiled_functions()[-1]
    assert name.endswith("test_the_switch_is_read_at_every_call.<locals>.f")
    assert state == "compiled"

    SETTINGS.computing.set(torch_compile=False)
    assert torch.equal(f(x), x * 2 + 1)
    assert f.eager(x) is not None


def test_decorator_keyword_form_and_eager_attribute():
    @tc.compiled(dynamic=True)
    def f(x):
        return x.sum()

    assert f.eager(torch.ones(3)) == 3
    assert _wrapped_state(f).options["dynamic"] is True
    SETTINGS.computing.set(torch_compile=True)
    assert f(torch.ones(5)) == 5


def test_a_compile_failure_warns_once_and_runs_eagerly(monkeypatch):
    def broken_backend(gm, example_inputs):
        raise RuntimeError("no compiler on this box")

    monkeypatch.setattr(tc, "_BACKEND", broken_backend)
    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled
    def f(x):
        return x + 1

    x = torch.zeros(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert torch.equal(f(x), x + 1)
        assert torch.equal(f(x), x + 1)
    messages = [str(w.message) for w in caught if w.category is AlganWarning]
    assert len(messages) == 1, messages
    assert "no compiler on this box" in messages[0]
    assert "torch_compile=False" in messages[0]
    assert _wrapped_state(f).failed
    name, verdict = tc.compiled_functions()[-1]
    assert name.endswith(".f")
    assert verdict.startswith("failed:")
    assert "no compiler on this box" in verdict


def test_an_error_raised_by_the_function_itself_propagates():
    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled
    def f(x):
        if x.shape[0] > 2:
            raise ValueError("the function's own complaint")
        return x

    with pytest.raises(ValueError, match="own complaint"):
        f(torch.zeros(3))
    # Not demoted: the compiler did nothing wrong.
    assert not _wrapped_state(f).failed


def test_reset_forgets_compiled_callables_and_verdicts(monkeypatch):
    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled
    def f(x):
        return x - 1

    f(torch.ones(2))
    state = _wrapped_state(f)
    assert state.compiled is not None
    state.failed = True
    tc.reset_compiled_functions()
    assert state.compiled is None
    assert not state.failed


def test_the_cli_health_check_reports_the_switch(capsys):
    from algan.cli import _cmd_check

    _cmd_check(None)
    out = capsys.readouterr().out
    assert "torch.compile" in out


def test_a_tensor_subclass_argument_is_traced_as_a_plain_tensor():
    """The merged scene hands the projection a ``Color``; Dynamo refuses it."""
    from algan.constants.color import Color

    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled
    def f(x, floor):
        return torch.where(x < floor, floor, x) * 2

    color = Color([0.1, 0.5, 0.9, 1.0])
    out = f(color, 0.2)
    assert type(out) is torch.Tensor
    assert torch.equal(out, f.eager(color.as_subclass(torch.Tensor), 0.2))
    assert not _wrapped_state(f).failed


def test_a_backend_that_fails_when_its_code_runs_is_demoted(monkeypatch, tmp_path):
    """torch 2.7's Metal backend generated a wrapper that raised NameError at
    call time -- not a compiler exception, but the compiler's failure.
    """
    generated = tmp_path / "torchinductor_test" / "kernel.py"
    generated.parent.mkdir()
    generated.write_text("def call(*args):\n    return ps0\n")
    namespace = {}
    exec(compile(generated.read_text(), str(generated), "exec"), namespace)

    def backend(gm, example_inputs):
        return namespace["call"]

    monkeypatch.setattr(tc, "_BACKEND", backend)
    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled
    def f(x):
        return x + 1

    x = torch.zeros(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert torch.equal(f(x), x + 1)
    assert [w for w in caught if w.category is AlganWarning]
    assert _wrapped_state(f).failed
    assert "NameError" in _wrapped_state(f).reason
