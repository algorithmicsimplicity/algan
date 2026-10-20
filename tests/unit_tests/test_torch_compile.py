"""The ``torch.compile`` switch and the decorator every pipeline function uses.

The wrapper is exercised against Dynamo's ``eager`` backend -- tracing with no
code generation -- so these tests cost milliseconds rather than an Inductor
build, and against a backend that raises, so the fallback contract is pinned:
a compile failure warns once and runs the function eagerly, an error the
function itself raises propagates untouched, and the switch is read live.
"""

from __future__ import annotations

import sys
import types
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
    # The render device is the other half of what 'auto' resolves against, and
    # on the Apple GPU arm it declines on its own (test_auto_declines_torch_
    # compile_on_metal, below). Pin it so this asks only about the platform.
    monkeypatch.setattr(
        "algan.settings._startup.render_device", lambda: torch.device("cpu")
    )
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


@pytest.mark.parametrize("width", [0, 128, 256, 512])
def test_inductor_graph_cache_keys_the_resolved_cpu_vector_width(monkeypatch, width):
    """A cache moved between CPUs must not reuse loops with another lane count."""
    from torch._inductor import config, cpu_vec_isa

    calls = []

    def compile_stub(fn, **kwargs):
        calls.append(kwargs)
        return fn

    monkeypatch.setattr(tc, "_BACKEND", "inductor")
    monkeypatch.setattr(tc, "_remember_vec_isa_probe", lambda: None)
    monkeypatch.setattr(torch, "compile", compile_stub)
    monkeypatch.setattr(
        cpu_vec_isa,
        "pick_vec_isa",
        lambda: types.SimpleNamespace(bit_width=lambda: width),
    )

    @tc.compiled(dynamic=True)
    def project(x):
        return x + 1

    with config.patch({"cpp.simdlen": None}):
        SETTINGS.computing.set(torch_compile=True)
        assert torch.equal(project(torch.zeros(3)), torch.ones(3))
        assert config.cpp.simdlen is None

    assert calls == [
        {
            "backend": "inductor",
            "dynamic": True,
            "fullgraph": False,
            "options": {"cpp.simdlen": width},
        }
    ]


def test_vector_width_resolution_respects_compile_overrides(monkeypatch):
    from torch._inductor import config, cpu_vec_isa

    seen = []

    def pick():
        seen.append((config.cpp.simdlen, config.cpp.vec_isa_ok))
        return types.SimpleNamespace(bit_width=lambda: 256)

    monkeypatch.setattr(cpu_vec_isa, "pick_vec_isa", pick)
    requested = {
        "dynamic": True,
        "options": {"cpp.simdlen": 256, "cpp.vec_isa_ok": True, "max_autotune": True},
    }
    with config.patch({"cpp.simdlen": None, "cpp.vec_isa_ok": None}):
        result = tc._inductor_compile_options(requested)
        assert (config.cpp.simdlen, config.cpp.vec_isa_ok) == (None, None)
    assert seen == [(256, True)]
    assert result == requested
    assert result is not requested
    assert result["options"] is not requested["options"]


def test_vector_width_resolution_preserves_compile_modes(monkeypatch):
    from torch._inductor import cpu_vec_isa, list_mode_options

    monkeypatch.setattr(
        cpu_vec_isa,
        "pick_vec_isa",
        lambda: types.SimpleNamespace(bit_width=lambda: 256),
    )
    requested = {"mode": "reduce-overhead", "dynamic": False, "fullgraph": True}
    result = tc._inductor_compile_options(requested)
    assert result == {
        "dynamic": False,
        "fullgraph": True,
        "options": {
            **list_mode_options("reduce-overhead", dynamic=False),
            "cpp.simdlen": 256,
        },
    }
    assert requested["mode"] == "reduce-overhead"
    invalid = {"mode": "default", "options": {}}
    assert tc._inductor_compile_options(invalid) == invalid


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
    from algan_cli import _cmd_check

    _cmd_check(None)
    out = capsys.readouterr().out
    assert "torch.compile" in out


def test_named_moving_axes_compile_one_graph_for_every_size():
    """``dynamic_dims`` and ``batch_args`` keep a moving batch on one graph.

    Left to automatic dynamic shapes, these calls trace a static graph, then
    one per axis seen moving, then one for the batch of one; every graph is
    traced again in each new process.
    """
    from torch._dynamo.utils import counters

    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled(dynamic_dims={"rows": (0,), "points": (0,)}, batch_args=("rows",))
    def f(rows, points, scale=1.0):
        return (rows.unsqueeze(1) * points.unsqueeze(0)).sum(-1) * scale

    counters.clear()
    for batch, count in ((3, 4), (1, 4), (5, 7), (2, 9), (1, 2)):
        rows = torch.arange(batch * 3.0).view(batch, 3)
        points = torch.linspace(0, 1, count * 3).view(count, 3)
        out = f(rows, points=points)
        assert out.shape == (batch, count)
        assert torch.equal(out, f.eager(rows, points))
    assert counters["stats"]["unique_graphs"] == 1


def test_a_named_axis_must_name_a_parameter():
    with pytest.raises(TypeError, match="no parameter named row"):

        @tc.compiled(dynamic_dims={"row": (0,)})
        def f(rows):
            return rows


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


def _generated_call(tmp_path, body):
    """A ``call`` defined in a file under a ``torchinductor`` cache directory."""
    generated = tmp_path / "torchinductor_test" / "kernel.py"
    generated.parent.mkdir()
    generated.write_text(f"import torch\n\ndef call(*args):\n    {body}\n")
    namespace = {}
    exec(compile(generated.read_text(), str(generated), "exec"), namespace)
    return namespace["call"]


@pytest.mark.parametrize(
    "raised",
    [
        "torch.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB')",
        "RuntimeError('CUDA error: out of memory')",
    ],
    ids=["torch", "driver"],
)
def test_running_out_of_memory_in_generated_code_is_not_a_compile_failure(
    monkeypatch, tmp_path, raised
):
    """The render loop retries an out-of-memory batch with fewer frames. The
    error used to be taken for the compiler's, so the function was rerun at
    once on the same batch and stayed eager for the rest of the process.
    """
    call = _generated_call(tmp_path, f"raise {raised}")
    monkeypatch.setattr(tc, "_BACKEND", lambda gm, example_inputs: call)
    SETTINGS.computing.set(torch_compile=True)
    eager_calls = []

    @tc.compiled
    def f(x):
        eager_calls.append(x)
        return x + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="out of memory"):
            f(torch.zeros(2))
    assert not [w for w in caught if w.category is AlganWarning]
    assert not _wrapped_state(f).failed
    assert not eager_calls


def test_running_out_of_memory_while_compiling_is_not_a_compile_failure(
    monkeypatch,
):
    def backend(gm, example_inputs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(tc, "_BACKEND", backend)
    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled
    def f(x):
        return x + 1

    with pytest.raises(RuntimeError, match="out of memory"):
        f(torch.zeros(2))
    assert not _wrapped_state(f).failed


def test_a_compiler_that_runs_out_of_memory_is_a_compile_failure(monkeypatch):
    """Its own output says "out of memory", but the function is not at fault
    and runs eagerly instead.
    """

    def backend(gm, example_inputs):
        raise RuntimeError("cc1plus: out of memory allocating 65536 bytes")

    monkeypatch.setattr(tc, "_BACKEND", backend)
    SETTINGS.computing.set(torch_compile=True)

    @tc.compiled
    def f(x):
        return x + 1

    x = torch.zeros(2)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert torch.equal(f(x), x + 1)
    assert _wrapped_state(f).failed


def test_auto_declines_torch_compile_on_metal(monkeypatch):
    """``'auto'`` is off on MPS, and an explicit setting still forces it on.

    PyTorch's Inductor backend calls itself an early prototype on Metal, and on
    this codebase it both fails harmlessly (``torch.compile failed for
    _triangle_projection_fused``, after which that function runs eagerly, so the
    switch buys nothing) and breaks a render outright: a UHD pass died in
    ``raster_pipeline._pair_expand_rows`` on "SymIntArrayRef expected to contain
    only concrete integers" at a ``torch.empty`` whose size is already an
    ``int``. The same job with the switch off rendered past that point.

    A default, not a refusal -- which is what lets someone re-test whether
    PyTorch has fixed it.
    """
    import torch as _torch

    monkeypatch.setattr(tc, "torch_compile_support", lambda: (True, ""))
    monkeypatch.delenv("ALGAN_TORCH_COMPILE", raising=False)
    SETTINGS.computing.set(torch_compile="auto")

    monkeypatch.setattr(
        "algan.settings._startup.render_device", lambda: _torch.device("cpu")
    )
    assert tc.torch_compile_enabled() is True, "auto should stay on elsewhere"

    monkeypatch.setattr(
        "algan.settings._startup.render_device", lambda: _torch.device("mps")
    )
    assert tc.torch_compile_enabled() is False, "auto should decline on Metal"
    assert "prototype" in tc._auto_declines_this_device()

    # Explicitly asked for, it still runs -- the switch is a default, not a ban.
    SETTINGS.computing.set(torch_compile=True)
    assert tc.torch_compile_enabled() is True

    # And the env override wins over both, in either direction.
    SETTINGS.computing.set(torch_compile="auto")
    monkeypatch.setenv("ALGAN_TORCH_COMPILE", "1")
    assert tc.torch_compile_enabled() is True
