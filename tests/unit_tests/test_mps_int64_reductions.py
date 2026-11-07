"""int64 ``amin``/``amax`` scatters never reach MPS (``mps_compat.scatter_reduce_``).

On the Mac runner's GPU, torch 2.13.0 cannot build the kernel for an int64
``scatter_reduce_`` ``amin`` or ``amax``: the call raises a failed Metal
pipeline compile, after it has already queued the sign-bit pass that brackets
the kernel, and a second failure in the process leaves it unable to compile
anything new (``mps_compat.scatter_reduce_`` has the runs). So the renderer
makes every scatter reduction through ``mps_compat``, which computes an int64
extremum from two int32 ones on MPS and in MPS-friendly mode, and the last
test here fails on a direct call anywhere in ``algan``.

The CPU tests make the CPU's int64 extremum raise as the GPU's does and ask
for torch's own answer; the ``apple_gpu`` test reduces on the Apple GPU
whenever the machine has one.
"""

import ast
from pathlib import Path

import pytest
import torch

import algan
from algan.rendering import mps_compat
from algan.settings import SETTINGS

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


@pytest.fixture
def mps_friendly(monkeypatch):
    """MPS-friendly mode, whatever ``ALGAN_MPS_FRIENDLY`` says."""
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)


def _refuse_int64_extrema(monkeypatch):
    """Make the CPU's int64 ``amin``/``amax`` scatter raise, as MPS's does.

    Every other dtype and reduction still runs, so a route that reduces in
    int32 gets through and a route that reaches the int64 op does not.
    """

    def refusing(op):
        def call(self, *args, **kwargs):
            reduce = kwargs.get("reduce", args[3] if len(args) > 3 else None)
            if self.dtype == torch.int64 and reduce in ("amin", "amax"):
                raise RuntimeError(
                    "Failed to created pipeline state object (simulated MPS refusal)"
                )
            return op(self, *args, **kwargs)

        return call

    for name in ("scatter_reduce_", "scatter_reduce"):
        monkeypatch.setattr(torch.Tensor, name, refusing(getattr(torch.Tensor, name)))
    monkeypatch.setattr(torch, "scatter_reduce", refusing(torch.scatter_reduce))


def _cases(seed):
    """``(base, index, src)`` int64 triples that need all 64 bits.

    Random words, near-equal runs that differ only in the low word (and in
    its top bit, which a signed int32 compare would misorder), negative values
    whose high words tie, the int64 extremes, and slots nothing scatters to.
    """
    g = torch.Generator().manual_seed(seed)
    n, slots = 5000, 700
    index = torch.randint(0, slots - 50, (n,), generator=g)
    wide = torch.randint(INT64_MIN, INT64_MAX, (n,), generator=g)
    high = torch.randint(-4, 4, (n,), generator=g) << 32
    low = torch.randint(0, 2**32, (n,), generator=g)
    near = high | low
    edge = torch.tensor([INT64_MIN, INT64_MAX, -1, 0, 2**31, 2**32 - 1, -(2**31)])
    pick = torch.randint(0, 3, (n,), generator=g)
    src = torch.where(pick == 0, wide, near)
    src[: edge.numel()] = edge
    src[-edge.numel() :] = edge.flip(0)
    base = torch.where(
        torch.rand(slots, generator=g) < 0.5,
        torch.randint(INT64_MIN, INT64_MAX, (slots,), generator=g),
        (torch.randint(-4, 4, (slots,), generator=g) << 32)
        | torch.randint(0, 2**32, (slots,), generator=g),
    )
    yield base, index, src
    yield base.clone().fill_(INT64_MAX), index, src
    yield base.clone().fill_(INT64_MIN), index, src


@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", ["amin", "amax"])
def test_an_int64_extremum_comes_back_exact_without_the_int64_op(
    mps_friendly, monkeypatch, reduce, include_self
):
    """In MPS-friendly mode the reduction is torch's own answer, bit for bit,
    with the int64 op refusing as it does on the GPU.
    """
    wants = [
        (
            base,
            index,
            src,
            base.clone().scatter_reduce_(
                0, index, src, reduce, include_self=include_self
            ),
        )
        for seed in (0, 1)
        for base, index, src in _cases(seed)
    ]
    _refuse_int64_extrema(monkeypatch)
    for base, index, src, want in wants:
        out = base.clone()
        got = mps_compat.scatter_reduce_(
            out, 0, index, src, reduce, include_self=include_self
        )
        assert got is out
        assert torch.equal(got, want), (
            f"{int((got != want).sum())} of {want.numel()} differ"
        )
        got = mps_compat.scatter_reduce(
            base, 0, index, src, reduce, include_self=include_self
        )
        assert torch.equal(got, want), (
            f"{int((got != want).sum())} of {want.numel()} differ"
        )


def test_an_int64_extremum_off_the_route_raises_before_the_op(
    mps_friendly, monkeypatch
):
    """Only a 1-D reduction whose index and src have one length has the
    two-word route, so a reduction over rows, or over the head of a longer
    src, raises ``NotImplementedError`` instead of reaching the int64 op.
    """
    _refuse_int64_extrema(monkeypatch)
    out = torch.zeros((3, 4), dtype=torch.int64)
    index = torch.zeros((3, 2), dtype=torch.int64)
    with pytest.raises(NotImplementedError, match="1-D"):
        mps_compat.scatter_reduce_(out, 1, index, index, "amin")
    out = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(NotImplementedError, match="one length"):
        mps_compat.scatter_reduce_(
            out, 0, torch.zeros(2, dtype=torch.int64), torch.arange(3), "amax"
        )


def test_every_other_reduction_is_the_op_itself(mps_friendly, monkeypatch):
    """Float, int32 and int64 sums go straight to torch, once each."""
    calls = []
    real = torch.Tensor.scatter_reduce_

    def spy(self, *args, **kwargs):
        calls.append(self.dtype)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "scatter_reduce_", spy)
    g = torch.Generator().manual_seed(2)
    index = torch.randint(0, 10, (100,), generator=g)
    for dtype, reduce in (
        (torch.float32, "amin"),
        (torch.int32, "amax"),
        (torch.int64, "sum"),
        (torch.int64, "prod"),
    ):
        src = torch.randint(-3, 4, (100,), generator=g).to(dtype)
        want = real(torch.ones(10, dtype=dtype), 0, index, src, reduce)
        calls.clear()
        got = mps_compat.scatter_reduce_(
            torch.ones(10, dtype=dtype), 0, index, src, reduce
        )
        assert calls == [dtype]
        assert torch.equal(got, want)


@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", ["amin", "amax"])
def test_an_int64_extremum_on_the_apple_gpu_is_the_cpu_one(
    mps_friendly, reduce, include_self
):
    """Reduced on the Apple GPU whenever the machine has one, the answer is
    the CPU's int64 op's.
    """
    for base, index, src in _cases(3):
        want = base.clone().scatter_reduce_(
            0, index, src, reduce, include_self=include_self
        )
        got = mps_compat.scatter_reduce_(
            base.to(DEVICE),
            0,
            index.to(DEVICE),
            src.to(DEVICE),
            reduce,
            include_self=include_self,
        ).cpu()
        assert torch.equal(got, want), (
            f"{int((got != want).sum())} of {want.numel()} differ"
        )


#: Calls that reduce, by the method or function name torch spells them with.
_REDUCTIONS = {
    "scatter_reduce_": "mps_compat.scatter_reduce_",
    "scatter_reduce": "mps_compat.scatter_reduce",
    "index_reduce_": "mps_compat.index_reduce_",
    "index_reduce": "mps_compat.index_reduce_",
}


@pytest.mark.fast
def test_algan_reduces_through_mps_compat():
    """A direct scatter reduction anywhere in ``algan`` is one int64 ``amin``
    away from an MPS process that cannot compile another kernel.

    Over the AST rather than the text, so comments and docstrings may name the
    ops. A ``scatter``/``scatter_`` with a ``reduce=`` keyword is the older
    spelling of the same reductions, and counts too, as does importing one of
    the functions from torch. Only a file whose text names ``scatter`` or
    ``index_reduce`` can hold one, so only those are parsed.
    """
    root = Path(algan.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "mps_compat.py" and path.parent.name == "rendering":
            continue
        text = path.read_text(encoding="utf-8")
        if "scatter" not in text and "index_reduce" not in text:
            continue
        relative = path.relative_to(root.parent).as_posix()
        for node in ast.walk(ast.parse(text, filename=str(path))):
            if isinstance(node, ast.Attribute) and node.attr in _REDUCTIONS:
                offenders.append(
                    (
                        relative,
                        node.lineno,
                        f".{node.attr} -- use {_REDUCTIONS[node.attr]}",
                    )
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("scatter", "scatter_")
                and any(keyword.arg == "reduce" for keyword in node.keywords)
            ):
                offenders.append(
                    (
                        relative,
                        node.lineno,
                        f".{node.func.attr}(reduce=...) -- use mps_compat.scatter_reduce_",
                    )
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "torch"
            ):
                for alias in node.names:
                    if alias.name in _REDUCTIONS:
                        offenders.append(
                            (
                                relative,
                                node.lineno,
                                f"import of {alias.name} -- use {_REDUCTIONS[alias.name]}",
                            )
                        )
    assert not offenders, "\n".join(
        [
            "These reductions can reach MPS without mps_compat's guard:",
            *(f"{path}:{line}: {what}" for path, line, what in sorted(offenders)),
        ]
    )
