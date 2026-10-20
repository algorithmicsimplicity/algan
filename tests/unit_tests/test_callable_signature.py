"""Reading a callback's signature without evaluating its annotations.

Algan checks how a user's function can be called by reading its signature.
From Python 3.14 annotations are evaluated lazily, when something reads them,
so a function annotated with a type imported only for a type checker is
defined, and called, without error, and ``inspect.signature`` used to be the
one thing that raised ``NameError`` for it.
"""

from __future__ import annotations

import functools
import inspect
import sys

import pytest
import torch

import algan
from algan.scene_manager import SceneManager
from algan.utils.python_utils import callable_signature

needs_lazy_annotations = pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="before Python 3.14 such an annotation raises where it is defined",
)


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def defined_without_future_annotations(source: str):
    """The function ``source`` defines, compiled as a script without
    ``from __future__ import annotations`` would compile it.
    """
    namespace = {}
    exec(compile(source, "<script>", "exec", dont_inherit=True), namespace)
    (function,) = (value for value in namespace.values() if callable(value))
    return function


def test_it_reads_what_inspect_reads():
    def field(points, t=0.5, *, scale: float = 1.0):
        return points

    assert callable_signature(field) == inspect.signature(field)

    @functools.wraps(field)
    def adapter(*args, **kwargs):
        return field(*args, **kwargs)

    assert str(callable_signature(adapter)) == str(inspect.signature(field))
    assert str(callable_signature(adapter, follow_wrapped=False)) == (
        "(*args, **kwargs)"
    )

    with pytest.raises(TypeError):
        callable_signature(1)


@needs_lazy_annotations
def test_an_annotation_naming_an_undefined_type_is_not_evaluated():
    field = defined_without_future_annotations(
        "def field(points: Tensor, t: np.ndarray) -> Tensor:\n    return points\n"
    )
    with pytest.raises(NameError):
        inspect.signature(field)
    assert list(callable_signature(field).parameters) == ["points", "t"]


@needs_lazy_annotations
def test_callbacks_annotated_for_a_type_checker_are_accepted():
    lift = defined_without_future_annotations(
        "def lift(points: Tensor, t: Tensor) -> Tensor:\n"
        "    return points + t * torch.tensor([0.0, 1.0, 0.0])\n"
    )
    lift.__globals__["torch"] = torch
    follow = defined_without_future_annotations(
        "def follow(mob: Mob, t: Tensor) -> None:\n    pass\n"
    )
    linear = defined_without_future_annotations(
        "def linear(t: Tensor) -> Tensor:\n    return t\n"
    )

    line = algan.Line([0, 0, 0], [1, 0, 0], add_to_scene=False).spawn(False)
    algan.Homotopy(line, lift, runtime=1, easing=linear)
    line.add_updater(follow)
