"""A ``continue`` under a ``ti.static`` gate is invalid SPIR-V. Ban it.

The one instance of this cost six tests on the Apple GPU and a day to find
(``DESIGN_mps_support.md`` §1.2c), and nothing about it was visible on the
backends this project develops on.

**The mechanism.** ``ti.static`` is resolved by Taichi's AST transformer, which
emits the taken branch's statements *inline* and leaves no ``IfStmt`` behind.
So ``if ti.static(cond): continue`` reaches the SPIR-V codegen as a **bare**
``ContinueStmt`` in the loop body, followed by every statement after it.
``visit(ContinueStmt)`` emits an ``OpBranch`` and sets a flag that the next
``IfStmt`` boundary consumes by opening a new ``OpLabel``; with no boundary
left to reach, everything after the ``continue`` is emitted into a block that
has already been terminated, and the module fails validation with ``Load must
appear in a block``. A statically-unrolled ``for k in ti.static(range(n))``
does the same thing for the same reason -- the iterations are inlined, so a
``continue`` inside one is followed by the rest of them.

**Why a source check and not a runtime one.** LLVM does not mind, so the CPU
and CUDA backends execute the invalid module correctly and no test on either
can see it. Metal answers with a nil compute pipeline and Vulkan with
``vkCreateComputePipelines failed`` -- both a machine away. The AST is here.

``taichi_patches/0002`` fixes the codegen as well, so the forked wheel an Apple
GPU needs is not exposed either way; this stands because Algan should not need
a forked Taichi to emit a valid module, and because the gated form
(``if ti.static(not cond): <the rest>``) is the clearer kernel anyway.
"""

import ast
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[2] / "algan"

#: Kernel modules. The hazard is in Taichi's codegen, so only code Taichi
#: compiles can carry it, and this project keeps all of that behind the
#: ``_taichi`` suffix (see CLAUDE.md's linting notes, which key three separate
#: things off the same convention).
_KERNEL_MODULES = sorted(_PACKAGE.rglob("*_taichi.py"))


def _is_ti_static(node):
    """Whether ``node`` is a call to ``ti.static(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "static"
    )


def _static_gated_continues(tree):
    """``(line, what gated it)`` for every ``continue`` under a static gate."""
    found = []

    def walk(node, gate):
        for child in ast.iter_child_nodes(node):
            child_gate = gate
            if isinstance(child, ast.If) and _is_ti_static(child.test):
                child_gate = gate or "if ti.static(...)"
            elif isinstance(child, ast.For) and _is_ti_static(child.iter):
                child_gate = gate or "for ... in ti.static(...)"
            elif isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                # A real loop: a `continue` inside it belongs to *it*, and
                # Taichi emits a real IfStmt for the runtime condition that
                # guards it, so the gate does not carry across.
                child_gate = None
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_gate = None
            if isinstance(child, ast.Continue) and child_gate:
                found.append((child.lineno, child_gate))
            walk(child, child_gate)

    walk(tree, None)
    return found


def test_kernel_modules_exist():
    """Guard the guard: a glob that matches nothing passes vacuously."""
    assert len(_KERNEL_MODULES) > 10


@pytest.mark.parametrize("path", _KERNEL_MODULES, ids=lambda p: p.name)
def test_no_continue_under_a_static_gate(path):
    offenders = _static_gated_continues(ast.parse(path.read_text()))
    assert not offenders, "\n".join(
        [
            f"{path.relative_to(_PACKAGE.parent)} has a `continue` inside a "
            "compile-time gate, which emits a bare ContinueStmt and leaves "
            "every statement after it in an already-terminated block "
            "(invalid SPIR-V; see this file's docstring):",
            *(f"  line {line}: inside {gate}" for line, gate in offenders),
            "Gate the statements that follow instead: "
            "`if ti.static(not cond): <the rest>`.",
        ]
    )
