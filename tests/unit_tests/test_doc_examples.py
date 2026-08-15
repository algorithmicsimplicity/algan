"""Check that the code in ``docs/`` still matches the code in ``algan/``.

The documentation carries two kinds of Python block and only one of them was
ever executed by anything:

* ``.. algan::`` blocks are rendered during a full documentation build, so a
  broken one fails that build -- but the build renders every example, takes
  minutes, and does not run in CI;
* ``.. code-block:: python`` blocks are syntax-highlighted and nothing more.

Most of the documentation's code is the second kind, and most of it is a
*fragment* -- three lines using an undefined ``mob`` -- which can never become a
runnable scene without inventing scaffolding around it. So this module checks
what each block can actually support, in three tiers:

``test_doc_example_uses_public_api``
    Static, every block, no execution. Resolves the names and attribute chains a
    block uses against the real ``algan`` namespace. Catches an API that was
    renamed or removed out from under a fragment.

``test_doc_example_authors_without_error``
    Executes every block that is a complete script, with ``save_video`` /
    ``save_frame`` stubbed out. Catches everything that goes wrong while
    *authoring* a scene: wrong constructor arguments, a value of the wrong
    width, a method that no longer exists.

``test_doc_example_renders`` (``slow``)
    Actually renders those scripts at ``SMOKE_TEST``. This is the only tier that
    sees a render-time failure -- an updater that raises when it is evaluated
    over a batch of frames, say -- and it is far too slow for the fast suite.

A block that is deliberately not runnable (an anti-example showing what raises,
a Manim-side snippet in a migration comparison) opts out with a
``# algan-doc-check: skip`` comment. A block that is broken by a bug we already
know about goes in ``KNOWN_BROKEN`` with a reason, so it is skipped loudly
rather than deleted quietly.
"""

from __future__ import annotations

import ast
import builtins
import os
import re
from pathlib import Path

import pytest

import algan
from algan.scene_manager import SceneManager

DOCS_SOURCE = Path(__file__).resolve().parents[2] / "docs" / "source"

# Written as an reStructuredText comment on the line(s) immediately above a
# block, so it never reaches the rendered page:
#
#     .. algan-doc-check: skip -- needs an asset that does not ship with the docs
#
#     .. code-block:: python
SKIP_MARKER = "algan-doc-check: skip"

# How far above a directive to look for the marker.
_MARKER_LOOKBACK = 4

# Blocks that fail for a reason already being worked on. Keyed by the block id
# ("<path>:<line>") that the parametrization reports, so a failure names exactly
# what to add here. Prefer fixing the bug; this is for when the doc is correct
# and the engine is not.
KNOWN_BROKEN: dict[str, str] = {
    # Animated color_texture: a Surface color texture with a leading time
    # dimension raises at construction. The documentation describes the
    # intended behaviour, so the doc stays and the example is skipped until the
    # engine catches up.
}

# Attribute chains rooted at one of these resolve against the live object, which
# is what catches a setting or a rate function that was renamed. Names are
# looked up on the `algan` module.
CHECKED_ROOTS = ("SETTINGS", "rate_funcs")

_BUILTINS = frozenset(dir(builtins))
_ALGAN_NAMES = frozenset(algan.__all__) | {
    n for n in dir(algan) if not n.startswith("_")
}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


class DocExample:
    """One Python block lifted out of a documentation page."""

    def __init__(
        self, path: Path, line: int, code: str, directive: str, opted_out: bool = False
    ):
        self.path = path
        self.line = line
        self.code = code
        self.directive = directive
        self.opted_out = opted_out

    @property
    def id(self) -> str:
        return f"{self.path.relative_to(DOCS_SOURCE).as_posix()}:{self.line}"

    @property
    def skipped_by_marker(self) -> bool:
        return self.opted_out or SKIP_MARKER in self.code

    @property
    def is_complete_script(self) -> bool:
        """Whether the block can stand alone as a scene script.

        ``.. algan::`` bodies are complete by construction -- the directive
        rejects anything else. A ``code-block`` qualifies when it imports algan
        and renders exactly once.
        """
        if self.directive == "algan":
            return True
        imports_algan = "from algan import" in self.code or "import algan" in self.code
        saves = len(re.findall(r"\.save_(?:video|frame)\s*\(", self.code))
        return imports_algan and saves == 1


def _iter_blocks(path: Path):
    """Yield ``(line, code, directive, opted_out)`` per Python block in a page."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i, n = 0, len(lines)
    while i < n:
        match = re.match(r"^(\s*)\.\.\s+(algan|code-block::\s*python)\b.*$", lines[i])
        if not match:
            i += 1
            continue
        indent = len(match.group(1))
        directive = "algan" if match.group(2) == "algan" else "code-block"
        body: list[str] = []
        base: int | None = None
        start = i + 1
        j = i + 1
        while j < n:
            line = lines[j]
            if not line.strip():
                if body:
                    body.append("")
                j += 1
                continue
            current = len(line) - len(line.lstrip())
            if current <= indent:
                break
            stripped = line.strip()
            # Directive options (":name: value") precede the body.
            if stripped.startswith(":") and not body:
                j += 1
                continue
            if base is None:
                base = current
                start = j + 1
            body.append(line[base:] if len(line) > base else line.lstrip())
            j += 1
        code = "\n".join(body).rstrip()
        if code:
            preceding = lines[max(0, i - _MARKER_LOOKBACK) : i]
            opted_out = any(SKIP_MARKER in line for line in preceding)
            yield start, code, directive, opted_out
        i = j


def _collect_examples() -> list[DocExample]:
    examples = []
    for page in sorted(DOCS_SOURCE.rglob("*.rst")):
        if "_templates" in page.parts:
            continue
        for line, code, directive, opted_out in _iter_blocks(page):
            examples.append(DocExample(page, line, code, directive, opted_out))
    return examples


EXAMPLES = _collect_examples()
COMPLETE = [e for e in EXAMPLES if e.is_complete_script and not e.skipped_by_marker]


def _ids(examples):
    return [e.id for e in examples]


def test_documentation_has_examples_to_check():
    """Guard the extractor itself.

    If a directive is renamed or the indentation convention changes, the two
    tests below would silently pass over an empty list.
    """
    assert len(EXAMPLES) > 100, f"only extracted {len(EXAMPLES)} doc examples"
    assert len(COMPLETE) > 20, f"only {len(COMPLETE)} complete doc scripts"


# --------------------------------------------------------------------------
# tier 1 -- static
# --------------------------------------------------------------------------


class _NameCollector(ast.NodeVisitor):
    """Free names loaded by a block, and attribute chains rooted at a name."""

    def __init__(self):
        self.loads: list[tuple[str, int]] = []
        self.bound: set[str] = set()
        self.chains: list[tuple[str, int]] = []
        self.algan_star_import = False

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.append((node.id, node.lineno))
        else:
            self.bound.add(node.id)
        self.generic_visit(node)

    def _bind_args(self, args):
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            self.bound.add(arg.arg)
        if args.vararg:
            self.bound.add(args.vararg.arg)
        if args.kwarg:
            self.bound.add(args.kwarg.arg)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)
        self._bind_args(node.args)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._bind_args(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                if node.module == "algan":
                    self.algan_star_import = True
                    self.bound |= _ALGAN_NAMES
                else:
                    # A star import from anything else (the Manim side of a
                    # migration comparison) puts names we cannot enumerate into
                    # scope, so this block is not ours to check.
                    self.bound.add("*")
            else:
                self.bound.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        parts, current = [], node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            self.chains.append((".".join(reversed(parts)), node.lineno))
        self.generic_visit(node)


def _resolve_chain(chain: str):
    """Resolve a dotted chain against the live algan module.

    Returns ``None`` when it resolves, or the first component that does not.
    """
    parts = chain.split(".")
    obj = getattr(algan, parts[0], None)
    if obj is None:
        return parts[0]
    for i, attr in enumerate(parts[1:], start=1):
        # Stop at a call result or a subscript -- we only follow plain
        # attribute access, which is what a renamed setting shows up as.
        if not hasattr(obj, attr):
            return ".".join(parts[: i + 1])
        obj = getattr(obj, attr)
    return None


@pytest.mark.parametrize("example", EXAMPLES, ids=_ids(EXAMPLES))
def test_doc_example_uses_public_api(example: DocExample):
    """Every name a documented block uses still exists in algan."""
    if example.skipped_by_marker:
        pytest.skip("opted out with the algan-doc-check marker")

    try:
        tree = ast.parse(example.code)
    except SyntaxError as error:
        pytest.fail(f"{example.id} is not valid Python: {error}")

    collector = _NameCollector()
    collector.visit(tree)

    # A block that star-imports algan is claiming its capitalized bare names --
    # classes and constants -- come from the public API, so an unresolvable one
    # is a removed or renamed export. This is where `Synchronized`, `Lagged` and
    # `Sequenced` would have been caught.
    #
    # Lowercase free names are deliberately allowed: the docs are full of useful
    # fragments that star-import algan and then operate on an undefined `mob` or
    # `square`, and demanding scaffolding around those would cost the reader far
    # more than the check is worth.
    if collector.algan_star_import and "*" not in collector.bound:
        unknown = sorted(
            {
                name
                for name, _ in collector.loads
                if name[:1].isupper()
                and name not in collector.bound
                and name not in _BUILTINS
            }
        )
        assert not unknown, (
            f"{example.id} uses {unknown}, which algan does not export -- "
            f"renamed or removed?"
        )

    broken = []
    for chain, _ in collector.chains:
        if chain.split(".")[0] not in CHECKED_ROOTS:
            continue
        missing = _resolve_chain(chain)
        if missing is not None:
            broken.append(f"{chain} (no {missing})")
    assert not broken, f"{example.id} refers to {broken}, which no longer resolve"


# --------------------------------------------------------------------------
# tiers 2 and 3 -- execution
# --------------------------------------------------------------------------


def _run_example(example: DocExample, *, render: bool):
    """Execute one complete block, optionally letting it render.

    Runs with the working directory set to ``docs/source`` so the relative
    asset paths the examples use (``world_map.png``) resolve the way they do
    during a documentation build.
    """
    from algan import SETTINGS, SMOKE_TEST
    from algan.scene import Scene

    reason = KNOWN_BROKEN.get(example.id)
    if reason is not None:
        pytest.skip(f"known bug: {reason}")

    SETTINGS.video.set(SMOKE_TEST)

    # `save_video` / `save_frame` are `active_scene_method` descriptors. Reading
    # them off the class runs the descriptor and hands back a resolved function,
    # so saving `Scene.save_video` and assigning it back later would quietly
    # replace the descriptor with a plain function and break class-level access
    # for every later test. Take the raw entries out of the class __dict__.
    stubbed = ("save_video", "save_frame")
    originals = {name: Scene.__dict__.get(name) for name in stubbed}
    previous_cwd = Path.cwd()

    def _skip_render(*args, **kwargs):
        return None

    SceneManager.reset()
    os.chdir(DOCS_SOURCE)
    try:
        if not render:
            for name in stubbed:
                setattr(Scene, name, staticmethod(_skip_render))
        namespace = {"__name__": "__algan_doc_example__"}
        exec(compile(example.code, example.id, "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
        if not render:
            for name, original in originals.items():
                if original is None:
                    delattr(Scene, name)
                else:
                    setattr(Scene, name, original)
        SceneManager.reset()


@pytest.mark.parametrize("example", COMPLETE, ids=_ids(COMPLETE))
def test_doc_example_authors_without_error(example: DocExample):
    """Complete documented scripts build their scene without raising.

    Rendering is stubbed out, so this covers authoring only: constructor
    arguments, attribute widths, and methods that still exist. A failure here
    means a reader copying the block hits the same exception.
    """
    _run_example(example, render=False)


@pytest.mark.slow
@pytest.mark.parametrize("example", COMPLETE, ids=_ids(COMPLETE))
def test_doc_example_renders(example: DocExample):
    """Complete documented scripts render end to end at SMOKE_TEST.

    Marked ``slow`` because it renders every example: this is the only tier
    that catches a failure occurring during materialization or rasterization
    rather than during authoring, and it costs far more than the fast suite's
    whole budget.
    """
    _run_example(example, render=True)
