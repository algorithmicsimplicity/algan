"""Check that the code in ``docs/`` still matches the code in ``algan/``.

The documentation carries two kinds of Python block and only one of them was
ever executed by anything:

* ``.. algan::`` blocks are rendered during a full documentation build, so a
  broken one fails that build -- but that build renders every example and takes
  minutes, and the docs job in CI runs the structural build instead, which parses
  each directive without executing its body;
* ``.. code-block:: python`` blocks are syntax-highlighted and nothing more.

Most of the documentation's code is the second kind, and most of it is a
*fragment* -- three lines using an undefined ``mob`` -- which can never become a
runnable scene without inventing scaffolding around it. So this module checks
what each block can actually support, in four tiers:

``test_algan_directive_is_well_formed``
    Structural, every ``.. algan::`` block on both surfaces -- documentation
    pages *and* docstrings in ``algan/`` -- no execution. Checks what the
    directive itself requires of a block before it runs: a name argument, a body
    that star-imports algan, one video out. The docs job in CI builds with
    ``-t skip-manim``, so it never executes a body but does parse every
    directive; this tier is the cheap stand-in for both halves.

``test_doc_example_uses_public_api``
    Static, every block, no execution. Resolves the names and attribute chains a
    block uses against the real ``algan`` namespace. Catches an API that was
    renamed or removed out from under a fragment.

``test_doc_example_authors_without_error``
    Executes every block that is a complete script, with ``save_video`` /
    ``save_frame`` stubbed out. Catches everything that goes wrong while
    *authoring* a scene: wrong constructor arguments, a value of the wrong
    width, a method that no longer exists.

``test_doc_example_renders`` (opt-in)
    Actually renders those scripts at ``SMOKE_TEST``. This is the only tier that
    sees a render-time failure -- an updater that raises when it is evaluated
    over a batch of frames, say.

    It used to be unrunnable: every example renders in this one process, and
    before the texture-timeline fix a colour texture was sized by a fixed row
    count rather than by its data, so the tier peaked at ~14.7 GB and was
    OOM-killed on a 16 GB machine. With that fixed it costs 2.3 GB and about two
    minutes for 77 examples, so the gate is now about *time*, not headroom.

    Still opt-in, via ``ALGAN_RUN_DOC_RENDERS=1``, because two minutes is more
    than the fast suite's whole budget and CI would pay it on every run -- and
    the measurement above is on a warm Taichi cache, which a fresh runner is
    not. Leaving it out of the ``fast`` suite would not hold it back: CI names
    its paths explicitly rather than passing ``--fast``, and runs everything
    under them, which is exactly how this tier took a runner down once already.

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
from collections import Counter
from pathlib import Path

import pytest

import algan
from algan.scene_manager import SceneManager

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
PACKAGE_ROOT = REPO_ROOT / "algan"

# Written as an reStructuredText comment on the line(s) immediately above a
# block, so it never reaches the rendered page:
#
#     .. algan-doc-check: skip -- needs an asset that does not ship with the docs
#
#     .. code-block:: python
SKIP_MARKER = "algan-doc-check: skip"

# How far above a directive to look for the marker.
_MARKER_LOOKBACK = 4

# The render tier is opt-in. See the module docstring: it is not merely
# expensive, it exhausts memory, and staying out of the fast suite does not keep
# it out of CI because CI names its paths instead of passing --fast.
RUN_DOC_RENDERS = os.getenv("ALGAN_RUN_DOC_RENDERS") == "1"
SKIP_RENDERS_REASON = (
    "rendering every documented example costs ~2 minutes and 2.3 GB, most of "
    "the fast suite's budget, and CI would pay it every run; the authoring tier "
    "covers the same scripts. Set ALGAN_RUN_DOC_RENDERS=1 to run it."
)

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
CHECKED_ROOTS = ("SETTINGS", "easings")

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
        self,
        path: Path,
        line: int,
        code: str,
        directive: str,
        opted_out: bool = False,
        name: str = "",
    ):
        self.path = path
        self.line = line
        self.code = code
        self.directive = directive
        self.opted_out = opted_out
        self.name = name

    @property
    def id(self) -> str:
        # Blocks come from documentation pages and from docstrings in the
        # package, so anchor the path at whichever root contains it.
        root = DOCS_SOURCE if DOCS_SOURCE in self.path.parents else REPO_ROOT
        return f"{self.path.relative_to(root).as_posix()}:{self.line}"

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
    """Yield ``(line, code, directive, opted_out, name)`` per Python block."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i, n = 0, len(lines)
    while i < n:
        match = re.match(
            r"^(\s*)\.\.\s+(algan|code-block::\s*python)\b(.*)$",
            lines[i],
        )
        if not match:
            i += 1
            continue
        indent = len(match.group(1))
        directive = "algan" if match.group(2) == "algan" else "code-block"
        # The argument of `.. algan:: Name` -- empty when the author left it off,
        # which is exactly what `AlganDirective.required_arguments` rejects.
        name = match.group(3).lstrip(":").strip()
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
            yield start, code, directive, opted_out, name
        i = j


def _iter_markdown_blocks(path: Path):
    """Yield ``(line, code)`` per fenced ``python`` block in a Markdown file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != "```python":
            i += 1
            continue
        start = i + 1
        j = start
        while j < n and lines[j].strip() != "```":
            j += 1
        code = "\n".join(lines[start:j]).rstrip()
        if code:
            yield start + 1, code
        i = j + 1


def _collect_markdown_examples() -> list[DocExample]:
    """The Python in ``README.md``.

    Nothing checked it, and it drifted: the Quickstart called ``spawn(runtime=)``,
    ``Sync(runtime=)`` and ``Mob.shift()``, none of which exist -- three
    failures in the first eight lines of code anyone runs, on the front page and
    on PyPI. It is an ordinary complete script, so it goes through the same
    tiers as every other one.
    """
    examples = []
    for page in (REPO_ROOT / "README.md",):
        if not page.exists():
            continue
        for line, code in _iter_markdown_blocks(page):
            examples.append(DocExample(page, line, code, "code-block", False, ""))
    return examples


def _collect_examples() -> list[DocExample]:
    examples = []
    for page in sorted(DOCS_SOURCE.rglob("*.rst")):
        if "_templates" in page.parts:
            continue
        for line, code, directive, opted_out, name in _iter_blocks(page):
            examples.append(DocExample(page, line, code, directive, opted_out, name))
    return examples + _collect_markdown_examples()


def _collect_docstring_directives() -> list[DocExample]:
    """Every ``.. algan::`` block written inside the package's own docstrings.

    Scanned as text rather than through ``ast``: several classes get their
    documentation from a module-level string assigned to ``__doc__`` later
    (``manim_compat``'s ``MathTex`` and ``Title``), which no docstring walk sees.
    """
    examples = []
    for module in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "external_libraries" in module.parts:
            continue
        if ".. algan::" not in module.read_text(encoding="utf-8"):
            continue
        for line, code, directive, opted_out, name in _iter_blocks(module):
            if directive != "algan":
                continue
            examples.append(DocExample(module, line, code, directive, opted_out, name))
    return examples


EXAMPLES = _collect_examples()
COMPLETE = [e for e in EXAMPLES if e.is_complete_script and not e.skipped_by_marker]

# Rendered examples live on both surfaces -- documentation pages and docstrings
# on the API they document -- and the directive treats them identically.
ALGAN_DIRECTIVES = [
    e for e in EXAMPLES if e.directive == "algan"
] + _collect_docstring_directives()


def _ids(examples):
    return [e.id for e in examples]


def test_documentation_has_examples_to_check():
    """Guard the extractor itself.

    If a directive is renamed or the indentation convention changes, the two
    tests below would silently pass over an empty list.
    """
    assert len(EXAMPLES) > 100, f"only extracted {len(EXAMPLES)} doc examples"
    assert len(COMPLETE) > 20, f"only {len(COMPLETE)} complete doc scripts"
    assert len(ALGAN_DIRECTIVES) > 100, (
        f"only extracted {len(ALGAN_DIRECTIVES)} algan directives"
    )
    assert any(e.path.suffix == ".py" for e in ALGAN_DIRECTIVES), (
        "no docstring examples extracted -- the package scan found nothing"
    )


# --------------------------------------------------------------------------
# tier 0 -- what the documentation build itself requires
# --------------------------------------------------------------------------


@pytest.mark.parametrize("example", ALGAN_DIRECTIVES, ids=_ids(ALGAN_DIRECTIVES))
def test_algan_directive_is_well_formed(example: DocExample):
    """Every ``.. algan::`` block is one the documentation build can render.

    The build is the only thing that executes these, it takes minutes because it
    renders each one, and CI runs it *without* rendering (``-t skip-manim -W``).
    So the structural half of a broken directive fails the docs job while the
    body's half only shows up in a full local build. Both are cheap to check
    here, and the alternative -- noticing on master -- has happened: a nameless
    ``.. algan::`` in ``Scene.use_manim_defaults`` red-flagged every docs run
    from the commit that added it.

    The three rules are the directive's own (``required_arguments = 1``, and a
    body executed in a namespace holding nothing but ``__name__``) plus
    ``_find_video``'s, which needs exactly one video to embed.
    """
    assert example.name, (
        f"{example.id}: `.. algan::` needs a name argument -- the directive "
        f"takes one and Sphinx errors out without it. Follow the "
        f"Example{{N}}{{Owner}} convention in DOCSTRINGS.md."
    )
    assert re.search(r"^\s*from algan import \*", example.code, re.M), (
        f"{example.id} ({example.name}): the body runs in an empty namespace, "
        f"so it must start with `from algan import *`."
    )
    saves = len(re.findall(r"\.save_(?:video|frame)\s*\(", example.code))
    assert saves == 1, (
        f"{example.id} ({example.name}): calls save_video/save_frame {saves} "
        f"times; the directive embeds exactly one video and errors otherwise."
    )


def test_algan_directive_names_are_unique():
    """Names identify an example's output file, so they have to be unique.

    A reused one does not collide -- the directive disambiguates by appending
    how many times it has seen the name -- but which of the two gets ``-1`` then
    depends on the order Sphinx happens to read the pages in, so the asset a
    page embeds is not stable across builds.
    """
    counts = Counter(e.name for e in ALGAN_DIRECTIVES if e.name)
    duplicates = {name: count for name, count in counts.items() if count > 1}
    assert not duplicates, f"reused example names: {duplicates}"


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


@pytest.mark.skipif(not RUN_DOC_RENDERS, reason=SKIP_RENDERS_REASON)
@pytest.mark.parametrize("example", COMPLETE, ids=_ids(COMPLETE))
def test_doc_example_renders(example: DocExample):
    """Complete documented scripts render end to end at SMOKE_TEST.

    The only tier that catches a failure occurring during materialization or
    rasterization rather than during authoring. Opt in with
    ``ALGAN_RUN_DOC_RENDERS=1``; see the module docstring for why it is not on
    by default.
    """
    _run_example(example, render=True)
    # Reclaim between examples. Every example renders in this one process and
    # the arenas are not all released on scope exit, so without this the tier
    # grows without bound and is killed part way through.
    algan.release_torch_memory()
