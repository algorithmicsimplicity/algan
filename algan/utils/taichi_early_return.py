r"""Early ``return`` inside an inlined ``@ti.func``, on either compiler.

Both compilers reject a ``return`` that sits inside *runtime* control flow --
a non-``ti.static`` ``if``, ``for`` or ``while`` -- in a non-real ``@ti.func``
("Return inside non-static if/for is not supported"; quadrants
``ast_transformer.build_Return``, taichi 1.7.4 ``ast_transformer.py:821``).
The func is inlined at every call site, so its body has no exit of its own to
jump to, and a user writing a fragment-shader stage or a scatter func has to
thread the result through a variable by hand. ``return`` at the top level of
the body, or under ``if ti.static(...)`` / ``for ... in ti.static(...)``, has
always been legal: the transformer inlines the taken branch and stops
building the body at the ``return``.

:func:`apply` wraps the one place each compiler turns a func's source into an
AST -- ``FuncBase.get_tree_and_ctx`` on quadrants, ``kernel_impl._get_tree_and_ctx``
on taichi -- and, for a non-kernel, non-real func whose body actually contains
a nested ``return``, rewrites that body into **single-exit form** before the
transformer sees it. Everything else is left byte-for-byte alone: a body with
no nested ``return`` is not touched (the same tree object goes back), so every
existing kernel's IR, and its offline-cache key, is unchanged.

The rewrite
-----------

Let ``h`` be the first top-level statement of the body that contains a nested
``return``. Statements before ``h`` are untouched. From ``h`` on::

    __algan_ret_val = <initialiser>       # omitted for a void func
    __algan_ret_flag = 0
    <the region body[h:], converted>
    return __algan_ret_val                # omitted for a void func

and the conversion is the standard structured single-exit transformation:

* ``return expr`` becomes ``__algan_ret_val = expr; __algan_ret_flag = 1``
  (a bare ``return`` sets only the flag; ``return a, b`` assigns
  ``__algan_ret_val0``, ``__algan_ret_val1``, ...). When it sits directly in
  the function body nothing follows it, so only the value is assigned. The
  statements after an unconditional ``return`` in a block are dead, exactly as
  the transformer treats them, and are dropped.
* After any compound statement that contains a converted ``return``, the rest
  of its block is wrapped in ``if __algan_ret_flag == 0:``. Inside a loop the
  rewrite can break out of (see below), the rest is instead preceded by
  ``if __algan_ret_flag != 0: break`` -- and only when that statement contains
  a loop of its own, since a ``return`` directly under it broke this loop
  already.
* A ``return`` inside a loop is followed by ``break`` **when that loop can be
  broken**: a ``while``, or a ``for`` that is nested in another runtime loop of
  the same body. A statically unrolled ``for ... in ti.static(...)`` cannot be
  broken from under a runtime ``if``, so its whole body is wrapped in
  ``if __algan_ret_flag == 0:`` instead and it runs to its end with the body
  skipped; the idle iterations are the cost, and since an unrolled loop is not
  a parallel one, the result is the same.
* **A ``return`` anywhere inside the func's outermost runtime ``for`` is
  refused**, rather than broken out of or body-guarded. ``break`` there does
  not compile ("Cannot break in the outermost loop" on both compilers), and
  the body guard -- which does compile -- is worse than the compile error it
  replaces: the func may be inlined at a kernel's **top level**, where that
  loop *is* the offloaded one and its iterations run in parallel, while
  ``__algan_ret_flag`` and ``__algan_ret_val`` are declared outside it and
  shared across them. Measured on quadrants, one kernel over one input with
  matches at 1000 and 50000: 20 launches, both indices returned. The pass
  rewrites a func's source, which is parsed once per call site but decided
  before any of them is known, so it cannot tell the offloaded case from the
  serial one and refuses both. **A ``while`` is the spelling that works**: it
  stays serial wherever it is inlined, its ``break`` compiles at a kernel's
  top level, and it stops at the hit instead of idling to the end. Only a
  ``for`` that is the *outermost runtime loop of the body* is refused -- one
  nested in a ``while`` or in another runtime ``for`` is broken out of as
  usual, and a ``ti.static`` loop above it does not count as one.

The value variable
------------------

Taichi types a local from its first assignment, and a local declared inside an
``if`` is not visible after it, so ``__algan_ret_val`` has to be declared at
``h`` with an expression of the right type -- **without** evaluating any
``return`` expression early, which would reorder side effects and read
variables that do not exist yet. Two sources are accepted, in this order:

1. **A return annotation** (``-> ti.f32``, ``-> ti.math.vec4``). A primitive
   declares ``__algan_ret_val: T = 0``; a vector or matrix type declares
   ``__algan_ret_val = T(0)``. This is the robust spelling, and the only one
   that cannot pick the wrong type -- but it is read only for a **single**
   returned value: a ``-> tuple[...]`` annotation is not taken apart, so a
   multi-value ``return`` still needs source 2.
2. **A hoistable return expression**, chosen per returned position: an
   expression built only from parameters, names assigned at the top level
   before ``h``, names never assigned in the function (module globals such as
   ``ti``), literals, attribute reads, constant-index subscripts, the
   arithmetic operators, and calls to a whitelist of pure compiler
   intrinsics (``ti.math.vec4``, ``ti.min``, ``ti.cast``, ``v.dot(...)``, ...).
   That expression is evaluated once at ``h`` as the initialiser -- it is
   pure, so the evaluation is unobservable -- and again where its ``return``
   was. ``//`` and ``%`` are excluded (an integer division by zero traps on the
   CPU), as is every call that is not on the whitelist.

   Which hoistable expression is taken matters, because the initialiser fixes
   the *type* of every answer: a hoistable expression is preferred over a bare
   literal (``return 0`` beside ``return x`` would otherwise make an ``f32``
   func return ``i32``, truncating every value), in source order; and when only
   literals are available, one ``float`` among them widens an ``int``
   initialiser rather than narrowing the others.

When neither applies the body is left untouched and the compiler reports its
usual error, preceded by an :class:`~algan.errors.AlganWarning` naming what
this pass could not do; annotating the return type is the fix. Every
``return`` is cast to the type of the initialiser, as a local assigned twice
would be -- so a func whose only declarable initialiser is an integer literal
(``return 0`` as the fallback of a search whose other ``return`` is an
unhoistable ``f32``) returns integers, with nothing to show for it but the
compiler's own "Assign may lose precision" warning. Annotate that one. Not
handled, and declined the same way: a ``return`` under ``with``,
``try`` or ``match``; a loop whose iterable is an ``IfExp``; a statically
unrolled loop that carries its own ``break``/``continue`` (the body guard
would put it under a runtime ``if``); a ``return`` of a tuple that is not a
literal, or of tuples of differing length, or a bare ``return`` mixed with a
valued one.

``ALGAN_TAICHI_EARLY_RETURN=0`` turns the whole thing off. Like the sibling
warm-start module this is version-gated to the compilers it was verified
against (taichi 1.7.x, quadrants 1.3.x) and is a no-op with a reason on any
other; ``algan check`` prints :func:`skipped_reason`.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import warnings

from algan.environment import env_flag
from algan.errors import AlganWarning

_APPLIED = False

#: Why the rewrite is not installed, or ``None`` when it is (or was never
#: asked for). Read by ``algan check``.
_SKIPPED_REASON = None

#: Names the rewrite introduces. Double-underscored so they cannot collide
#: with anything a shader author would write; a function body is not a class
#: body, so Python does not mangle them.
FLAG = "__algan_ret_flag"
VALUE = "__algan_ret_val"

UNTOUCHED = "untouched"
REWRITTEN = "rewritten"

#: What the hook has decided, by outcome. A body found not to need the rewrite
#: (or refused) is remembered on the raw function and counted once; a rewritten
#: one is counted at every call site that re-parses it, since the rewrite has
#: to run again on each fresh tree. ``rewritten`` staying at zero across a
#: render is the statement that this pass is inert over Algan's own kernels,
#: which is what ``tests/unit_tests/test_taichi_early_return.py`` holds it to.
STATS = {"rewritten": 0, "untouched": 0, "declined": 0}


class EarlyReturnUnsupported(Exception):
    """The body has a nested ``return`` this pass cannot rewrite; says why."""


def skipped_reason():
    """``None`` if the rewrite is live, else why it is not."""
    return _SKIPPED_REASON


def _skip(reason):
    global _SKIPPED_REASON
    _SKIPPED_REASON = reason
    return False


# ---------------------------------------------------------------------------
# AST predicates
# ---------------------------------------------------------------------------


def _is_static_call(node):
    """``ti.static(...)``, ``qd.static(...)`` or a bare ``static(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "static"
    return isinstance(func, ast.Name) and func.id == "static"


def _is_static_if(node):
    return isinstance(node, ast.If) and _is_static_call(node.test)


def _is_static_for(node):
    return isinstance(node, ast.For) and _is_static_call(node.iter)


def _is_loop(node):
    return isinstance(node, (ast.For, ast.While))


#: Statement kinds the rewrite refuses to look through. ``With`` is not a
#: control-flow scope to the transformer, but its body could be anything.
_OPAQUE = (ast.With, ast.AsyncWith, ast.Try, ast.AsyncFor)
if hasattr(ast, "TryStar"):
    _OPAQUE += (ast.TryStar,)
if hasattr(ast, "Match"):
    _OPAQUE += (ast.Match,)

#: Nodes that own their own ``return``s (or none at all); never descended.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _walk(node):
    """The descendants of ``node`` in source order, not entering nested scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPES):
            continue
        yield child
        yield from _walk(child)


def _has_return(node):
    return isinstance(node, ast.Return) or any(
        isinstance(n, ast.Return) for n in _walk(node)
    )


def _has_loop(node):
    return _is_loop(node) or any(_is_loop(n) for n in _walk(node))


def _nested_returns(body):
    """``Return`` nodes under runtime control flow, in source order."""
    found = []

    def visit(stmts, runtime):
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                if runtime:
                    found.append(stmt)
            elif isinstance(stmt, ast.If):
                inner = runtime or not _is_static_if(stmt)
                visit(stmt.body, inner)
                visit(stmt.orelse, inner)
            elif isinstance(stmt, ast.For):
                inner = runtime or not _is_static_for(stmt)
                visit(stmt.body, inner)
                visit(stmt.orelse, inner)
            elif isinstance(stmt, ast.While):
                visit(stmt.body, True)
                visit(stmt.orelse, True)
            elif isinstance(stmt, _OPAQUE):
                # A `return` in here is refused later; count it so the body
                # is not silently passed through as legal.
                for n in _walk(stmt):
                    if isinstance(n, ast.Return):
                        found.append(n)
            # Anything else is a simple statement (or a nested scope).

    visit(body, False)
    return found


def _offloadable_for_with_return(stmts, inside_runtime_loop=False):
    """The func's outermost runtime ``for`` that contains a ``return``, if any.

    "Outermost" is lexical and relative to the func body: a runtime ``for``
    with no ``while`` and no other runtime ``for`` above it. That is the loop
    that becomes the kernel's offloaded, parallel one when the func is inlined
    at a kernel's top level -- which the pass cannot see from the source it is
    rewriting -- so a ``return`` anywhere inside one is refused rather than
    guarded. A ``ti.static`` loop above it is unrolled and does not shelter it;
    everything below the first runtime loop is breakable and fine.
    """
    for stmt in stmts:
        if isinstance(stmt, ast.While):
            found = _offloadable_for_with_return(stmt.body, True)
        elif isinstance(stmt, ast.For):
            if _is_static_for(stmt):
                found = _offloadable_for_with_return(stmt.body, inside_runtime_loop)
            elif inside_runtime_loop:
                found = None  # breakable, and so is every loop under it
            else:
                found = stmt if _has_return(stmt) else None
        elif isinstance(stmt, ast.If):
            found = _offloadable_for_with_return(
                stmt.body, inside_runtime_loop
            ) or _offloadable_for_with_return(stmt.orelse, inside_runtime_loop)
        else:
            found = None
        if found is not None:
            return found
    return None


def _owns_break_or_continue(loop):
    """Whether ``loop``'s body has a ``break``/``continue`` that targets *it*."""

    def visit(stmts):
        for stmt in stmts:
            if isinstance(stmt, (ast.Break, ast.Continue)):
                return True
            if isinstance(stmt, ast.If):
                if visit(stmt.body) or visit(stmt.orelse):
                    return True
            elif isinstance(stmt, _OPAQUE):  # noqa: SIM102 -- the arms below differ
                if any(isinstance(n, (ast.Break, ast.Continue)) for n in _walk(stmt)):
                    return True
            # A nested loop owns its own break/continue.
        return False

    return visit(loop.body)


# ---------------------------------------------------------------------------
# Hoistability: which return expressions may be evaluated early at `h`
# ---------------------------------------------------------------------------

_PURE_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.MatMult,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
)
_PURE_UNARY = (ast.UAdd, ast.USub, ast.Not, ast.Invert)

#: Builtins the transformer evaluates as pure conversions.
_PURE_BUILTINS = frozenset({"float", "int", "bool", "abs", "min", "max", "round"})

#: Attribute names that, called on a module (``ti.min``, ``ti.math.vec4``,
#: ``tm.normalize``) or on a value (``v.dot(w)``, ``v.norm()``), denote a pure
#: compiler intrinsic on both compilers. ``random`` is deliberately absent.
_PURE_CALLEES = frozenset(
    {
        # constructors
        "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4",
        "mat2", "mat3", "mat4", "Vector", "Matrix", "eye", "zero", "one",
        # conversions
        "cast", "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
        "bit_cast",
        # scalar math
        "min", "max", "abs", "sqrt", "rsqrt", "exp", "exp2", "log", "log2", "pow",
        "sin", "cos", "tan", "asin", "acos", "atan2", "sinh", "cosh", "tanh", "asinh",
        "acosh", "atanh", "cot", "floor", "ceil", "round", "select", "sign", "fract",
        "mod", "fmod", "radians", "degrees", "isnan", "isinf", "clamp", "step",
        "smoothstep", "mix", "popcnt",
        # vector / matrix math
        "normalize", "normalized", "length", "distance", "dot", "cross", "norm",
        "norm_sqr", "outer_product", "transpose", "inverse", "determinant", "trace",
        "sum", "any", "all", "reflect", "refract", "rot2", "rot3", "rotation2d",
        "rotation3d", "scale", "translate",
    }
)  # fmt: skip


def _literal_value(node):
    """The number ``node`` spells out, or ``None`` if it is not a literal.

    ``-1`` is a ``UnaryOp`` over a ``Constant``, not a ``Constant``, and the
    difference decides whether an expression can type the value variable, so
    the sign is folded here rather than being taken for an expression.
    """
    if isinstance(node, ast.Constant) and type(node.value) in (bool, int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _literal_value(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else +inner
    return None


def _const_index(node):
    """A non-negative integer literal, or a tuple of them."""
    if isinstance(node, ast.Tuple):
        return all(_const_index(e) for e in node.elts)
    return (
        isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0
    )


def _hoistable(node, name_ok):
    """Whether ``node`` is a pure expression over names bound at ``h``."""
    if isinstance(node, ast.Constant):
        return type(node.value) in (bool, int, float)
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and name_ok(node.id)
    if isinstance(node, ast.Attribute):
        return _hoistable(node.value, name_ok)
    if isinstance(node, ast.Subscript):
        return _hoistable(node.value, name_ok) and _const_index(node.slice)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_hoistable(e, name_ok) for e in node.elts)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, _PURE_UNARY) and _hoistable(node.operand, name_ok)
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, _PURE_BINOPS)
            and _hoistable(node.left, name_ok)
            and _hoistable(node.right, name_ok)
        )
    if isinstance(node, ast.BoolOp):
        return all(_hoistable(v, name_ok) for v in node.values)
    if isinstance(node, ast.Compare):
        return _hoistable(node.left, name_ok) and all(
            _hoistable(c, name_ok) for c in node.comparators
        )
    if isinstance(node, ast.IfExp):
        return all(_hoistable(n, name_ok) for n in (node.test, node.body, node.orelse))
    if isinstance(node, ast.Call):
        callee = node.func
        if isinstance(callee, ast.Name):
            callee_ok = callee.id in _PURE_BUILTINS and name_ok(callee.id)
        elif isinstance(callee, ast.Attribute):
            callee_ok = callee.attr in _PURE_CALLEES and _hoistable(
                callee.value, name_ok
            )
        else:
            callee_ok = False
        return (
            callee_ok
            and all(
                not isinstance(a, ast.Starred) and _hoistable(a, name_ok)
                for a in node.args
            )
            and all(
                k.arg is not None and _hoistable(k.value, name_ok)
                for k in node.keywords
            )
        )
    return False


def _bound_at(func_def, h):
    """``name_ok`` for the hoist point: bound there, and not shadowed later.

    Parameters and names assigned directly in ``body[:h]`` are declared at
    ``h``; a name assigned nowhere in the function is a global (``ti``, a
    constant, a helper). A name assigned only later, or only inside a nested
    block (Taichi scopes locals to their block), is not.
    """
    params = {a.arg for a in ast.walk(func_def.args) if isinstance(a, ast.arg)}
    assigned = {
        n.id
        for n in _walk(func_def)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    top = set()
    for stmt in func_def.body[:h]:
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            targets = [stmt.target]
        else:
            continue
        for target in targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    top.add(n.id)
    declared = params | top

    def name_ok(name):
        return name in declared or name not in assigned

    return name_ok


def _initialiser(elements, name_ok):
    """What to declare one value variable from, given every ``return``'s
    expression in that position, or ``None`` when none of them will do.

    Taichi types a local from its first assignment and *casts* every later
    one, so the initialiser does not merely have to be evaluable at ``h`` --
    it decides the type of every answer the func can give. A bare literal
    carries no type from the function's own arithmetic (``return 0`` beside
    ``return x`` would make an ``f32`` func return ``i32``, truncating every
    value, with nothing but the compiler's "may lose precision" warning to
    say so), so a hoistable expression is preferred over one, in source
    order, and a literal is used only when it is all there is. Among
    literals, one ``float`` answer makes the variable a float: an integer
    literal is widened rather than narrowing the others.
    """
    hoistable = [e for e in elements if _hoistable(e, name_ok)]
    if not hoistable:
        return None
    typed = next((e for e in hoistable if _literal_value(e) is None), None)
    if typed is not None:
        return copy.deepcopy(typed)
    chosen = hoistable[0]
    value = _literal_value(chosen)
    if isinstance(value, int) and any(
        isinstance(_literal_value(e), float) for e in elements
    ):
        return ast.copy_location(ast.Constant(value=float(value)), chosen)
    return copy.deepcopy(chosen)


# ---------------------------------------------------------------------------
# The rewrite
# ---------------------------------------------------------------------------


class _LoopInfo:
    """What the conversion needs to know about the innermost enclosing loop."""

    __slots__ = ("is_static", "breakable", "runtime_above")

    def __init__(self, node, parent):
        self.is_static = _is_static_for(node)
        # Is there a runtime loop at or above this one (within the region)?
        self.runtime_above = (not self.is_static) or (
            parent is not None and parent.runtime_above
        )
        if isinstance(node, ast.While):
            self.breakable = True
        elif self.is_static:
            self.breakable = False
        else:
            # A runtime `for`: only when it is certainly not the outermost
            # loop of the kernel it is inlined into.
            self.breakable = parent is not None and parent.runtime_above


def _name(ident, ctx, anchor):
    return ast.copy_location(ast.Name(id=ident, ctx=ctx), anchor)


def _assign(ident, value, anchor):
    return ast.copy_location(
        ast.Assign(targets=[_name(ident, ast.Store(), anchor)], value=value), anchor
    )


def _flag_test(op, anchor):
    return ast.copy_location(
        ast.Compare(
            left=_name(FLAG, ast.Load(), anchor),
            ops=[op()],
            comparators=[ast.copy_location(ast.Constant(value=0), anchor)],
        ),
        anchor,
    )


def _guard(body, anchor):
    """``if __algan_ret_flag == 0: <body>``."""
    return ast.copy_location(
        ast.If(test=_flag_test(ast.Eq, anchor), body=body, orelse=[]), anchor
    )


def _propagate_break(anchor):
    """``if __algan_ret_flag != 0: break``."""
    return ast.copy_location(
        ast.If(
            test=_flag_test(ast.NotEq, anchor),
            body=[ast.copy_location(ast.Break(), anchor)],
            orelse=[],
        ),
        anchor,
    )


def _return_arity(stmt):
    """``None`` for a bare return, else the number of values."""
    value = stmt.value
    if value is None or (isinstance(value, ast.Constant) and value.value is None):
        return None
    if isinstance(value, ast.Tuple):
        return len(value.elts)
    return 1


def _return_elements(stmt):
    value = stmt.value
    return list(value.elts) if isinstance(value, ast.Tuple) else [value]


class _Rewriter:
    def __init__(self, func_def, annotation_kind):
        self.func_def = func_def
        self.annotation_kind = annotation_kind
        self.value_names = ()

    # -- driver -----------------------------------------------------------

    def run(self):
        body = self.func_def.body
        if not _nested_returns(body):
            return UNTOUCHED
        h = next(i for i, stmt in enumerate(body) if _needs(stmt))
        region = body[h:]
        self._check_supported(region)

        returns = [n for stmt in region for n in _returns_in(stmt)]
        arities = {_return_arity(r) for r in returns}
        if len(arities) != 1:
            raise EarlyReturnUnsupported(
                "its returns disagree on what they return (a bare `return` next to "
                "`return value`, or tuples of different length)"
            )
        arity = arities.pop()
        anchor = region[0]
        decls = []
        if arity is None:
            self.value_names = ()
        else:
            self.value_names = (
                (VALUE,) if arity == 1 else tuple(f"{VALUE}{i}" for i in range(arity))
            )
            decls.extend(self._declarations(returns, arity, h, anchor))
        decls.append(
            _assign(FLAG, ast.copy_location(ast.Constant(value=0), anchor), anchor)
        )

        # Convert a *copy* of the region. The conversion rewrites compound
        # statements in place as it walks, and can still refuse one it reaches
        # (a nested scope carrying a `return` is not something
        # `_check_supported` can see), which would otherwise leave the caller
        # holding a half-rewritten body -- values assigned to a variable the
        # declarations were never spliced in to declare -- and hand *that* to
        # the compiler. Declining has to mean the body is as the user wrote it.
        converted = self._convert_block(
            copy.deepcopy(region), None, at_function_top=True
        )

        tail = []
        if arity is not None:
            last = region[-1]
            if arity == 1:
                value = _name(VALUE, ast.Load(), last)
            else:
                value = ast.copy_location(
                    ast.Tuple(
                        elts=[_name(n, ast.Load(), last) for n in self.value_names],
                        ctx=ast.Load(),
                    ),
                    last,
                )
            tail.append(ast.copy_location(ast.Return(value=value), last))

        body[h:] = decls + converted + tail
        ast.fix_missing_locations(self.func_def)
        return REWRITTEN

    def _check_supported(self, region):
        offloadable = _offloadable_for_with_return(region)
        if offloadable is not None:
            raise EarlyReturnUnsupported(
                "a `return` inside the func's outermost `for`. Wherever the func is "
                "inlined at a kernel's top level that loop is the offloaded one and "
                "its iterations run in parallel, so the flag and value this rewrite "
                "declares outside it would be shared by iterations racing to set "
                "them, and the answer would vary between launches; `break` there "
                "does not compile either. Write the search as a `while`, whose "
                "`break` keeps the loop serial and stops it at the hit"
            )
        for stmt in region:
            for node in [stmt, *_walk(stmt)]:
                if isinstance(node, _OPAQUE) and _has_return(node):
                    raise EarlyReturnUnsupported(
                        f"a `return` inside a `{type(node).__name__.lower()}` block"
                    )
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.iter, ast.IfExp)
                    and _has_return(node)
                ):
                    raise EarlyReturnUnsupported(
                        "a `return` inside a loop whose iterable is a conditional expression"
                    )
                if (
                    _is_static_for(node)
                    and _has_return(node)
                    and _owns_break_or_continue(node)
                ):
                    raise EarlyReturnUnsupported(
                        "a `return` inside a `ti.static` loop that also has its own "
                        "`break`/`continue` (the rewrite would put it under a runtime `if`)"
                    )
                if isinstance(node, ast.Return) and node.value is not None:
                    value = node.value
                    if isinstance(value, ast.Tuple) and any(
                        isinstance(e, ast.Starred) for e in value.elts
                    ):
                        raise EarlyReturnUnsupported("a starred tuple return")

    def _declarations(self, returns, arity, h, anchor):
        if (
            arity == 1
            and self.annotation_kind is not None
            and self.func_def.returns is not None
        ):
            annotation = copy.deepcopy(self.func_def.returns)
            zero = ast.copy_location(ast.Constant(value=0), anchor)
            if self.annotation_kind == "primitive":
                return [
                    ast.copy_location(
                        ast.AnnAssign(
                            target=_name(VALUE, ast.Store(), anchor),
                            annotation=annotation,
                            value=zero,
                            simple=1,
                        ),
                        anchor,
                    )
                ]
            if self.annotation_kind == "matrix":
                call = ast.copy_location(
                    ast.Call(func=annotation, args=[zero], keywords=[]), anchor
                )
                return [_assign(VALUE, call, anchor)]
        name_ok = _bound_at(self.func_def, h)
        elements = [_return_elements(stmt) for stmt in returns]
        decls = []
        for position, ident in enumerate(self.value_names):
            initialiser = _initialiser([e[position] for e in elements], name_ok)
            if initialiser is None:
                raise EarlyReturnUnsupported(
                    "no `return` expression is pure and built only from parameters, "
                    "names assigned before the first early return, and compiler "
                    "intrinsics, so the result variable cannot be declared; annotate "
                    "the return type (`-> ti.f32`, `-> ti.math.vec4`) to give it one"
                )
            decls.append(_assign(ident, initialiser, anchor))
        return decls

    # -- conversion -------------------------------------------------------

    def _convert_block(self, stmts, loop, at_function_top=False):
        out = []
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Return):
                out.extend(self._convert_return(stmt, loop, at_function_top))
                return out  # what follows an unconditional return is dead
            if not _has_return(stmt):
                out.append(stmt)
                continue
            out.append(self._convert_compound(stmt, loop))
            rest = stmts[i + 1 :]
            if not rest:
                return out
            converted = self._convert_block(rest, loop)
            if loop is not None and loop.breakable:
                if _has_loop(stmt):
                    out.append(_propagate_break(rest[0]))
                out.extend(converted)
            else:
                out.append(_guard(converted, rest[0]))
            return out
        return out

    def _convert_return(self, stmt, loop, at_function_top):
        out = []
        if self.value_names:
            for ident, element in zip(self.value_names, _return_elements(stmt)):
                out.append(_assign(ident, element, stmt))
        if not at_function_top:
            out.append(
                _assign(FLAG, ast.copy_location(ast.Constant(value=1), stmt), stmt)
            )
            if loop is not None and loop.breakable:
                out.append(ast.copy_location(ast.Break(), stmt))
        return out

    def _convert_compound(self, stmt, loop):
        if isinstance(stmt, ast.If):
            stmt.body = self._convert_block(stmt.body, loop)
            stmt.orelse = self._convert_block(stmt.orelse, loop)
            return stmt
        if _is_loop(stmt):
            info = _LoopInfo(stmt, loop)
            body = self._convert_block(stmt.body, info)
            if not info.breakable:
                body = [_guard(body, stmt.body[0])]
            stmt.body = body
            return stmt
        raise EarlyReturnUnsupported(  # pragma: no cover - refused by _check_supported
            f"a `return` inside a `{type(stmt).__name__.lower()}` block"
        )


def _needs(stmt):
    """Whether a top-level statement contains a nested (illegal) return."""
    return bool(_nested_returns([stmt]))


def _returns_in(stmt):
    if isinstance(stmt, ast.Return):
        yield stmt
    for n in _walk(stmt):
        if isinstance(n, ast.Return):
            yield n


def rewrite_function_def(func_def, annotation_kind=None):
    """Rewrite ``func_def``'s body in place if it needs it.

    Returns :data:`UNTOUCHED` when the body has no ``return`` under runtime
    control flow (and was not modified), :data:`REWRITTEN` when it was, and
    raises :class:`EarlyReturnUnsupported` -- leaving the body untouched --
    when it has one this pass cannot handle. ``annotation_kind`` is
    ``"primitive"``, ``"matrix"`` or ``None`` for how the func's return
    annotation resolved on the live compiler.
    """
    return _Rewriter(func_def, annotation_kind).run()


def rewrite_source(source, annotation_kind=None):
    """The rewrite as source text, for tests and for reading what it does."""
    tree = ast.parse(source)
    func_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    rewrite_function_def(func_def, annotation_kind)
    return ast.unparse(func_def)


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------

_MATRIX_TYPE = None
_PRIMITIVE_TYPE_IDS = frozenset()


def _annotation_kind(func):
    """How the decorated func's return annotation resolved, if at all."""
    return_type = getattr(func, "return_type", None)
    if not return_type or len(return_type) != 1:
        return None
    resolved = return_type[0]
    if id(resolved) in _PRIMITIVE_TYPE_IDS:
        return "primitive"
    if _MATRIX_TYPE is not None and isinstance(resolved, _MATRIX_TYPE):
        return "matrix"
    return None


def _rewrite_tree(tree, func):
    """Rewrite the parsed func body when it needs it; cache the no-op case.

    The decision is a pure function of the source, which the wrapper pins for
    its lifetime, so a body found not to need the rewrite is remembered on the
    raw function and never walked again -- funcs are re-parsed and
    re-transformed at every call site, and this pass must not become a
    frontend cost of its own.
    """
    func_def = tree.body[0] if tree.body else None
    if not isinstance(func_def, ast.FunctionDef):
        return
    raw = getattr(func, "func", None)
    cached = getattr(raw, "_algan_early_return", None)
    if cached is not None and cached != REWRITTEN:
        return  # UNTOUCHED, or declined (and already warned)
    try:
        outcome = rewrite_function_def(func_def, _annotation_kind(func))
        STATS[outcome] += 1
    except EarlyReturnUnsupported as exc:
        outcome = f"declined: {exc}"
        STATS["declined"] += 1
        warnings.warn(
            f"`{func_def.name}` has a `return` inside runtime control flow that "
            f"the single-exit rewrite (algan.utils.taichi_early_return) cannot "
            f"handle: {exc}. The compiler will now reject it.",
            AlganWarning,
            stacklevel=2,
        )
    if cached is None:
        with contextlib.suppress(AttributeError):
            raw._algan_early_return = outcome


def _positional_or_keyword(args, kwargs, name, index, default):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _load_type_probes():
    global _MATRIX_TYPE, _PRIMITIVE_TYPE_IDS
    from algan.taichi_compat import submodule

    _MATRIX_TYPE = submodule("lang.matrix").MatrixType
    _PRIMITIVE_TYPE_IDS = frozenset(submodule("types.primitive_types").type_ids)


def apply():
    """Install the rewrite (idempotent, safe no-op on mismatch).

    Installed after :func:`algan.utils.taichi_warmstart.apply`, and wraps
    whatever is bound then -- on taichi that is the warm-start's memoized
    ``_get_tree_and_ctx``, which this composes with rather than replaces.
    """
    global _APPLIED, _SKIPPED_REASON
    if _APPLIED:
        return
    if not env_flag("ALGAN_TAICHI_EARLY_RETURN", True):
        _SKIPPED_REASON = "ALGAN_TAICHI_EARLY_RETURN=0"
        return

    try:
        from algan.taichi_compat import BACKEND, backend_version
    except Exception:
        _SKIPPED_REASON = "the kernel compiler could not be imported"
        return

    installer = {"taichi": _apply_taichi, "quadrants": _apply_quadrants}.get(BACKEND)
    if installer is None:
        _SKIPPED_REASON = f"no early-return rewrite is written for {BACKEND!r}"
        return
    _SKIPPED_REASON = None
    _APPLIED = bool(installer(tuple(backend_version())))


def _apply_quadrants(version):
    """quadrants 1.3.x: wrap ``FuncBase.get_tree_and_ctx``.

    ``Func.__call__`` reaches it as a method with ``is_kernel=False``;
    ``Kernel.materialize`` (``is_kernel`` left at its default) and a real
    function's ``do_compile`` (``is_real_function=True``) pass through.
    """
    from algan.taichi_compat import submodule

    try:
        func_base = submodule("lang._func_base")
        _load_type_probes()
    except Exception:
        return _skip("quadrants' func internals are not where 1.3 keeps them")
    if version[:2] != (1, 3):
        return _skip(
            f"the early-return rewrite was verified against quadrants 1.3; this is "
            f"{'.'.join(str(p) for p in version) or 'an unknown version'}"
        )
    base = getattr(func_base, "FuncBase", None)
    orig = getattr(base, "get_tree_and_ctx", None)
    if orig is None:
        return _skip("quadrants 1.3's FuncBase.get_tree_and_ctx has moved")

    # (self, py_args, template_slot_locations=(), is_kernel=True,
    #  arg_features=None, ast_builder=None, is_real_function=False, ...)
    def get_tree_and_ctx(self, *args, **kwargs):
        tree, ctx = orig(self, *args, **kwargs)
        if not _positional_or_keyword(args, kwargs, "is_kernel", 2, True) and not (
            _positional_or_keyword(args, kwargs, "is_real_function", 5, False)
        ):
            _rewrite_tree(tree, self)
        return tree, ctx

    get_tree_and_ctx._algan_original = orig
    base.get_tree_and_ctx = get_tree_and_ctx
    return True


def _apply_taichi(version):
    """taichi 1.7.x: wrap ``kernel_impl._get_tree_and_ctx``.

    A module-level function every caller looks up by name at call time, which
    is why the warm-start memo could replace it and why this can wrap the
    replacement.
    """
    from algan.taichi_compat import submodule

    try:
        kernel_impl = submodule("lang.kernel_impl")
        _load_type_probes()
    except Exception:
        return _skip("taichi's kernel_impl internals are not where 1.7 keeps them")
    if version[:2] != (1, 7):
        return _skip(
            f"the early-return rewrite was verified against taichi 1.7; this is "
            f"{'.'.join(str(p) for p in version) or 'an unknown version'}"
        )
    orig = getattr(kernel_impl, "_get_tree_and_ctx", None)
    if orig is None:
        return _skip("taichi 1.7's _get_tree_and_ctx has moved")

    # (self, excluded_parameters=(), is_kernel=True, arg_features=None,
    #  args=None, ast_builder=None, is_real_function=False)
    def _get_tree_and_ctx(self, *args, **kwargs):
        tree, ctx = orig(self, *args, **kwargs)
        if not _positional_or_keyword(args, kwargs, "is_kernel", 1, True) and not (
            _positional_or_keyword(args, kwargs, "is_real_function", 5, False)
        ):
            _rewrite_tree(tree, self)
        return tree, ctx

    _get_tree_and_ctx._algan_original = orig
    kernel_impl._get_tree_and_ctx = _get_tree_and_ctx
    return True
