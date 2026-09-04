r"""Warm-start accelerator for kernel materialization, on either compiler.

Even with a hot offline cache, every program run pays a large Python-side
"materialize" cost per kernel instantiation before the compiler can even *look
up* the cache: the offline-cache key is a hash of the frontend IR, and building
that IR (``transform_tree``) walks the whole kernel AST in Python. For this
project's monolithic wavefront kernels that is tens of seconds per run, and
profiling shows most of it is redundant:

* ``Builder.__call__`` eagerly computes ``ctx.get_pos_info(node)`` -- a
  caret-underlined, textwrap-wrapped source excerpt -- for every statement
  and expression node it visits, purely to attach debug info (~45% of the
  transform on taichi, ~41% on quadrants).
* Non-real ``ti.func``\\ s are inlined by re-running source retrieval
  (``getsourcelines`` + per-line ``textwrap.fill``) and a full re-transform
  of the func body at **every call site**, and the same kernel body is
  re-transformed per template instantiation.

Both are pure functions of the decorated function's source and the node
position, so :func:`apply` memoizes them from outside the compiler:

* ``get_pos_info`` results are cached per decorated function, keyed by
  (context line/indent offsets, node position, node class). Both backends.
* The source retrieval behind a transform -- taichi's ``_get_tree_and_ctx``,
  quadrants' ``get_source_info_and_src`` -- is cached per raw function object.
  The *AST itself is re-parsed fresh on every call*: the transformer annotates
  AST nodes in place (``node.ptr``), so a tree must never be shared between
  transforms.

Byte-identical by construction -- the memoized values are exactly the strings
the original code recomputes (validated against a live run: 0 mismatches over
136k calls, and hash-identical rendered output). ``ALGAN_TAICHI_WARMSTART=0``
turns the whole thing off; ``ALGAN_TAICHI_WARMSTART_VERIFY=1`` recomputes every
memoized result the original way and raises on any byte difference
(``benchmarks/_taichi_warmstart_check.py`` is what runs it).

Each patch replicates one compiler's internals verbatim, so each applies only
to the version it was written against -- taichi 1.7.x, quadrants 1.3.x -- and
degrades to a no-op anywhere else. **That no-op is the trap this module has
already fallen into once**: the quadrants port did not exist when the backend
became selectable, the taichi version gate refused to fire, and a quadrants
render silently paid ~25 s of frontend per process against taichi's ~4 s until
`taichi_patches/PLAN.md` §6.1 measured it. So a version this file does not know
is reported through :func:`skipped_reason` rather than passing in silence, and
``algan check`` prints it.
"""

from __future__ import annotations

import ast
import contextlib
import sys
import textwrap

from algan.environment import env_flag

_APPLIED = False

#: Why the accelerator is not installed, or ``None`` when it is (or was never
#: asked for). Read by ``algan check`` -- a compiler whose frontend cost is
#: being paid in full should say so somewhere a user looks.
_SKIPPED_REASON = None


def skipped_reason():
    """``None`` if the memoization is live, else why it is not."""
    return _SKIPPED_REASON


def _skip(reason):
    global _SKIPPED_REASON
    _SKIPPED_REASON = reason
    return False


def apply():
    """Install the memoizing patches (idempotent, safe no-op on mismatch)."""
    global _APPLIED, _SKIPPED_REASON
    if _APPLIED:
        return
    if not env_flag("ALGAN_TAICHI_WARMSTART", True):
        _SKIPPED_REASON = "ALGAN_TAICHI_WARMSTART=0"
        return

    try:
        from algan.taichi_compat import BACKEND, backend_version
    except Exception:
        _SKIPPED_REASON = "the kernel compiler could not be imported"
        return

    installer = {"taichi": _apply_taichi, "quadrants": _apply_quadrants}.get(BACKEND)
    if installer is None:
        _SKIPPED_REASON = f"no warm-start patch is written for {BACKEND!r}"
        return
    _SKIPPED_REASON = None
    _APPLIED = bool(installer(tuple(backend_version())))


def _wrap80(text):
    """``TextWrapper(width=80).wrap(text)``, with the common case inlined.

    Provably equivalent for the dominant case -- a line of at most 80
    characters whose only whitespace is plain spaces -- where the wrapped
    result is ``[]`` for an all-space line and ``[line.rstrip(" ")]``
    otherwise: leading spaces are kept on the first output line and the
    trailing whitespace chunk is dropped. Anything else (long lines, tabs,
    unicode whitespace) falls back to the real ``TextWrapper``.
    """
    if len(text) <= 80:
        simple = True
        for c in text:
            if c.isspace() and c != " ":
                simple = False
                break
        if simple:
            stripped = text.rstrip(" ")
            return [stripped] if stripped else []
    return textwrap.TextWrapper(width=80).wrap(text)


#: The exact keyword arguments quadrants fills source lines with
#: (``_func_base.get_tree_and_ctx``). Anything else goes to the real
#: ``textwrap.fill`` unmemoized, so the shim can never answer a question it was
#: not asked.
_FILL_KWARGS = {"tabsize": 4, "width": 9999}
_MISS = object()


class _MemoizingTextwrap:
    """``textwrap`` with :func:`fill` memoized, for one module's globals.

    ``get_tree_and_ctx`` re-fills every source line of a function on every
    transform, which for Algan's kernels is ~146,000 calls and ~3.7 s per
    process for a few thousand distinct lines. The result is a pure function of
    the line, so it is cached on the way through.

    A shim rather than a rewrite of ``get_tree_and_ctx``: that method is a
    hundred lines deep in quadrants' pruning machinery, and copying it here to
    change one list comprehension would tie this file to internals that move
    between releases. Rebinding the ``textwrap`` name in the one module that
    fills source lines touches nothing else -- ``__getattr__`` delegates every
    other attribute to the real module, so ``dedent`` and anything added later
    behave exactly as before.
    """

    def __init__(self, verify=False):
        self._cache = {}
        self._verify = verify

    def fill(self, text, **kwargs):
        if kwargs != _FILL_KWARGS:
            return textwrap.fill(text, **kwargs)
        hit = self._cache.get(text, _MISS)
        if hit is _MISS:
            hit = textwrap.fill(text, **kwargs)
            # Bounded so that a program generating kernels from unique source
            # (exec'd templates, notebook cells) cannot grow this without end.
            # Clearing wholesale rather than evicting: the next transform
            # refills what it needs, and an LRU's bookkeeping would cost more
            # than the misses it saves at this hit rate.
            if len(self._cache) >= 100_000:
                self._cache.clear()
            self._cache[text] = hit
        elif self._verify:
            ref = textwrap.fill(text, **kwargs)
            if ref != hit:
                raise RuntimeError(
                    f"taichi_warmstart textwrap.fill mismatch for {text!r}:\n"
                    f"cached: {hit!r}\nref:    {ref!r}"
                )
        return hit

    def __getattr__(self, name):
        return getattr(textwrap, name)


def _make_fast_get_pos_info(guard_src_bounds):
    """Build the cache-miss path: one compiler's ``get_pos_info``, verbatim.

    ``guard_src_bounds`` is the one place the two implementations differ.
    Quadrants added ``if node.lineno - 1 < len(self.src)`` around the
    single-line branch, so a node past the end of the captured source yields a
    header and no excerpt; taichi 1.7.4 has no such guard and raises
    ``IndexError``. Reproducing each compiler's own behaviour matters more
    than picking the nicer one -- a memo that changed it would be a behaviour
    patch wearing a performance patch's clothes.
    """

    def _fast_get_pos_info(self, node):
        msg = (
            f'File "{self.file}", line {node.lineno + self.lineno_offset},'
            f" in {self.func.func.__name__}:\n"
        )
        col_offset = self.indent + node.col_offset
        end_col_offset = self.indent + node.end_col_offset

        def gen_line(code, hint):
            hint += " " * (len(code) - len(hint))
            code = _wrap80(code)
            hint = _wrap80(hint)
            if not len(code):
                return "\n\n"
            return "".join([c + "\n" + h + "\n" for c, h in zip(code, hint)])

        if node.lineno == node.end_lineno:
            if not (guard_src_bounds and node.lineno - 1 >= len(self.src)):
                hint = " " * col_offset + "^" * (end_col_offset - col_offset)
                msg += gen_line(self.src[node.lineno - 1], hint)
        else:
            node_type = node.__class__.__name__

            if node_type in ["For", "While", "FunctionDef", "If"]:
                end_lineno = max(node.body[0].lineno - 1, node.lineno)
            else:
                end_lineno = node.end_lineno

            for i in range(node.lineno - 1, end_lineno):
                last = len(self.src[i])
                while last > 0 and (
                    self.src[i][last - 1].isspace()
                    or not self.src[i][last - 1].isprintable()
                ):
                    last -= 1
                first = 0
                while first < len(self.src[i]) and (
                    self.src[i][first].isspace() or not self.src[i][first].isprintable()
                ):
                    first += 1
                if i == node.lineno - 1:
                    hint = " " * col_offset + "^" * (last - col_offset)
                elif i == node.end_lineno - 1:
                    hint = " " * first + "^" * (end_col_offset - first)
                elif first < last:
                    hint = " " * first + "^" * (last - first)
                else:
                    hint = ""
                msg += gen_line(self.src[i], hint)
        return msg

    return _fast_get_pos_info


def _install_pos_info_memo(ctx_cls, guard_src_bounds):
    """Memoize ``ctx_cls.get_pos_info`` per decorated function.

    The cache dict lives on the Kernel/Func wrapper (``ctx.func``), so the
    entries share the wrapper's lifetime (dynamically composed pipeline funcs
    are collected together with their caches). The wrapper pins its source, so
    identical keys always map to identical strings.
    """
    orig_get_pos_info = ctx_cls.get_pos_info
    fast_get_pos_info = _make_fast_get_pos_info(guard_src_bounds)
    verify = env_flag("ALGAN_TAICHI_WARMSTART_VERIFY", False)

    def _memoized_get_pos_info(self, node):
        owner = self.func
        if owner is None:
            return orig_get_pos_info(self, node)
        cache = getattr(owner, "_algan_pos_info_cache", None)
        if cache is None:
            cache = {}
            try:
                owner._algan_pos_info_cache = cache
            except AttributeError:
                return orig_get_pos_info(self, node)
        key = (
            self.lineno_offset,
            self.indent,
            node.lineno,
            node.col_offset,
            getattr(node, "end_lineno", None),
            getattr(node, "end_col_offset", None),
            node.__class__.__name__,
        )
        hit = cache.get(key)
        if hit is None:
            hit = fast_get_pos_info(self, node)
            cache[key] = hit
        if verify:
            ref = orig_get_pos_info(self, node)
            if ref != hit:
                raise RuntimeError(
                    "taichi_warmstart get_pos_info mismatch for "
                    f"{key}:\nfast: {hit!r}\nref:  {ref!r}"
                )
        return hit

    # Kept reachable so a test can compare the memo against the implementation
    # it replaced, on whichever compiler is live, without compiling a kernel.
    _memoized_get_pos_info._algan_original = orig_get_pos_info
    ctx_cls.get_pos_info = _memoized_get_pos_info


def _apply_taichi(version):
    """taichi 1.7.x: the pos-info memo, and the source half of the transform."""
    from algan.taichi_compat import submodule

    try:
        _atu = submodule("lang.ast.ast_transformer_utils")
        _ki = submodule("lang.kernel_impl")
        _wrap_inspect = submodule("lang._wrap_inspect")
        getsourcefile = _wrap_inspect.getsourcefile
        getsourcelines = _wrap_inspect.getsourcelines
    except Exception:
        return _skip("taichi's transformer internals are not where 1.7 keeps them")

    if version[:2] != (1, 7):
        return _skip(
            f"the warm-start patch replicates taichi 1.7 internals; this is "
            f"{'.'.join(str(p) for p in version) or 'an unknown version'}"
        )
    ctx_cls = getattr(_atu, "ASTTransformerContext", None)
    if (
        ctx_cls is None
        or not hasattr(ctx_cls, "get_pos_info")
        or not hasattr(_ki, "_get_tree_and_ctx")
        or not hasattr(_ki, "_get_global_vars")
    ):
        return _skip("taichi 1.7's transformer entry points have moved")

    _install_pos_info_memo(ctx_cls, guard_src_bounds=False)

    # --- _get_tree_and_ctx source memo ----------------------------------
    # Verbatim copy of taichi 1.7.4's _get_tree_and_ctx with the source
    # retrieval (getsourcelines + per-line textwrap.fill) cached on the raw
    # function object. Only the immutable strings are reused; the AST is
    # parsed fresh every call (see module docstring).
    def _cached_get_tree_and_ctx(
        self,
        excluded_parameters=(),
        is_kernel=True,
        arg_features=None,
        args=None,
        ast_builder=None,
        is_real_function=False,
    ):
        cached = getattr(self.func, "_algan_src_cache", None)
        if cached is None:
            file = getsourcefile(self.func)
            src, start_lineno = getsourcelines(self.func)
            src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
            cached = (file, src, start_lineno)
            with contextlib.suppress(AttributeError):
                self.func._algan_src_cache = cached
        file, src, start_lineno = cached
        tree = ast.parse(textwrap.dedent("\n".join(src)))

        func_body = tree.body[0]
        func_body.decorator_list = []

        global_vars = _ki._get_global_vars(self.func)

        if is_kernel or is_real_function:
            # inject template parameters into globals
            for i in self.template_slot_locations:
                template_var_name = self.arguments[i].name
                global_vars[template_var_name] = args[i]

        return tree, ctx_cls(
            excluded_parameters=excluded_parameters,
            is_kernel=is_kernel,
            func=self,
            arg_features=arg_features,
            global_vars=global_vars,
            argument_data=args,
            src=src,
            start_lineno=start_lineno,
            file=file,
            ast_builder=ast_builder,
            is_real_function=is_real_function,
        )

    _ki._get_tree_and_ctx = _cached_get_tree_and_ctx
    return True


def _apply_quadrants(version):
    """quadrants 1.3.x: the same two memos, against its own spellings.

    Quadrants renamed the context class and dissolved ``_get_tree_and_ctx``
    into ``FuncBase.get_tree_and_ctx``, which is too large and too entangled
    with its pruning passes to copy the way the taichi branch copies its
    counterpart. It does not have to be: the part worth caching is the
    ``get_source_info_and_src`` call at the top of it, which is a module-level
    function this can wrap without touching the transform itself.

    It also costs more here than it does on taichi, because quadrants builds
    every kernel AST **twice** -- a pruning pass and an enforcing pass -- so
    every source retrieval, and every position banner, is asked for twice
    before anything is compiled (`taichi_patches/PLAN.md` §6.1).
    """
    from algan.taichi_compat import submodule

    try:
        _atu = submodule("lang.ast.ast_transformer_utils")
        _wrap_inspect = submodule("lang._wrap_inspect")
        _func_base = submodule("lang._func_base")
    except Exception:
        return _skip("quadrants' transformer internals are not where 1.3 keeps them")

    if version[:2] != (1, 3):
        return _skip(
            f"the warm-start patch replicates quadrants 1.3 internals; this is "
            f"{'.'.join(str(p) for p in version) or 'an unknown version'}"
        )
    ctx_cls = getattr(_atu, "ASTTransformerFuncContext", None)
    if ctx_cls is None or not hasattr(ctx_cls, "get_pos_info"):
        return _skip("quadrants 1.3's transformer entry points have moved")
    if hasattr(ctx_cls, "_build_pos_info"):
        # Upstream #858 memoizes this itself (on `main`, in no release yet).
        # When a release carries it, ours is redundant rather than wrong --
        # stand down rather than stacking two caches on one function.
        return _skip("quadrants memoizes get_pos_info itself on this build")

    _install_pos_info_memo(ctx_cls, guard_src_bounds=True)

    # --- get_source_info_and_src memo -----------------------------------
    # `getsourcefile` + `getsourcelines` (each swapping `inspect.findsource`
    # around a linecache read) plus a frozen pydantic model built per call,
    # for a value that cannot change while the wrapper is alive. The returned
    # list is copied out because the caller owns what it is handed: it feeds a
    # list comprehension today, and a cache that hands out one shared mutable
    # list would be a landmine if that ever changes.
    #
    # `FunctionSourceInfo` is frozen and hashes by value, so handing back one
    # instance keeps `current_kernel.visited_functions` (a set) behaving
    # exactly as it does with fresh instances.
    orig_source_info = _wrap_inspect.get_source_info_and_src
    verify = env_flag("ALGAN_TAICHI_WARMSTART_VERIFY", False)

    def _cached_source_info_and_src(func):
        cached = getattr(func, "_algan_src_cache", None)
        if cached is None:
            cached = orig_source_info(func)
            with contextlib.suppress(AttributeError):
                func._algan_src_cache = cached
        if verify:
            ref_info, ref_src = orig_source_info(func)
            if ref_info != cached[0] or ref_src != cached[1]:
                raise RuntimeError(
                    f"taichi_warmstart source-info mismatch for {func!r}:\n"
                    f"cached: {cached[0]!r}\nref:    {ref_info!r}"
                )
        return cached[0], list(cached[1])

    # Rebound wherever it is *read*, not only where it is defined: both
    # consumers (`_func_base.get_tree_and_ctx`, `kernel.Kernel.materialize`)
    # did `from quadrants.lang._wrap_inspect import get_source_info_and_src`,
    # so each holds its own module global and patching the definition alone
    # would leave both call sites on the original. Modules imported *after*
    # this point pick the patched one up from `_wrap_inspect` by themselves.
    _wrap_inspect.get_source_info_and_src = _cached_source_info_and_src
    for module in list(sys.modules.values()):
        if (
            getattr(module, "__name__", "").startswith("quadrants.")
            and getattr(module, "get_source_info_and_src", None) is orig_source_info
        ):
            module.get_source_info_and_src = _cached_source_info_and_src

    # --- source-line fill memo ------------------------------------------
    # The other half of what the taichi branch's `_get_tree_and_ctx` copy
    # caches, reached differently: there the whole filled list is stored, here
    # the fill itself is memoized under `_func_base`'s `textwrap` name. Only
    # that module is rebound, and only if it is still the module we expect.
    if getattr(_func_base, "textwrap", None) is textwrap:
        _func_base.textwrap = _MemoizingTextwrap(verify=verify)
    return True
