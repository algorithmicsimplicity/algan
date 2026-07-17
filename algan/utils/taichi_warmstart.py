"""Warm-start accelerator for Taichi kernel materialization.

Even with a hot offline cache, every program run pays a large Python-side
"materialize" cost per kernel instantiation before Taichi can even *look up*
the cache: the offline-cache key is a hash of the frontend IR, and building
that IR (``transform_tree``) walks the whole kernel AST in Python. For this
project's monolithic wavefront kernels that is tens of seconds per run, and
profiling shows most of it is redundant:

* ``Builder.__call__`` eagerly computes ``ctx.get_pos_info(node)`` -- a
  caret-underlined, textwrap-wrapped source excerpt -- for every statement
  and expression node it visits, purely to attach debug info (~45% of the
  transform).
* Non-real ``ti.func``\\ s are inlined by re-running source retrieval
  (``getsourcelines`` + per-line ``textwrap.fill``) and a full re-transform
  of the func body at **every call site**, and the same kernel body is
  re-transformed per template instantiation.

Both are pure functions of the decorated function's source and the node
position, so :func:`apply` memoizes them from outside Taichi:

* ``ASTTransformerContext.get_pos_info`` results are cached per decorated
  function, keyed by (context line/indent offsets, node position, node class).
* The (file, source lines, start line) triple of ``_get_tree_and_ctx`` is
  cached per raw function object. The *AST itself is re-parsed fresh on every
  call*: Taichi's transformer annotates AST nodes in place (``node.ptr``), so
  a tree must never be shared between transforms.

Byte-identical by construction -- the memoized values are exactly the strings
the original code recomputes (validated against a live run: 0 mismatches over
136k calls, and hash-identical rendered output). Measured on a warm-cache
single-frame render, this halves the AST-transform phase.

The patch replicates taichi 1.7 internals verbatim, so it applies only on
taichi 1.7.x and degrades to a silent no-op anywhere else (or when
``ALGAN_TAICHI_WARMSTART=0``).
"""
import ast
import os
import sys
import textwrap

_APPLIED = False


def apply():
    """Install the memoizing patches (idempotent, safe no-op on mismatch)."""
    global _APPLIED
    if _APPLIED:
        return
    if os.environ.get("ALGAN_TAICHI_WARMSTART", "1") == "0":
        return

    try:
        import taichi
        import taichi.lang.ast.ast_transformer_utils as _atu
        import taichi.lang.kernel_impl as _ki
        from taichi.lang._wrap_inspect import getsourcefile, getsourcelines
    except Exception:
        return

    if tuple(getattr(taichi, "__version__", ()))[:2] != (1, 7):
        return
    ctx_cls = getattr(_atu, "ASTTransformerContext", None)
    if (ctx_cls is None or not hasattr(ctx_cls, "get_pos_info")
            or not hasattr(_ki, "_get_tree_and_ctx")
            or not hasattr(_ki, "_get_global_vars")):
        return

    # --- get_pos_info memo + fast first visit ---------------------------
    # The cache dict lives on the Kernel/Func wrapper (``ctx.func``), so the
    # entries share the wrapper's lifetime (dynamically composed pipeline
    # funcs are collected together with their caches). The wrapper pins its
    # source, so identical keys always map to identical strings.
    #
    # Cache *misses* are computed by a verbatim copy of taichi 1.7.4's
    # get_pos_info whose per-line ``TextWrapper(width=80).wrap`` is replaced
    # by a provably-equivalent fast path for the dominant case (a line of at
    # most 80 chars whose only whitespace is plain spaces): the wrapped
    # result is then [] for an all-space line and [line.rstrip(" ")]
    # otherwise (leading spaces kept on the first output line, the trailing
    # whitespace chunk dropped). Anything else -- long lines, tabs, unicode
    # whitespace -- falls back to the real TextWrapper.
    # ALGAN_TAICHI_WARMSTART_VERIFY=1 recomputes every result with the
    # original implementation and raises on any byte difference (used by
    # benchmarks/_taichi_warmstart_check.py).
    _orig_get_pos_info = ctx_cls.get_pos_info
    _verify = os.environ.get("ALGAN_TAICHI_WARMSTART_VERIFY", "0") == "1"

    def _wrap80(text):
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

    def _fast_get_pos_info(self, node):
        if sys.version_info < (3, 8):
            return _orig_get_pos_info(self, node)
        msg = (f'File "{self.file}", line {node.lineno + self.lineno_offset},'
               f" in {self.func.func.__name__}:\n")
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
                while last > 0 and (self.src[i][last - 1].isspace()
                                    or not self.src[i][last - 1].isprintable()):
                    last -= 1
                first = 0
                while first < len(self.src[i]) and (
                    self.src[i][first].isspace()
                    or not self.src[i][first].isprintable()
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

    def _memoized_get_pos_info(self, node):
        owner = self.func
        if owner is None:
            return _orig_get_pos_info(self, node)
        cache = getattr(owner, "_algan_pos_info_cache", None)
        if cache is None:
            cache = {}
            try:
                owner._algan_pos_info_cache = cache
            except AttributeError:
                return _orig_get_pos_info(self, node)
        key = (
            self.lineno_offset, self.indent,
            node.lineno, node.col_offset,
            getattr(node, "end_lineno", None),
            getattr(node, "end_col_offset", None),
            node.__class__.__name__,
        )
        hit = cache.get(key)
        if hit is None:
            hit = _fast_get_pos_info(self, node)
            cache[key] = hit
        if _verify:
            ref = _orig_get_pos_info(self, node)
            if ref != hit:
                raise RuntimeError(
                    "taichi_warmstart get_pos_info mismatch for "
                    f"{key}:\nfast: {hit!r}\nref:  {ref!r}")
        return hit

    ctx_cls.get_pos_info = _memoized_get_pos_info

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
            try:
                self.func._algan_src_cache = cached
            except AttributeError:
                pass
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

    _APPLIED = True
