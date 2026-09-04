"""Cached fast launcher for repeat kernel launches, on either compiler.

Every kernel launch pays a large Python-side cost before the C++ runtime is
reached: ``ensure_compiled`` re-extracts the template instantiation key from
every argument (the compiler's template mapper), and the launch method
re-discovers each argument's category through isinstance chains, re-validates
devices via string comparison, and re-checks gradient / layout /
foreign-framework cases that cannot occur mid-render. For this project's
render kernels (15-50 ndarray args, launched hundreds of times per render)
that redundancy dominates host-side launch cost. Measured with
``benchmarks/_quadrants_launch_overhead.py`` on a 20-ndarray kernel, one
process, warm: **taichi 1.7.4** 404 us per launch of which 363 us is Python
above ``prog.launch_kernel``; **quadrants 1.3.0** 294 us of which 261 us is
Python (mapper lookup ~35%, ``_recursive_set_args`` ~45%), against a 61 us
floor for a launch context built by hand. On a warm ``save_frame`` of a
``Square`` (30 launches) that Python is 4.2-4.6% of the frame's wall time on
both compilers -- and on quadrants it is not something the compiler's own
launch cache can take back: ``LaunchContextBufferCache`` marks a raw torch
tensor non-cacheable (``_func_base._recursive_set_args``), and its template
mapper cannot weak-reference a tuple, so for every Algan kernel that takes a
pipeline tuple its spec-key cache is disabled too and ``extract`` runs in full
on every launch. (See DESIGN_hybrid_raster.md item 9 -- this patch is the perf
half of that item, with no kernel-signature changes.)

:func:`apply` replaces ``Kernel.__call__`` with a memoizing dispatcher:

* The **first** launch of each (template values, per-ndarray dtype/ndim/
  grad) combination goes through the compiler's original ``__call__``
  unchanged -- full validation and materialization -- and then records a
  *launch plan*: the compiled kernel handle for that instantiation.
* Subsequent launches with the same key skip straight to the C++ calls the
  original path would have made: ``make_launch_context``, one
  ``set_arg_external_array_with_shape`` per ndarray carrying exactly the
  original's values, the scalars, then ``launch_kernel``.

The fast key is at least as fine as the compiler's own mapper key over the
supported argument universe: template values are restricted to the types the
mapper keys by value/identity (int/bool/float/str/None, functions, tuples of
those -- flat on taichi, flat or nested on quadrants, ``None`` included),
external arrays contribute (torch dtype, ndim, requires_grad) -- the mapper's
(element_type, ndim, needs_grad) features for scalar-dtype ndarrays -- and
scalar args are excluded from both keys. Vector/matrix-element ndarrays (every
BVH-taking kernel: the node arrays are ``ndarray(dtype=vector(4, f32))``)
additionally contribute the tensor's actual element shape, which both
reproduces the mapper's ``tensor_type(element_shape, dtype)`` feature and
keeps the fast path exactly as strict as the original validation (a mismatched
element shape misses the plan cache and takes -- and fails on -- the original
path); at set-arg the element dims are stripped from the runtime shape,
mirroring the compiler's own setter. Everything else falls back to the
original path per call (keyword calls -- which on quadrants is also how
``qd_stream`` and checkpoint resumes arrive -- gradient-carrying /
non-contiguous / foreign-device tensors, print-bearing kernels, active
autodiff tapes, ``qd.Tensor`` wrappers, any ndarray argument that is not a
torch tensor) or permanently per kernel (argpack / matrix / struct / texture
/ sparse / dataclass annotations, autodiff kernels, return values, graph and
checkpoint kernels).

**A non-torch ndarray argument always takes the original path.** That is what
keeps quadrants' ``LaunchContextBufferCache`` and its Metal byte-offset logic
in charge of ``Ndarray`` / ``ExternalMetalNdarray`` arguments, and it means
the fast path disengages on MPS: a torch MPS tensor is a foreign-device tensor
(the original path stages it through the host), and the zero-copy import
(:mod:`algan.rendering.mps_zero_copy`) hands the kernel an ndarray.

Byte-identical by construction -- the same compiled kernel receives the same
argument values; only redundant Python re-validation is skipped.
``ALGAN_TAICHI_FAST_LAUNCH_VERIFY=1`` re-derives the compiler's instantiation
on every fast hit and raises if it disagrees with the plan (used by
``benchmarks/_taichi_fast_launch_check.py``). ``ALGAN_TAICHI_FAST_LAUNCH=0``
turns it off. Each dispatcher replicates one compiler's launch path -- taichi
1.7.x, quadrants 1.3.x -- and stands down on any other version; a version this
file does not know is reported through :func:`skipped_reason` and printed by
``algan check``, for the reason :mod:`algan.utils.taichi_warmstart` gives: a
silent no-op reads exactly like a slow machine.
"""

from __future__ import annotations

from algan.environment import env_flag

_APPLIED = False

#: Why the dispatcher is not installed, or ``None`` when it is. Read by
#: ``algan check``.
_SKIPPED_REASON = None

# Engagement telemetry: fast-path launches vs. launches that took the
# original path (first launches, fallbacks). Read by the parity check so a
# silently disengaged fast path can never produce a vacuous pass.
STATS = {"fast": 0, "slow": 0}

# Runtime switch for in-process alternating A/B benchmarks (wall-clock
# comparisons across processes are thermally polluted on this hardware).
# The dispatcher stays installed; False routes every launch to the original
# path. Plans are retained.
ENABLED = True

#: Re-derive the compiler's instantiation on every hit and raise on a
#: disagreement. Seeded from ``ALGAN_TAICHI_FAST_LAUNCH_VERIFY`` when
#: :func:`apply` runs (a live variable: read there, not at import); a module
#: global rather than a bound constant so a test can turn it on after install.
VERIFY = False


def set_enabled(enabled):
    """Toggle the fast path at runtime (see ``ENABLED``)."""
    global ENABLED
    ENABLED = bool(enabled)


def skipped_reason():
    """``None`` if the dispatcher is live, else why it is not."""
    return _SKIPPED_REASON


def _skip(reason):
    global _SKIPPED_REASON
    _SKIPPED_REASON = reason
    return False


_TEMPLATE, _INT_S, _INT_U, _FLOAT, _EXT, _EXT_V = range(6)
_SCALAR_KEY_TYPES = (int, float, bool, str)


def apply():
    """Install the fast-launch dispatcher (idempotent, no-op on mismatch)."""
    global _APPLIED, _SKIPPED_REASON, VERIFY
    if _APPLIED:
        return
    if not env_flag("ALGAN_TAICHI_FAST_LAUNCH", True):
        _SKIPPED_REASON = "ALGAN_TAICHI_FAST_LAUNCH=0"
        return
    VERIFY = env_flag("ALGAN_TAICHI_FAST_LAUNCH_VERIFY", False)
    try:
        from algan.taichi_compat import BACKEND, backend_version
    except Exception:
        _SKIPPED_REASON = "the kernel compiler could not be imported"
        return
    installer = {"taichi": _apply_taichi, "quadrants": _apply_quadrants}.get(BACKEND)
    if installer is None:
        _SKIPPED_REASON = f"no fast-launch dispatcher is written for {BACKEND!r}"
        return
    _SKIPPED_REASON = None
    _APPLIED = bool(installer(tuple(backend_version())))


def _template_key_supported(v, nested):
    """Whether the mapper keys template value ``v`` by value or identity.

    Those are the values the fast key can carry verbatim: scalars and
    ``None`` by value, functions by identity, and tuples of those -- ``None``
    is a live element here, not a missing one: the injected pipeline /
    scatter tuples are indexed by material id, so a slot this batch does not
    use is a None the mapper keys by value like any other. Anything the
    mapper keys through a weakref or a pointer (SNodes, fields, lists,
    data-oriented objects) is unsupported and takes the original path.
    """
    tv = type(v)
    if tv in _SCALAR_KEY_TYPES or v is None:
        return True
    if tv is tuple:
        for item in v:
            if type(item) is tuple:
                if not nested or not _template_key_supported(item, nested):
                    return False
            elif not (
                type(item) in _SCALAR_KEY_TYPES
                or item is None
                or (callable(item) and not hasattr(item, "_data_oriented"))
            ):
                return False
        return True
    return callable(v) and not hasattr(v, "_data_oriented")


def _apply_taichi(version):
    """taichi 1.7.x: a hit replays the per-argument ``set_arg_*`` calls."""
    try:
        import numpy as np
        import torch

        from algan.taichi_compat import submodule

        _ki = submodule("lang.kernel_impl")
        _impl = submodule("lang.impl")
    except Exception:
        return _skip("taichi's launch internals are not where 1.7 keeps them")
    if version[:2] != (1, 7):
        return _skip(
            f"the fast-launch dispatcher replicates taichi 1.7 internals; this is "
            f"{'.'.join(str(p) for p in version) or 'an unknown version'}"
        )
    kernel_cls = getattr(_ki, "Kernel", None)
    if (
        kernel_cls is None
        or not hasattr(_ki, "template")
        or not hasattr(_ki, "ndarray_type")
        or not hasattr(_ki, "primitive_types")
        or not hasattr(_ki, "is_signed")
        or not hasattr(_ki, "cook_dtype")
        or not hasattr(_ki, "Layout")
    ):
        return _skip("taichi 1.7's launch internals have moved")

    _orig_call = kernel_cls.__call__
    _orig_reset = kernel_cls.reset
    _template_t = _ki.template
    _ndarray_t = _ki.ndarray_type.NdarrayType
    _real_ids = _ki.primitive_types.real_type_ids
    _int_ids = _ki.primitive_types.integer_type_ids
    _type_ids = _ki.primitive_types.type_ids
    _none_mode = _ki.AutodiffMode.NONE
    _soa = _ki.Layout.SOA
    _tensor_t = torch.Tensor
    # Same runtime type constraints the original setters enforce, applied
    # before plan lookup (scalar values are not part of the key, so a plan
    # must never accept a value the original path would have rejected).
    _int_ok = (int, np.integer)
    _float_ok = (int, float, np.integer, np.floating)
    # Arch is process-global between ti.init calls; Kernel.reset (hooked
    # below) clears it together with the plans.
    _state = {"arch_cuda": None}

    def _build_meta(self):
        """One (kind, launch slot, arg index) triple per argument, or None
        when the kernel uses a construct the fast path does not replicate.
        """
        if self.autodiff_mode != _none_mode or self.return_type is not None:
            return None
        meta = []
        slot = 0
        for i, karg in enumerate(self.arguments):
            anno = karg.annotation
            if isinstance(anno, _template_t):
                meta.append((_TEMPLATE, -1, i, None))
                continue
            if id(anno) in _real_ids:
                meta.append((_FLOAT, slot, i, None))
            elif id(anno) in _int_ids:
                kind = _INT_S if _ki.is_signed(_ki.cook_dtype(anno)) else _INT_U
                meta.append((kind, slot, i, None))
            elif isinstance(anno, _ndarray_t):
                if anno.needs_grad:
                    return None  # forced-gradient annotation
                if anno.dtype is None or id(anno.dtype) in _type_ids:
                    meta.append((_EXT, slot, i, None))
                else:
                    # Vector/matrix-element ndarray. ``ndim`` here is the
                    # element rank (1 = vector, 2 = matrix), used to strip the
                    # element dims from the runtime shape at set-arg and to
                    # pull the actual element shape into the fast key.
                    edim = getattr(anno.dtype, "ndim", None)
                    if not isinstance(edim, int) or edim <= 0:
                        return None
                    meta.append((_EXT_V, slot, i, (edim, anno.layout == _soa)))
            else:
                return None  # argpack / matrix / struct / texture / sparse
            slot += 1
        return meta

    def _record_plan(self, fast, args, key):
        """After a successful original launch, map the fast key to the
        instantiation Taichi itself selected for these arguments.
        """
        try:
            instance_id, _features = self.mapper.lookup(args)
            t_kernel = self.compiled_kernels.get(
                (self.func, instance_id, self.autodiff_mode)
            )
            if t_kernel is None:
                return
            fast["plans"][key] = t_kernel
        except Exception:
            fast["meta"] = None  # permanently disable for this kernel

    def _fast_call(self, *args, **kwargs):
        fast = self.__dict__.get("_algan_fast_plans")
        if fast is None:
            fast = {"meta": False, "plans": {}}
            self._algan_fast_plans = fast
        meta = fast["meta"]
        if meta is False:
            meta = _build_meta(self)
            fast["meta"] = meta
        rt = self.runtime
        if (
            not ENABLED
            or meta is None
            or kwargs
            or self.has_print
            or len(args) != len(self.arguments)
            or rt.target_tape is not None
            or rt.fwd_mode_manager is not None
        ):
            return _orig_call(self, *args, **kwargs)

        arch_cuda = _state["arch_cuda"]
        key_parts = []
        for kind, _slot, i, extra in meta:
            if kind in (_EXT, _EXT_V):
                v = args[i]
                # isinstance, not exact type: the engine passes plain data
                # subclasses (constants.color.Color) as kernel args, and the
                # original launch path accepts any torch.Tensor subclass via
                # the same isinstance check, reading the same buffer through
                # the same accessors. An exact-type test silently routed
                # every circuit_colors-taking kernel (first shade, shadow
                # trace, wavefront shade) to the slow path on every launch.
                if (
                    not isinstance(v, _tensor_t)
                    or v.requires_grad
                    or v.grad is not None
                    or not v.is_contiguous()
                ):
                    return _orig_call(self, *args, **kwargs)
                dev = v.device.type
                if dev == "cuda":
                    if arch_cuda is None:
                        # Kernel launches imply an initialized runtime; the
                        # arch is fixed until ti.init (reset clears this).
                        prog = _impl.get_runtime().prog
                        arch_cuda = prog.config().arch == _ki._ti_core.Arch.cuda
                        _state["arch_cuda"] = arch_cuda
                    if not arch_cuda:
                        return _orig_call(self, *args, **kwargs)
                elif dev != "cpu":
                    return _orig_call(self, *args, **kwargs)
                key_parts.append(v.dtype)
                key_parts.append(v.dim())
                if kind == _EXT_V:
                    # The actual element shape: reproduces the mapper's
                    # tensor_type(element_shape, dtype) feature, and ensures a
                    # shape the original validation would reject can never hit
                    # a recorded plan (it misses and takes the original path).
                    edim = extra[0]
                    key_parts.append(
                        tuple(v.shape[:edim]) if extra[1] else tuple(v.shape[-edim:])
                    )
            elif kind == _TEMPLATE:
                v = args[i]
                if not _template_key_supported(v, nested=False):
                    return _orig_call(self, *args, **kwargs)
                key_parts.append(v)
            elif kind == _FLOAT:
                if not isinstance(args[i], _float_ok):
                    return _orig_call(self, *args, **kwargs)
            else:  # _INT_S / _INT_U
                if not isinstance(args[i], _int_ok):
                    return _orig_call(self, *args, **kwargs)

        key = tuple(key_parts)
        t_kernel = fast["plans"].get(key)
        if t_kernel is None:
            STATS["slow"] += 1
            ret = _orig_call(self, *args, **kwargs)
            _record_plan(self, fast, args, key)
            return ret
        STATS["fast"] += 1

        if VERIFY:
            instance_id, _features = self.mapper.lookup(args)
            ref = self.compiled_kernels.get(
                (self.func, instance_id, self.autodiff_mode)
            )
            if ref is not t_kernel:
                raise RuntimeError(
                    "taichi_fast_launch: instantiation mismatch for kernel "
                    f"{self.func.__name__}"
                )

        launch_ctx = t_kernel.make_launch_context()
        for kind, slot, i, extra in meta:
            if kind == _EXT:
                v = args[i]
                launch_ctx.set_arg_external_array_with_shape(
                    (slot,), v.data_ptr(), v.element_size() * v.nelement(), v.shape, 0
                )
            elif kind == _EXT_V:
                # Strip the element dims from the shape, exactly as
                # kernel_impl.set_arg_ext_array does for vector/matrix-element
                # ndarrays ("element shapes are already specialized in
                # codegen"); byte count still covers the whole buffer.
                v = args[i]
                edim = extra[0]
                launch_ctx.set_arg_external_array_with_shape(
                    (slot,),
                    v.data_ptr(),
                    v.element_size() * v.nelement(),
                    v.shape[edim:] if extra[1] else v.shape[:-edim],
                    0,
                )
            elif kind == _INT_S:
                launch_ctx.set_arg_int((slot,), int(args[i]))
            elif kind == _FLOAT:
                launch_ctx.set_arg_float((slot,), float(args[i]))
            elif kind == _INT_U:
                launch_ctx.set_arg_uint((slot,), int(args[i]))
        prog = _impl.get_runtime().prog
        compiled = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel)
        prog.launch_kernel(compiled, launch_ctx)
        return None

    def _reset(self):
        self.__dict__.pop("_algan_fast_plans", None)
        _state["arch_cuda"] = None
        return _orig_reset(self)

    _fast_call._algan_original = _orig_call
    kernel_cls.__call__ = _fast_call
    kernel_cls.reset = _reset
    return True


def _apply_quadrants(version):
    """quadrants 1.3.x: a hit replays the batched ``set_args_*`` calls.

    The launch path is ``Kernel.__call__`` -> ``ensure_compiled`` (mapper
    lookup, materialize) -> ``launch_kernel`` (``_recursive_set_args`` per
    argument into a buffer, the batched ``set_args_{float,int,uint}`` pybinds,
    ``prog.launch_kernel``). A hit does only the last three, against the
    ``CompiledKernelData`` the original path already compiled for this key --
    there is no ``compile_kernel`` on a hit, unlike taichi.

    Argument indices are plain ints here (taichi's are one-tuples), the
    scalars go through the batched setters the original uses, and every
    internal is taken off ``lang.kernel``'s own namespace, so a release that
    moves one fails the guard below rather than the first launch.
    """
    try:
        import numpy as np
        import torch

        from algan.taichi_compat import submodule

        _kernel_mod = submodule("lang.kernel")
        _ndarray_type = submodule("types.ndarray_type")
        _enums = submodule("types.enums")
    except Exception:
        return _skip("quadrants' launch internals are not where 1.3 keeps them")
    if version[:2] != (1, 3):
        return _skip(
            f"the fast-launch dispatcher replicates quadrants 1.3 internals; this is "
            f"{'.'.join(str(p) for p in version) or 'an unknown version'}"
        )
    kernel_cls = getattr(_kernel_mod, "Kernel", None)
    needed = (
        "template",
        "primitive_types",
        "is_signed",
        "cook_dtype",
        "handle_exception_from_cpp",
        "impl",
        "_tensor_wrapper",
        "_TENSOR_WRAPPER_TYPES",
        "AutodiffMode",
        "Arch",
        "KernelLaunchContext",
    )
    if (
        kernel_cls is None
        or any(not hasattr(_kernel_mod, name) for name in needed)
        or not hasattr(_ndarray_type, "NdarrayType")
        or not hasattr(_enums, "Layout")
        or not hasattr(_kernel_mod.KernelLaunchContext, "set_args_int")
        or not hasattr(
            _kernel_mod.KernelLaunchContext, "set_arg_external_array_with_shape"
        )
    ):
        return _skip("quadrants 1.3's launch internals have moved")

    _orig_call = kernel_cls.__call__
    _orig_reset = kernel_cls.reset
    _impl = _kernel_mod.impl
    _template_t = _kernel_mod.template
    _ndarray_t = _ndarray_type.NdarrayType
    _real_ids = _kernel_mod.primitive_types.real_type_ids
    _int_ids = _kernel_mod.primitive_types.integer_type_ids
    _type_ids = _kernel_mod.primitive_types.type_ids
    _none_mode = _kernel_mod.AutodiffMode.NONE
    _aos = _enums.Layout.AOS
    _arch_cuda = _kernel_mod.Arch.cuda
    _arch_python = _kernel_mod.Arch.python
    _tensor_wrapper = _kernel_mod._tensor_wrapper
    _wrapper_types = _kernel_mod._TENSOR_WRAPPER_TYPES
    _handle_exception = _kernel_mod.handle_exception_from_cpp
    _tensor_t = torch.Tensor
    _int_ok = (int, np.integer)
    _float_ok = (int, float, np.integer, np.floating)
    _state = {"arch_cuda": None}

    def _build_meta(self):
        if self.autodiff_mode != _none_mode or self.return_type is not None:
            return None
        meta = []
        slot = 0
        for i, karg in enumerate(self.arg_metas):
            anno = karg.annotation
            if anno is _template_t or isinstance(anno, _template_t):
                meta.append((_TEMPLATE, -1, i, None))
                continue
            if id(anno) in _real_ids:
                meta.append((_FLOAT, slot, i, None))
            elif id(anno) in _int_ids:
                kind = (
                    _INT_S
                    if _kernel_mod.is_signed(_kernel_mod.cook_dtype(anno))
                    else _INT_U
                )
                meta.append((kind, slot, i, None))
            elif type(anno) is _ndarray_t:
                if anno.needs_grad:
                    return None  # forced-gradient annotation
                if anno.layout != _aos:
                    # The mapper keys the element shape off the *trailing*
                    # dims whatever the layout, while the setter strips the
                    # leading ones for SOA; 1.3 hard-codes AOS, and this is
                    # not the place to pick a side if that changes.
                    return None
                if anno.dtype is None or id(anno.dtype) in _type_ids:
                    meta.append((_EXT, slot, i, None))
                else:
                    edim = getattr(anno.dtype, "ndim", None)
                    if not isinstance(edim, int) or edim <= 0:
                        return None
                    meta.append((_EXT_V, slot, i, edim))
            else:
                # qd.Tensor / dataclass / matrix / struct / sparse / buffer view
                return None
            slot += 1
        return meta

    def _record_plan(self, fast, args, key):
        """Map the fast key to the (KernelCxx, CompiledKernelData) the
        original launch just used for these arguments.
        """
        try:
            instance_id, _features = self.mapper.lookup(
                self.raise_on_templated_floats, args
            )
            ckey = (self.func, instance_id, self.autodiff_mode)
            t_kernel = self.materialized_kernels.get(ckey)
            compiled = self.compiled_kernel_data_by_key.get(ckey)
            if t_kernel is None or compiled is None or self.graph_do_while_levels:
                return
            fast["plans"][key] = (t_kernel, compiled)
        except Exception:
            fast["meta"] = None  # permanently disable for this kernel

    def _fast_call(self, *args, **kwargs):
        fast = self.__dict__.get("_algan_fast_plans")
        if fast is None:
            fast = {"meta": False, "plans": {}}
            self._algan_fast_plans = fast
        meta = fast["meta"]
        if meta is False:
            meta = _build_meta(self)
            fast["meta"] = meta
        rt = self.runtime
        if (
            not ENABLED
            or meta is None
            or kwargs  # also how qd_stream and _qd_from_checkpoint arrive
            or self.has_print
            or self.use_checkpoints
            or self.use_graph
            or len(args) != len(self.arg_metas)
            or rt.target_tape
            or rt.fwd_mode_manager
        ):
            return _orig_call(self, *args, **kwargs)
        runtime = _impl.get_runtime()
        if runtime._arch == _arch_python:
            return _orig_call(self, *args, **kwargs)
        if _tensor_wrapper._any_tensor_constructed:
            for v in args:
                if type(v) in _wrapper_types:
                    return _orig_call(self, *args, **kwargs)

        arch_cuda = _state["arch_cuda"]
        key_parts = []
        for kind, _slot, i, extra in meta:
            if kind in (_EXT, _EXT_V):
                v = args[i]
                # isinstance, not exact type -- see the taichi branch: the
                # engine passes torch.Tensor subclasses (Color) as arguments.
                # Anything else (an Ndarray, a numpy array) is the original
                # path's to marshal.
                if (
                    not isinstance(v, _tensor_t)
                    or v.requires_grad
                    or v.grad is not None
                    or not v.is_contiguous()
                ):
                    return _orig_call(self, *args, **kwargs)
                dev = v.device.type
                if dev == "cuda":
                    if arch_cuda is None:
                        arch_cuda = runtime.prog.config().arch == _arch_cuda
                        _state["arch_cuda"] = arch_cuda
                    if not arch_cuda:
                        return _orig_call(self, *args, **kwargs)
                elif dev != "cpu":
                    return _orig_call(self, *args, **kwargs)
                key_parts.append(v.dtype)
                key_parts.append(v.dim())
                if kind == _EXT_V:
                    # The mapper's tensor_type(element_shape, dtype) feature
                    # is keyed off the trailing element dims (AOS, guarded
                    # above); a shape the original validation would reject
                    # misses the plan cache and fails on the original path.
                    key_parts.append(tuple(v.shape[-extra:]))
            elif kind == _TEMPLATE:
                v = args[i]
                if not _template_key_supported(v, nested=True):
                    return _orig_call(self, *args, **kwargs)
                key_parts.append(v)
            elif kind == _FLOAT:
                if not isinstance(args[i], _float_ok):
                    return _orig_call(self, *args, **kwargs)
            else:  # _INT_S / _INT_U
                if not isinstance(args[i], _int_ok):
                    return _orig_call(self, *args, **kwargs)

        key = tuple(key_parts)
        plan = fast["plans"].get(key)
        if plan is None:
            STATS["slow"] += 1
            ret = _orig_call(self, *args, **kwargs)
            _record_plan(self, fast, args, key)
            return ret
        STATS["fast"] += 1
        t_kernel, compiled = plan

        if VERIFY:
            ckey = self.ensure_compiled(*args)
            if (
                self.materialized_kernels.get(ckey) is not t_kernel
                or self.compiled_kernel_data_by_key.get(ckey) is not compiled
            ):
                raise RuntimeError(
                    "taichi_fast_launch: instantiation mismatch for kernel "
                    f"{self.func.__name__}"
                )

        launch_ctx = t_kernel.make_launch_context()
        int_slots = []
        int_vals = []
        uint_slots = []
        uint_vals = []
        float_slots = []
        float_vals = []
        for kind, slot, i, extra in meta:
            if kind == _EXT:
                v = args[i]
                launch_ctx.set_arg_external_array_with_shape(
                    slot, v.data_ptr(), v.element_size() * v.nelement(), v.shape, 0
                )
            elif kind == _EXT_V:
                # Element dims stripped from the shape, as
                # _recursive_set_args does ("element shapes are already
                # specialized in codegen"); byte count covers the buffer.
                v = args[i]
                launch_ctx.set_arg_external_array_with_shape(
                    slot,
                    v.data_ptr(),
                    v.element_size() * v.nelement(),
                    v.shape[:-extra],
                    0,
                )
            elif kind == _INT_S:
                int_slots.append(slot)
                int_vals.append(int(args[i]))
            elif kind == _FLOAT:
                float_slots.append(slot)
                float_vals.append(float(args[i]))
            elif kind == _INT_U:
                uint_slots.append(slot)
                uint_vals.append(int(args[i]))
        if float_slots:
            launch_ctx.set_args_float(float_slots, float_vals)
        if int_slots:
            launch_ctx.set_args_int(int_slots, int_vals)
        if uint_slots:
            launch_ctx.set_args_uint(uint_slots, uint_vals)
        # use_graph defaults to False on a fresh context, which is what the
        # original sets for the non-graph kernels that reach this point.
        try:
            runtime.prog.launch_kernel(compiled, launch_ctx)
        except Exception as e:
            # The original's translation of C++ exceptions, verbatim.
            e = _handle_exception(e)
            if runtime.print_full_traceback:
                raise e
            raise e from None
        return None

    def _reset(self):
        self.__dict__.pop("_algan_fast_plans", None)
        _state["arch_cuda"] = None
        return _orig_reset(self)

    _fast_call._algan_original = _orig_call
    kernel_cls.__call__ = _fast_call
    kernel_cls.reset = _reset
    return True
