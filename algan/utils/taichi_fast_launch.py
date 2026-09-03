"""Cached fast launcher for repeat Taichi kernel launches.

Every Taichi kernel launch pays a large Python-side cost before the C++
runtime is reached: ``ensure_compiled`` re-extracts the template
instantiation key from every argument
(``TaichiCallableTemplateMapper.extract``), and ``launch_kernel``
re-discovers each argument's category through isinstance chains,
re-validates devices via string comparison, and re-checks gradient / layout
/ foreign-framework cases that cannot occur mid-render.  For this project's
render kernels (15-50 ndarray args, launched hundreds of times per render,
~0.15-0.3 ms of Python per ndarray argument) that redundancy dominates
host-side launch cost: ~7-9 ms per launch measured here, several seconds of
a small render (see DESIGN_hybrid_raster.md item 9 -- this patch is the
perf half of that item, with no kernel-signature changes).

:func:`apply` replaces ``Kernel.__call__`` with a memoizing dispatcher:

* The **first** launch of each (template values, per-ndarray dtype/ndim/
  grad) combination goes through Taichi's original ``__call__`` unchanged
  -- full validation and materialization -- and then records a *launch
  plan*: the compiled kernel handle for that instantiation.
* Subsequent launches with the same key skip straight to the C++ calls the
  original path would have made: ``make_launch_context``, one
  ``set_arg_{int,uint,float}`` / ``set_arg_external_array_with_shape`` per
  argument carrying exactly the original's values, then ``compile_kernel``
  (C++-side cached) and ``launch_kernel``.

The fast key is at least as fine as Taichi's own mapper key over the
supported argument universe: template values are restricted to the types
the mapper keys by value/identity (int/bool/float/str/None, functions, flat
tuples of those, None included), external arrays contribute (torch dtype, ndim,
requires_grad) -- the mapper's (element_type, ndim, needs_grad) features
for scalar-dtype ndarrays -- and scalar args are excluded from both keys.
Vector/matrix-element ndarrays (every BVH-taking kernel: the node arrays
are ``ndarray(dtype=vector(4, f32))``) additionally contribute the
tensor's actual element shape, which both reproduces the mapper's
``tensor_type(element_shape, dtype)`` feature and keeps the fast path
exactly as strict as the original validation (a mismatched element shape
misses the plan cache and takes -- and fails on -- the original path); at
set-arg the element dims are stripped from the runtime shape, mirroring
``kernel_impl.set_arg_ext_array``. Everything else falls back to the
original path per call (keyword calls, gradient-carrying / non-contiguous
/ foreign-device tensors, print-bearing kernels, active autodiff tapes) or
permanently per kernel (argpack / matrix / struct / texture / sparse
annotations, autodiff kernels, return values).

Byte-identical by construction -- the same compiled kernel receives the
same argument values; only redundant Python re-validation is skipped.
``ALGAN_TAICHI_FAST_LAUNCH_VERIFY=1`` re-derives Taichi's instantiation on
every fast hit and raises if it disagrees with the plan (used by
``benchmarks/_taichi_fast_launch_check.py``).  Applies on taichi 1.7.x
only; silent no-op anywhere else (or when ``ALGAN_TAICHI_FAST_LAUNCH=0``).
"""

from __future__ import annotations

from algan.environment import env_flag

_APPLIED = False

# Engagement telemetry: fast-path launches vs. launches that took the
# original path (first launches, fallbacks). Read by the parity check so a
# silently disengaged fast path can never produce a vacuous pass.
STATS = {"fast": 0, "slow": 0}

# Runtime switch for in-process alternating A/B benchmarks (wall-clock
# comparisons across processes are thermally polluted on this hardware).
# The dispatcher stays installed; False routes every launch to the original
# path. Plans are retained.
ENABLED = True


def set_enabled(enabled):
    """Toggle the fast path at runtime (see ``ENABLED``)."""
    global ENABLED
    ENABLED = bool(enabled)


_TEMPLATE, _INT_S, _INT_U, _FLOAT, _EXT, _EXT_V = range(6)


def apply():
    """Install the fast-launch dispatcher (idempotent, no-op on mismatch)."""
    global _APPLIED
    if _APPLIED:
        return
    if not env_flag("ALGAN_TAICHI_FAST_LAUNCH", True):
        return
    try:
        import numpy as np
        import torch

        from algan.taichi_compat import submodule
        from algan.taichi_compat import ti as taichi

        _ki = submodule("lang.kernel_impl")
        _impl = submodule("lang.impl")
    except Exception:
        return
    if tuple(getattr(taichi, "__version__", ()))[:2] != (1, 7):
        return
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
        return

    _orig_call = kernel_cls.__call__
    _orig_reset = kernel_cls.reset
    _verify = env_flag("ALGAN_TAICHI_FAST_LAUNCH_VERIFY", False)
    _template_t = _ki.template
    _ndarray_t = _ki.ndarray_type.NdarrayType
    _real_ids = _ki.primitive_types.real_type_ids
    _int_ids = _ki.primitive_types.integer_type_ids
    _type_ids = _ki.primitive_types.type_ids
    _none_mode = _ki.AutodiffMode.NONE
    _soa = _ki.Layout.SOA
    _tensor_t = torch.Tensor
    _scalar_key_types = (int, float, bool, str)
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
                tv = type(v)
                if tv in _scalar_key_types or v is None:
                    key_parts.append(v)
                elif tv is tuple:
                    for item in v:
                        # ``None`` is a live element here, not a missing one:
                        # the injected pipeline / scatter tuples are indexed by
                        # material id, so a slot this batch does not use is a
                        # None the mapper keys by value like any other.
                        if not (
                            type(item) in _scalar_key_types
                            or item is None
                            or (callable(item) and not hasattr(item, "_data_oriented"))
                        ):
                            return _orig_call(self, *args, **kwargs)
                    key_parts.append(v)
                elif callable(v) and not hasattr(v, "_data_oriented"):
                    key_parts.append(v)
                else:
                    return _orig_call(self, *args, **kwargs)
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

        if _verify:
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

    kernel_cls.__call__ = _fast_call
    kernel_cls.reset = _reset
    _APPLIED = True
