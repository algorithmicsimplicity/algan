"""Renderer settings: what a render produces, and the experimental switches.

``SETTINGS.raytracing`` is deliberately small. It holds the settings that change
what the image *looks like* -- ``samples_per_pixel``, ``max_bounces``,
``shadows``, lighting and tonemapping -- and those are the ones documented and
supported.

The renderer also carries roughly fifty kernel and performance switches. Those
live on ``SETTINGS.raytracing.experimental``, and setting one on the parent
raises with a pointer to the experimental section rather than silently accepting
it. The split is about the promise made, not the mechanism: engine code still
reads every field off ``SETTINGS.raytracing`` directly, and only writes are gated.

**Nothing here is listed twice.** The field set, each field's type, and each
field's setter are all *derived* from the storage module
(``algan/rendering/raytracing/settings.py``), because every table that mirrored
it drifted from it. There used to be three: a 119-row map from each lowercase
field to an UPPER_CASE global of the same name, a 50-row map from a field to
its setter, and a type table. The first is what let nine switches reach the
engine with no way to set them -- each had a global and a setter, and nobody
added the row -- and a tenth would have gone the same way. There is one
spelling for each setting now, so a switch declared in that module IS a field.

What remains listed is only what cannot be derived: :data:`_PUBLIC_FIELDS` (a
promise, not a fact about the code), :data:`_POLYMORPHIC_FIELDS` (three mode
switches that spell one state as a bool and another as a string, where
inferring the type from the default is wrong), :data:`_MINIMUMS` (bounds read
off each field's documented meaning) and :data:`_INERT_FIELDS`.

Every write is validated: the accepted type comes from the value the field
ships with, numeric fields carry their lower bound, and floats must be finite.
Before that, only fields with a setter were checked at all, and only as far as
that setter's own ``bool()``/``float()`` went -- ``max_bounces = 'x'`` stored
the string and failed much later inside a kernel with nothing pointing back
here.

:class:`RayTracingPreset` captures a configuration for reuse. Like the video
presets it is immutable, so ``set()`` on one returns a copy.

Read these live (``rt_settings.x`` at call time) rather than importing them by
value at module import, which would freeze them before user code runs.
"""

from __future__ import annotations

import difflib
import math
from contextlib import contextmanager
from copy import deepcopy

from algan.errors import AlganConfigurationError, AlganError

#: The live field values, and the type each one ships with. Populated the first
#: time the storage module is resolved -- before any ``set`` can have run, so
#: these are the shipped defaults rather than whatever the configuration
#: currently holds. That matters for ``shadow_terminator``, whose default is a
#: bool but which stores ``2`` once someone selects ``"relax"``.
_DEFAULT_TYPES: dict[str, type] = {}

#: Resolved storage module, cached. It used to be re-imported on every read,
#: which made ``SETTINGS.raytracing.x`` 34x slower than the module global it
#: was reading (868 ns against 25 ns measured) -- and that gap is the whole
#: reason engine code reaches past this object for the module instead.
_MODULE = None


def _module():
    """The module the fields live in, resolved lazily and cached.

    Lazily because this package is imported while ``algan.settings`` is being
    assembled, and the storage module imports back into the renderer; by the
    time anything asks for a field, both are long since built.
    """
    global _MODULE
    if _MODULE is None:
        import importlib

        _MODULE = importlib.import_module("algan.rendering.raytracing.settings")
        _DEFAULT_TYPES.update(
            (name, type(value))
            for name, value in vars(_MODULE).items()
            if _is_field(name, value)
        )
    return _MODULE


def _is_field(name: str, value) -> bool:
    """Whether a module-level name is one of the renderer's settings.

    Derived rather than listed. The predecessor of this rule was a
    hand-maintained 119-row table mapping each lowercase field to an
    UPPER_CASE global of the same name, and the two spellings are exactly what
    let nine switches reach the engine with no way to set them: they had a
    global and a setter, and nobody added the row. There is one spelling now,
    so a switch declared in that module IS a field and cannot be forgotten.

    A field is public, lowercase, and holds a scalar. Everything else there is
    a helper (callable), an internal (``_``-prefixed), a genuine constant that
    is not configuration (``ANALYTIC_AA_SLIVER_MODES``, ``WF_TEX_BEZ`` -- still
    UPPER_CASE, as constants are), or an import.
    """
    return (
        not name.startswith("_")
        and name == name.lower()
        and isinstance(value, (bool, int, float, str))
    )


def _field_names() -> frozenset[str]:
    _module()
    return frozenset(_DEFAULT_TYPES)


def _shadowed_fields(module) -> list[str]:
    """Fields whose name a later ``def`` in the storage module took over.

    One spelling means a field and a helper cannot share a name, and Python
    will not say so: the later binding simply wins, the field stops existing,
    and the only symptom is that it silently drops out of ``SETTINGS``. Three
    did exactly that when the two spellings were merged -- two trivial
    pass-through accessors (``merge_gpu_peak_factor``,
    ``project_gpu_peak_factor``, which became ``return`` of themselves) and one
    real one (now ``effective_analytic_aa_secondary_samples``).

    Checked against the declarations rather than the live namespace, because
    the live namespace is precisely what the shadowing has already destroyed.
    """
    import ast
    import inspect

    declared = []
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                declared.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared.append(node.target.id)
    names = {n for n in declared if not n.startswith("_") and n == n.lower()}
    return sorted(n for n in names if callable(getattr(module, n, None)))


def _setter(module, field):
    """The field's own setter, when it has one, else ``None``.

    Derived rather than listed, for the same reason as :func:`_is_field`: the
    hand-maintained table this replaces had to name 50 of them, and a setter
    added without its row was silently bypassed.
    """
    return getattr(module, f"set_{field}", None)


# The settings that describe what the renderer *produces*. Every other field is
# a performance or capability switch whose meaning is tied to the current kernel
# implementation; those live under ``.experimental`` so that the supported
# surface is obvious from tab-completion and repr.
_PUBLIC_FIELDS = frozenset(
    {
        "samples_per_pixel",
        "max_bounces",
        "shadows",
        "ambient_light",
        "light_intensity",
        "indirect_bounce_strength",
        "glossy_reflection",
        "glossy_prefilter",
        "analytic_aa",
        "linear_color_space",
        "tonemapping",
        "tonemap_method",
        "tonemap_exposure",
        "unsupported_feature_policy",
    }
)


def _experimental_fields() -> frozenset[str]:
    return _field_names() - _PUBLIC_FIELDS


# Settings no renderer this package can actually launch reads. Both are
# consumed only by ``raytrace_kernels_taichi.path_trace_physical_stbvh``, the
# never-wired "physical mode" Monte Carlo kernel: ``tracer`` launches
# ``path_trace_scene_stbvh`` for samples_per_pixel > 1 and the wavefront tracer
# otherwise, and the only other reference to the physical kernel is a unit
# test. Setting either therefore did nothing at all, silently, which is worse
# than not offering them -- so writing one says so. Reads keep working: engine
# code binds this object and reads fields off it on the hot path, and the
# values are still what the dead kernel would use.
_INERT_FIELDS = {
    "light_intensity": (
        "'light_intensity' is not read by any renderer this build can launch "
        "(only by the unwired physical-mode Monte Carlo kernel), so setting it "
        "would silently do nothing. Scale a light with its own intensity= "
        "instead: PointLight(intensity=2.0), DirectionalLight(intensity=2.0)."
    ),
    "ambient_light": (
        "'ambient_light' is not read by any renderer this build can launch "
        "(only by the unwired physical-mode Monte Carlo kernel), so setting it "
        "would silently do nothing. Add an AmbientLight to the Scene instead: "
        "AmbientLight(color=WHITE, intensity=0.3).spawn()."
    ),
}


#: Fields whose setter deliberately accepts a type other than the one the field
#: ships with, so the derived type check has to stand aside for them. Each is a
#: mode switch that spells one of its states as a bool and another as a string.
_POLYMORPHIC_FIELDS = frozenset(
    {
        "shadow_terminator",  # bool, plus "relax" for the third state
        "shadow_anyhit",  # bool, plus "gather" for the KBUF gather-march
        "wavefront_sort_materials",  # str, plus True meaning "auto"
        "wf_gen_fused",  # str "auto", plus True/False forcing the mode
    }
)

#: ``field -> (bound, exclusive)`` lower bounds, each read off the field's own
#: documented meaning in ``algan/rendering/raytracing/settings.py`` rather than
#: guessed: a count of rays or levels cannot be below 1, a strength or a
#: tolerance-as-a-fraction cannot be negative, and a multiplier the memory model
#: divides its estimates by cannot be zero. No upper bounds -- nothing in those
#: comments says where the top is, and inventing one would reject a legitimate
#: value later.
_MINIMUMS = {
    # counts: 1 is the smallest meaningful one
    "samples_per_pixel": (1, False),
    "analytic_aa_secondary_samples": (1, False),
    "glossy_prefilter_max_levels": (1, False),
    "refract_initial_pool_ratio": (1, False),
    "wavefront_tile_rays": (1, False),
    "wavefront_tile_min": (1, False),
    "wavefront_tile_max": (1, False),
    # strengths, tolerances and fractions: 0 is a documented, meaningful value
    "max_bounces": (0, False),
    "ambient_light": (0, False),
    "ambient_strength": (0, False),
    "ambient_strength_linear": (0, False),
    "indirect_bounce_strength": (0, False),
    "light_intensity": (0, False),
    "tonemap_exposure": (0, False),
    "shadow_eps_relative": (0, False),
    "shadow_near_fraction": (0, False),
    "analytic_aa_bez_min_half_width": (0, False),
    "analytic_aa_secondary_min_energy": (0, False),
    "wf_gen_fused_gain": (0, False),
    "wf_gen_fused_min_win": (0, False),
    "wf_textured_features": (0, False),
    # multipliers the memory model scales an estimate by, and a flattening
    # tolerance: zero is degenerate, not merely small -- it under-estimates a
    # transient peak to nothing, or asks for infinite subdivision
    "merge_gpu_peak_factor": (0, True),
    "project_gpu_peak_factor": (0, True),
    "sparse_discovery_safety": (0, True),
    "wavefront_tile_safety": (0, True),
    "analytic_aa_chord_tolerance": (0, True),
}

#: String fields whose accepted values are enumerated in the storage module.
#: Named by the tuple that holds them so the two cannot drift apart. The other
#: four string fields validate inside their own setter.
_CHOICES = {
    "analytic_aa_sliver": "ANALYTIC_AA_SLIVER_MODES",
    "analytic_aa_run_rule": "ANALYTIC_AA_RUN_RULES",
}


def _check_value(field, value, module):
    """Validate one field's value against what it ships with, and normalize it.

    ``module`` is the resolved storage module. Taking it as an argument is not
    just convenience: resolving it is what populates :data:`_DEFAULT_TYPES`, so
    a caller that validates before resolving it would find that dict empty and
    check nothing at all.

    Fields that had a setter used to be the only ones checked
    at all, and only as far as their setter's own ``bool()``/``float()``
    coercion went; every other field was written straight through, so
    ``max_bounces = 'x'`` stored the string and failed much later inside a
    kernel with nothing pointing back here.

    The expected type is derived from the value the field ships with rather
    than declared in a table, because a 106-row table beside 106 defaults is a
    second source of truth that drifts. :data:`_POLYMORPHIC_FIELDS` is the
    exemption list for the three fields where that inference is wrong.

    ``bool`` is checked before ``int``: it is a subclass, so ``max_bounces =
    True`` would otherwise pass as the integer 1.
    """
    if field in _POLYMORPHIC_FIELDS:
        return value
    expected = _DEFAULT_TYPES.get(field)
    if expected is bool:
        if not isinstance(value, bool):
            raise AlganConfigurationError(
                f"'{field}' must be True or False, got {value!r}"
            )
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AlganConfigurationError(
                f"'{field}' must be an integer, got {value!r}"
            )
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AlganConfigurationError(f"'{field}' must be a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            # A NaN tolerance propagates into a kernel and comes out as missing
            # geometry, with nothing naming the setting that caused it.
            raise AlganConfigurationError(
                f"'{field}' must be a finite number, got {value!r}"
            )
    elif expected is str:
        if not isinstance(value, str):
            raise AlganConfigurationError(f"'{field}' must be a string, got {value!r}")
        choices = _CHOICES.get(field)
        if choices is not None:
            allowed = getattr(module, choices)
            if value not in allowed:
                raise AlganConfigurationError(
                    f"'{field}' must be one of {', '.join(map(repr, allowed))}, "
                    f"got {value!r}"
                )
        return value
    else:
        return value

    bound = _MINIMUMS.get(field)
    if bound is not None:
        minimum, exclusive = bound
        if value <= minimum if exclusive else value < minimum:
            comparison = "greater than" if exclusive else "at least"
            raise AlganConfigurationError(
                f"'{field}' must be {comparison} {minimum}, got {value!r}"
            )
    return value


def _unknown(name: str):
    suggestion = difflib.get_close_matches(name, sorted(_field_names()), n=1)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise AlganConfigurationError(f"Unknown RayTracingSettings setting '{name}'.{hint}")


class RayTracingPreset:
    """Immutable captured ray-tracing configuration."""

    def __init__(self, values: dict[str, object]):
        object.__setattr__(self, "_values", deepcopy(values))

    @property
    def is_preset(self):
        return True

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        if name.startswith("set_") and name[4:] in self._values:
            return lambda value: self.set(**{name[4:]: value})
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AlganConfigurationError(
            "RayTracingPreset is immutable; use preset.set(...) to create a copy"
        )

    def set(self, source=None, **kwargs):
        values = {}
        if source is not None:
            if not isinstance(source, (RayTracingPreset, RayTracingSettings)):
                raise AlganConfigurationError(
                    "RayTracingPreset.set expected RayTracingPreset or "
                    f"RayTracingSettings, got {type(source).__name__}"
                )
            values.update(source.to_dict())
        values.update(kwargs)

        normalized = {}
        for name, value in values.items():
            if name not in self._values:
                _unknown(name)
            normalized[name] = value
        values = deepcopy(self._values)
        values.update(normalized)
        return RayTracingPreset(values)

    def to_dict(self):
        return deepcopy(self._values)

    def as_preset(self):
        return RayTracingPreset(self._values)


class _ExperimentalRayTracingSettings:
    """Performance and capability switches tied to the current kernels.

    These are real settings, but they are not part of Algan's supported
    surface: names, defaults and semantics follow the renderer implementation
    and can change between releases. Reach for them when profiling or working
    around a renderer limitation, not in ordinary scenes.
    """

    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __dir__(self):
        return sorted(_experimental_fields())

    def __repr__(self):
        values = self._parent.to_dict()
        shown = ", ".join(
            f"{name}={values[name]!r}" for name in sorted(_experimental_fields())
        )
        return f"RayTracingSettings.experimental({shown})"

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        self.set(**{name: value})

    def set(self, source=None, **kwargs):
        self._parent._set(source, kwargs, allow_experimental=True)
        return self

    def to_dict(self):
        values = self._parent.to_dict()
        return {name: values[name] for name in sorted(_experimental_fields())}

    @contextmanager
    def override(self, **kwargs):
        previous = self._parent.to_dict()
        self.set(**kwargs)
        try:
            yield self
        finally:
            self._parent._restore(previous)


class RayTracingSettings:
    """Stable mutable view over the ray-tracer's live configuration.

    The settings reachable directly on this object describe what the renderer
    produces (sampling, bounces, shadows, lighting, tonemapping). Internal
    performance switches live on :attr:`experimental`.

    ``algan.rendering.raytracing.settings`` is the storage behind both views,
    and holds each field under the same name this object exposes it by. Engine
    code binds that module and reads the fields off it directly on the hot
    path -- a read there is ~25 ns against ~870 ns through this object's
    ``__getattr__`` -- which is why the storage lives there rather than here.
    """

    @property
    def is_preset(self):
        return False

    @property
    def experimental(self):
        view = self.__dict__.get("_experimental")
        if view is None:
            view = _ExperimentalRayTracingSettings(self)
            object.__setattr__(self, "_experimental", view)
        return view

    @classmethod
    def field_names(cls):
        return _field_names()

    @classmethod
    def public_field_names(cls):
        return _PUBLIC_FIELDS

    def __dir__(self):
        return sorted(_PUBLIC_FIELDS | {"experimental", "set", "override", "to_dict"})

    def __repr__(self):
        values = self.to_dict()
        shown = ", ".join(f"{name}={values[name]!r}" for name in sorted(_PUBLIC_FIELDS))
        return (
            f"RayTracingSettings({shown}, "
            f"experimental=<{len(_experimental_fields())} switches>)"
        )

    def __getattr__(self, name):
        # Reads stay unrestricted: engine modules bind this object once and
        # read experimental switches off it on the hot path.
        if name.startswith("set_") and name[4:] in _field_names():
            return lambda value: self._set(None, {name[4:]: value}, True)
        module = _module()
        if name in _DEFAULT_TYPES:
            return getattr(module, name)
        # Preserve helper functions such as analytic_aa_tri_active().
        try:
            return getattr(module, name)
        except AttributeError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        if name not in _field_names():
            raise AttributeError(name)
        self.set(**{name: value})

    def set(self, source=None, **kwargs):
        return self._set(source, kwargs, allow_experimental=False)

    def _set(self, source, kwargs, allow_experimental):
        values = {}
        if source is not None:
            if not isinstance(source, (RayTracingPreset, RayTracingSettings)):
                raise AlganConfigurationError(
                    "RayTracingSettings.set expected RayTracingPreset or "
                    f"RayTracingSettings, got {type(source).__name__}"
                )
            values.update(source.to_dict())
            # A whole captured configuration necessarily carries the internal
            # switches; restoring one is not a request to tune them by hand.
            allow_experimental = True
        values.update(kwargs)

        # Only a field the caller named is refused. Restoring a captured
        # configuration (``source``) still round-trips every field, inert ones
        # included -- a snapshot is not a request to tune anything.
        for name in kwargs:
            message = _INERT_FIELDS.get(name)
            if message is not None:
                raise AlganConfigurationError(message)

        # Resolved before the validation loop, not after: _DEFAULT_TYPES is
        # populated as a side effect of this call, and _check_value reads it.
        module = _module()

        normalized = []
        for name, value in values.items():
            field = name
            if field not in _field_names():
                _unknown(field)
            if not allow_experimental and field not in _PUBLIC_FIELDS:
                raise AlganConfigurationError(
                    f"'{field}' is an experimental renderer switch. Set it with "
                    f"SETTINGS.raytracing.experimental.set({field}=...) if you "
                    "accept that its name and behaviour can change."
                )
            normalized.append((field, _check_value(field, value, module)))

        previous = self.to_dict()
        field = None
        try:
            for field, value in normalized:
                setter = _setter(module, field)
                if setter is not None:
                    setter(value)
                else:
                    setattr(module, field, value)
                    if field == "refract_initial_pool_ratio":
                        module.REFRACT_SPLIT_SLOTS = module.refract_initial_pool_ratio
        except AlganError:
            # Algan's own errors already say what is wrong and which setting;
            # UnsupportedFeatureError in particular is a distinct type callers
            # catch, so it must not be flattened into a configuration error.
            # Listed first because AlganConfigurationError *is* a ValueError.
            self._restore(previous)
            raise
        except (ValueError, TypeError) as exc:
            # A setter rejecting its argument -- set_tonemap_method raises a
            # bare ValueError, and a coercion like float(x) raises TypeError.
            # Both are configuration mistakes and should arrive as one, naming
            # the field, rather than as a raw builtin from two frames down.
            self._restore(previous)
            raise AlganConfigurationError(f"'{field}': {exc}") from exc
        except Exception:
            self._restore(previous)
            raise
        return self

    def to_dict(self):
        module = _module()
        return {field: deepcopy(getattr(module, field)) for field in _field_names()}

    def as_preset(self):
        return RayTracingPreset(self.to_dict())

    def effective_analytic_aa_secondary_samples(self):
        """Return the effective secondary sample count for current AA mode."""
        return _module().effective_analytic_aa_secondary_samples()

    def _restore(self, values):
        """Restore an already-validated snapshot without invoking setters."""
        module = _module()
        for field, value in values.items():
            if field not in _field_names():
                _unknown(field)
            setattr(module, field, deepcopy(value))
        module.REFRACT_SPLIT_SLOTS = module.refract_initial_pool_ratio
        return self

    @contextmanager
    def override(self, **kwargs):
        previous = self.to_dict()
        self.set(**kwargs)
        try:
            yield self
        finally:
            self._restore(previous)
