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

Every write is validated. A field's accepted type is derived from the value it
ships with rather than declared in a table, because a 106-row table beside 106
defaults is a second source of truth that drifts; :data:`_POLYMORPHIC_FIELDS`
lists the three mode switches where that inference is wrong. Numeric fields
additionally carry a lower bound taken from their own documented meaning
(:data:`_MINIMUMS`) -- a count of rays cannot be below 1, a multiplier the
memory model scales an estimate by cannot be 0 -- and floats must be finite.
Before this, only the fields with a ``_SETTER_OVERRIDES`` entry were checked at
all, and only as far as that setter's own ``bool()``/``float()`` coercion went:
``max_bounces = 'x'`` stored the string and failed much later inside a kernel,
with nothing pointing back at the setting.

:class:`RayTracingPreset` captures a configuration for reuse. Like the video
presets it is immutable, so ``set()`` on one returns a copy.

Read these live (``rt_settings.X`` at call time) rather than importing them by
value at module import, which would freeze them before user code runs.
"""

from __future__ import annotations

import difflib
import importlib
import math
from contextlib import contextmanager
from copy import deepcopy

from algan.errors import AlganConfigurationError, AlganError

_FIELD_TO_LEGACY = {
    name.lower(): name
    for name in (
        "MAX_BOUNCES",
        "SAMPLES_PER_PIXEL",
        "UNSUPPORTED_FEATURE_POLICY",
        "LINEAR_COLOR_SPACE",
        "TONEMAPPING",
        "TONEMAP_EXPOSURE",
        "TONEMAP_METHOD",
        "POST_PROCESS_TONEMAP",
        "INDIRECT_BOUNCE_STRENGTH",
        "LIGHT_INTENSITY",
        "AMBIENT_LIGHT",
        "GATE_EMPTY_TRAVERSALS",
        "WF_REVALIDATE_PENDING",
        "WF_NEAR_FIRST",
        "WF_OPAQUE_CLOSEST",
        "WF_OPAQUE_PREPASS",
        "INPLACE_AA",
        "WAVEFRONT_TILE_RAYS",
        "WAVEFRONT_TILE_AUTO",
        "WAVEFRONT_TILE_SAFETY",
        "WAVEFRONT_TILE_MIN",
        "WAVEFRONT_TILE_MAX",
        "WF_COMPACT_ACTIVE_ONLY",
        "REFRACT_INITIAL_POOL_RATIO",
        "FRAGMENT_SHADING",
        "PROMOTE_CONSTANTS",
        "WF_SKIP_UNLIT_NORMAL",
        "WF_GEN_FUSED",
        "WF_GEN_FUSED_GAIN",
        "WF_GEN_FUSED_MIN_WIN",
        "SPARSE_DISCOVERY_SAFETY",
        "WF_MEM_TRIM",
        "BVH_REFIT",
        "BVH_DEFER",
        "HYBRID_RASTER",
        "RASTER_SS",
        "RASTER_BEZ_PRECOMPUTE",
        "RASTER_TRI_PRECOMPUTE",
        "RASTER_EMPTY_SKIP",
        "RASTER_FUSED_GATHER",
        "RASTER_OPAQUE_TRUNC_KERNEL",
        "RASTER_PAIR_EXPAND_KERNEL",
        "RASTER_PAIR_FLAGS",
        "RASTER_COVERED_SHADE",
        "RASTER_SPARSE_COVERAGE",
        "SHEET_BAND_STATS_KERNEL",
        "SHEET_MASK_KERNEL",
        "SHEET_ONE_MESH_KERNEL",
        "SHEET_POSITIONED_DEPTH",
        "SHEET_RANK_KERNEL",
        "SHEET_RESOLVE",
        "SHEET_SAMPLE_DEPTH",
        "SHEET_SAMPLE_DEPTH_KERNEL",
        "SHEET_SHADE_SPLIT",
        "SHEET_RESOLVE_MEMO",
        "SHEET_SHELL_CEILING_KERNEL",
        "ANALYTIC_AA",
        "ANALYTIC_AA_BEZ",
        "ANALYTIC_AA_TRI",
        "ANALYTIC_AA_SEAM",
        "ANALYTIC_AA_RUN_FULL",
        "ANALYTIC_AA_ONE_MESH",
        "AREA_LIGHT_SOFT_SHADOWS",
        "SOLID_SHELL_ALPHA",
        "DIRECT_SPECULAR_LOBE",
        "WEIGHT_FLOOR_EXIT",
        "BEZ_BVH_SPLIT",
        "WELD_SURFACE_SEAMS",
        "ANALYTIC_AA_SLIVER",
        "ANALYTIC_AA_SECONDARY_SAMPLES",
        "ANALYTIC_AA_SECONDARY_MIN_ENERGY",
        "ANALYTIC_AA_BEZ_MIN_HALF_WIDTH",
        "ANALYTIC_AA_CHORD_TOLERANCE",
        "GLOSSY_REFLECTION",
        "GLOSSY_INTERLEAVE",
        "GLOSSY_PREFILTER",
        "GLOSSY_PREFILTER_MAX_LEVELS",
        "WF_TEXTURED",
        "MERGE_DEDUP_GEOMETRY",
        "MERGE_DEDUP_TIME",
        "TEXTURE_TIME_FLAT",
        "TEXTURE_CONTENT_DEDUP",
        "TEXTURE_WINDOW_COLLAPSE",
        "TEXTURE_OPACITY_IN_KERNEL",
        "TEXTURE_U8_STORAGE",
        "TEXTURE_TIME_LERP",
        "MERGE_ON_GPU",
        "MERGE_GPU_PEAK_FACTOR",
        "MERGE_TRACK_PEAK",
        "PROJECT_ON_GPU",
        "PROJECT_GPU_PEAK_FACTOR",
        "PN_CRITERION_KERNEL",
        "PN_GEOMETRY_SLACK",
        "PN_ANISOTROPIC_DICE",
        "MESH_ID",
        "NESTED_IOR",
        "POLYHEDRON_WINDING",
        "SHADOW_EPS_RELATIVE",
        "SHADOW_IDENTITY_REJECT",
        "SHADOW_NEAR_FRACTION",
        "SHADOW_TERMINATOR",
        "RGB_SHADOW_TINT",
        "WF_TEXTURED_FEATURES",
        "WAVEFRONT_SORT_MATERIALS",
        "SHADOWS",
        "POST_TONEMAP_KERNEL",
    )
}
_LEGACY_TO_FIELD = {value: key for key, value in _FIELD_TO_LEGACY.items()}

# The settings that describe what the renderer *produces*. Everything else in
# _FIELD_TO_LEGACY is a performance or capability switch whose meaning is tied
# to the current kernel implementation; those live under ``.experimental`` so
# that the supported surface is obvious from tab-completion and repr.
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

_EXPERIMENTAL_FIELDS = frozenset(_FIELD_TO_LEGACY) - _PUBLIC_FIELDS

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

_SETTER_OVERRIDES = {
    "unsupported_feature_policy": "set_unsupported_feature_policy",
    "wavefront_tile_auto": "set_wavefront_tile_auto",
    "wf_gen_fused": "set_gen_fused",
    "bvh_refit": "set_refit_bvh",
    "bvh_defer": "set_bvh_defer",
    "hybrid_raster": "set_hybrid_raster",
    "raster_ss": "set_raster_screen_space",
    "raster_bez_precompute": "set_raster_bez_precompute",
    "raster_tri_precompute": "set_raster_tri_precompute",
    "raster_empty_skip": "set_raster_empty_skip",
    "raster_fused_gather": "set_raster_fused_gather",
    "raster_opaque_trunc_kernel": "set_raster_opaque_trunc_kernel",
    "raster_pair_expand_kernel": "set_raster_pair_expand_kernel",
    "raster_pair_flags": "set_raster_pair_flags",
    "raster_covered_shade": "set_raster_covered_shade",
    "raster_sparse_coverage": "set_raster_sparse_coverage",
    "sheet_band_stats_kernel": "set_sheet_band_stats_kernel",
    "sheet_mask_kernel": "set_sheet_mask_kernel",
    "sheet_one_mesh_kernel": "set_sheet_one_mesh_kernel",
    "sheet_positioned_depth": "set_sheet_positioned_depth",
    "sheet_rank_kernel": "set_sheet_rank_kernel",
    "sheet_resolve": "set_sheet_resolve",
    "sheet_sample_depth": "set_sheet_sample_depth",
    "sheet_sample_depth_kernel": "set_sheet_sample_depth_kernel",
    "sheet_shade_split": "set_sheet_shade_split",
    "sheet_shell_ceiling_kernel": "set_sheet_shell_ceiling_kernel",
    "analytic_aa": "set_analytic_aa",
    "glossy_reflection": "set_glossy_reflection",
    "wf_textured": "set_textured_wavefront",
    "merge_on_gpu": "set_merge_on_gpu",
    "project_on_gpu": "set_project_on_gpu",
    "nested_ior": "set_nested_ior",
    "pn_criterion_kernel": "set_pn_criterion_kernel",
    "wf_textured_features": "set_textured_features",
    "wavefront_sort_materials": "set_material_sorting",
    "shadow_terminator": "set_shadow_terminator",
    "fragment_shading": "set_fragment_shading",
    "shadows": "set_ray_traced_shadows",
    "light_intensity": "set_light_intensity",
    "ambient_light": "set_ambient_light",
    "samples_per_pixel": "set_samples_per_pixel",
    "indirect_bounce_strength": "set_indirect_bounce_strength",
    "linear_color_space": "set_linear_color_space",
    "tonemapping": "set_tonemapping",
    "tonemap_exposure": "set_tonemap_exposure",
    "tonemap_method": "set_tonemap_method",
    "post_process_tonemap": "set_post_process_tonemap",
    "post_tonemap_kernel": "set_post_tonemap_kernel",
}


#: The type each field ships with, keyed by field name. Captured the first time
#: the legacy module is resolved, which is before any ``set`` can have written
#: to it -- so these are the shipped defaults rather than whatever the current
#: configuration happens to hold. That matters for ``shadow_terminator``, whose
#: default is a bool but which stores ``2`` once someone selects ``"relax"``.
_DEFAULT_TYPES = {}


def _module():
    module = importlib.import_module("algan.rendering.raytracing.settings")
    if not _DEFAULT_TYPES:
        _DEFAULT_TYPES.update(
            (field, type(getattr(module, legacy)))
            for field, legacy in _FIELD_TO_LEGACY.items()
        )
    return module


#: Fields whose setter deliberately accepts a type other than the one the field
#: ships with, so the derived type check has to stand aside for them. Each is a
#: mode switch that spells one of its states as a bool and another as a string.
_POLYMORPHIC_FIELDS = frozenset(
    {
        "shadow_terminator",  # bool, plus "relax" for the third state
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

#: String fields whose accepted values are enumerated in the legacy module.
#: Named by the tuple that holds them so the two cannot drift apart. The other
#: four string fields validate inside their own setter.
_CHOICES = {"analytic_aa_sliver": "ANALYTIC_AA_SLIVER_MODES"}


def _check_value(field, value, module):
    """Validate one field's value against what it ships with, and normalize it.

    ``module`` is the resolved legacy module. Taking it as an argument is not
    just convenience: resolving it is what populates :data:`_DEFAULT_TYPES`, so
    a caller that validates before resolving it would find that dict empty and
    check nothing at all.

    Fields with a ``_SETTER_OVERRIDES`` entry used to be the only ones checked
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
    suggestion = difflib.get_close_matches(name, sorted(_FIELD_TO_LEGACY), n=1)
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
        field = _LEGACY_TO_FIELD.get(name, name)
        if field in self._values:
            return self._values[field]
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
            field = _LEGACY_TO_FIELD.get(name, name)
            if field not in self._values:
                _unknown(field)
            normalized[field] = value
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
        return sorted(_EXPERIMENTAL_FIELDS)

    def __repr__(self):
        values = self._parent.to_dict()
        shown = ", ".join(
            f"{name}={values[name]!r}" for name in sorted(_EXPERIMENTAL_FIELDS)
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
        return {name: values[name] for name in sorted(_EXPERIMENTAL_FIELDS)}

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

    The legacy ``algan.rendering.raytracing.settings`` module remains the
    storage behind both views; engine code reads it live so that public
    setters take effect immediately.
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
        return frozenset(_FIELD_TO_LEGACY)

    @classmethod
    def public_field_names(cls):
        return _PUBLIC_FIELDS

    def __dir__(self):
        return sorted(_PUBLIC_FIELDS | {"experimental", "set", "override", "to_dict"})

    def __repr__(self):
        values = self.to_dict()
        shown = ", ".join(f"{name}={values[name]!r}" for name in sorted(_PUBLIC_FIELDS))
        return f"RayTracingSettings({shown}, experimental=<{len(_EXPERIMENTAL_FIELDS)} switches>)"

    def __getattr__(self, name):
        # Reads stay unrestricted: engine modules bind this object once and
        # read experimental switches off it on the hot path.
        if name.startswith("set_") and name[4:] in _FIELD_TO_LEGACY:
            return lambda value: self._set(None, {name[4:]: value}, True)
        module = _module()
        field = _LEGACY_TO_FIELD.get(name, name)
        legacy = _FIELD_TO_LEGACY.get(field)
        if legacy is not None:
            return getattr(module, legacy)
        # Preserve helper functions such as analytic_aa_tri_active().
        try:
            return getattr(module, name)
        except AttributeError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        field = _LEGACY_TO_FIELD.get(name, name)
        if field not in _FIELD_TO_LEGACY:
            raise AttributeError(name)
        self.set(**{field: value})

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
            message = _INERT_FIELDS.get(_LEGACY_TO_FIELD.get(name, name))
            if message is not None:
                raise AlganConfigurationError(message)

        # Resolved before the validation loop, not after: _DEFAULT_TYPES is
        # populated as a side effect of this call, and _check_value reads it.
        module = _module()

        normalized = []
        for name, value in values.items():
            field = _LEGACY_TO_FIELD.get(name, name)
            if field not in _FIELD_TO_LEGACY:
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
                setter_name = _SETTER_OVERRIDES.get(field)
                if setter_name is not None:
                    getattr(module, setter_name)(value)
                else:
                    setattr(module, _FIELD_TO_LEGACY[field], value)
                    if field == "refract_initial_pool_ratio":
                        module.REFRACT_SPLIT_SLOTS = module.REFRACT_INITIAL_POOL_RATIO
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
        return {
            field: deepcopy(getattr(module, legacy))
            for field, legacy in _FIELD_TO_LEGACY.items()
        }

    def as_preset(self):
        return RayTracingPreset(self.to_dict())

    def effective_analytic_aa_secondary_samples(self):
        """Return the effective secondary sample count for current AA mode."""
        return _module().analytic_aa_secondary_samples()

    def _restore(self, values):
        """Restore an already-validated snapshot without invoking setters."""
        module = _module()
        for field, value in values.items():
            if field not in _FIELD_TO_LEGACY:
                _unknown(field)
            setattr(module, _FIELD_TO_LEGACY[field], deepcopy(value))
        module.REFRACT_SPLIT_SLOTS = module.REFRACT_INITIAL_POOL_RATIO
        return self

    @contextmanager
    def override(self, **kwargs):
        previous = self.to_dict()
        self.set(**kwargs)
        try:
            yield self
        finally:
            self._restore(previous)
