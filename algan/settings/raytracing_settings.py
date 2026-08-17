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

:class:`RayTracingPreset` captures a configuration for reuse. Like the video
presets it is immutable, so ``set()`` on one returns a copy.

Read these live (``rt_settings.X`` at call time) rather than importing them by
value at module import, which would freeze them before user code runs.
"""

from __future__ import annotations

import difflib
import importlib
from contextlib import contextmanager
from copy import deepcopy

from algan.errors import AlganConfigurationError

_FIELD_TO_LEGACY = {
    name.lower(): name
    for name in (
        "MAX_BOUNCES",
        "SAMPLES_PER_PIXEL",
        "UNSUPPORTED_FEATURE_POLICY",
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
        "RASTER_PAIR_FLAGS",
        "RASTER_COVERED_SHADE",
        "RASTER_SPARSE_COVERAGE",
        "ANALYTIC_AA",
        "ANALYTIC_AA_BEZ",
        "ANALYTIC_AA_TRI",
        "ANALYTIC_AA_SEAM",
        "ANALYTIC_AA_SLIVER",
        "ANALYTIC_AA_SECONDARY_SAMPLES",
        "ANALYTIC_AA_SECONDARY_MIN_ENERGY",
        "ANALYTIC_AA_BEZ_MIN_HALF_WIDTH",
        "ANALYTIC_AA_CHORD_TOLERANCE",
        "GLOSSY_REFLECTION",
        "GLOSSY_INTERLEAVE",
        "WF_TEXTURED",
        "MERGE_ON_GPU",
        "MERGE_GPU_PEAK_FACTOR",
        "MERGE_TRACK_PEAK",
        "PROJECT_ON_GPU",
        "PROJECT_GPU_PEAK_FACTOR",
        "PN_CRITERION_KERNEL",
        "MESH_ID",
        "POLYHEDRON_WINDING",
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
        "analytic_aa",
        "tonemapping",
        "tonemap_method",
        "tonemap_exposure",
        "unsupported_feature_policy",
    }
)

_EXPERIMENTAL_FIELDS = frozenset(_FIELD_TO_LEGACY) - _PUBLIC_FIELDS

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
    "raster_pair_flags": "set_raster_pair_flags",
    "raster_covered_shade": "set_raster_covered_shade",
    "raster_sparse_coverage": "set_raster_sparse_coverage",
    "analytic_aa": "set_analytic_aa",
    "glossy_reflection": "set_glossy_reflection",
    "wf_textured": "set_textured_wavefront",
    "merge_on_gpu": "set_merge_on_gpu",
    "project_on_gpu": "set_project_on_gpu",
    "pn_criterion_kernel": "set_pn_criterion_kernel",
    "wf_textured_features": "set_textured_features",
    "wavefront_sort_materials": "set_material_sorting",
    "fragment_shading": "set_fragment_shading",
    "shadows": "set_ray_traced_shadows",
    "light_intensity": "set_light_intensity",
    "ambient_light": "set_ambient_light",
    "samples_per_pixel": "set_samples_per_pixel",
    "indirect_bounce_strength": "set_indirect_bounce_strength",
    "tonemapping": "set_tonemapping",
    "tonemap_exposure": "set_tonemap_exposure",
    "tonemap_method": "set_tonemap_method",
    "post_process_tonemap": "set_post_process_tonemap",
    "post_tonemap_kernel": "set_post_tonemap_kernel",
}


def _module():
    return importlib.import_module("algan.rendering.raytracing.settings")


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
            normalized.append((field, value))

        previous = self.to_dict()
        module = _module()
        try:
            for field, value in normalized:
                setter_name = _SETTER_OVERRIDES.get(field)
                if setter_name is not None:
                    getattr(module, setter_name)(value)
                else:
                    setattr(module, _FIELD_TO_LEGACY[field], value)
                    if field == "refract_initial_pool_ratio":
                        module.REFRACT_SPLIT_SLOTS = module.REFRACT_INITIAL_POOL_RATIO
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
