"""Custom & composable per-fragment shaders for the deterministic ray tracer.

A **fragment stage** is a Taichi ``@ti.func`` with the uniform stage contract
(see :mod:`algan.rendering.raytracing.shading_taichi`): it shades one surface hit
from the running colour, the interpolated surface attributes, its slice of the
per-primitive parameter block and the scene lights, and returns the new
``vec4`` (RGB + glow). Stages are composed into a **pipeline** (a list run
left-to-right, each fed the previous stage's output) via :func:`register_pipeline`,
which bakes the pipeline into a single injected ``@ti.func`` (see
``taichi-func-injection``) and hands back a per-primitive pipeline id.

Users reach this through
:meth:`algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`.
A stage
may be a built-in material (the ``@ti.func`` ports of the core lit shaders,
exposed here as :class:`FragmentStage` and also resolvable from their PyTorch
shader functions, e.g. ``phong_shader``) or a custom :class:`FragmentStage`
wrapping the user's own ``@ti.func`` plus its animatable parameter specs.

Because a stage is plain Taichi scalar math, existing built-in materials double
as fragment shaders and compose with custom ones, e.g.
``mob.set_fragment_shader([cosine_color, phong_shader])`` recolours each fragment
with a cosine wave and then lights the result with Blinn-Phong.
"""

import taichi as ti

from algan.rendering.raytracing.shading_taichi import (
    _USER_PIPELINE_BASE,
    _stage_default,
    _stage_lambert,
    _stage_phong,
    _stage_physical,
    _stage_standard,
    _stage_unlit,
    make_pipeline_func,
)


class FragmentStage:
    """A fragment shader stage: a Taichi ``@ti.func`` plus its parameter specs.

    ``ti_func`` must follow the stage contract in
    :func:`~algan.rendering.raytracing.shading_taichi._stage_phong`. ``param_specs``
    is an ordered list of ``(name, width, default)`` for the stage's animatable
    parameters; the stage reads them from ``params[tm, prim, off + slot]`` where
    ``slot`` is that parameter's cumulative offset within the stage (so the first
    param is at ``off + 0``). ``width`` is 1 for a scalar, 3 for an RGB triple.

    ``scatter`` optionally customises how a ray *continues* after this stage's
    pipeline shades a surface hit (reflection / refraction / pass-through) on
    the sorted-material wavefront: a ``@ti.func`` following the scatter
    contract documented in
    :mod:`algan.rendering.raytracing.shading_taichi`. When no stage of a
    pipeline supplies one, the default scatter applies the classic
    opacity/reflectivity/Fresnel-glass behaviour. When several stages supply
    one, the last stage's scatter wins.
    """

    def __init__(self, ti_func, param_specs=(), scatter=None):
        self.ti_func = ti_func
        self.param_specs = [(str(n), int(w), d) for (n, w, d) in param_specs]
        self.width = sum(w for _n, w, _d in self.param_specs)
        self.scatter = scatter


# Canonical 12-slot built-in material parameter layout (matches the slot map in
# shading_taichi: emissive[0:3] emissive_intensity[3] specular[4:7] shininess[7]
# roughness[8] metalness[9] flat_shading[10] env_map_intensity[11]).
_BUILTIN_MAT_SPECS = [
    ("emissive", 3, (0.0, 0.0, 0.0)),
    ("emissive_intensity", 1, 1.0),
    ("specular", 3, (0.0666, 0.0666, 0.0666)),
    ("shininess", 1, 30.0),
    ("roughness", 1, 1.0),
    ("metalness", 1, 0.0),
    ("flat_shading", 1, 0.0),
    ("env_map_intensity", 1, 1.0),
]

# The physical stage extends the canonical layout to the full 26-slot block
# (slots 12..25 -- see the slot map in shading_taichi); its first 12 slots are
# identical to the shared layout.
_PHYSICAL_MAT_SPECS = _BUILTIN_MAT_SPECS + [
    ("ior", 1, 1.5),
    ("specular_intensity", 1, 1.0),
    ("specular_color", 3, (1.0, 1.0, 1.0)),
    ("clearcoat", 1, 0.0),
    ("clearcoat_roughness", 1, 0.0),
    ("sheen", 1, 0.0),
    ("sheen_roughness", 1, 1.0),
    ("sheen_color", 3, (0.0, 0.0, 0.0)),
    ("transmission", 1, 0.0),
    ("iridescence", 1, 0.0),
]

# Built-in material stages (12-slot canonical params; physical 26). ``default``
# and ``unlit`` ignore most slots but share the layout so offsets stay uniform
# in a pipeline.
STAGE_DEFAULT = FragmentStage(_stage_default, _BUILTIN_MAT_SPECS)
STAGE_UNLIT = FragmentStage(_stage_unlit, _BUILTIN_MAT_SPECS)
STAGE_LAMBERT = FragmentStage(_stage_lambert, _BUILTIN_MAT_SPECS)
STAGE_PHONG = FragmentStage(_stage_phong, _BUILTIN_MAT_SPECS)
STAGE_STANDARD = FragmentStage(_stage_standard, _BUILTIN_MAT_SPECS)
STAGE_PHYSICAL = FragmentStage(_stage_physical, _PHYSICAL_MAT_SPECS)


def _builtin_shader_to_stage():
    """Map the built-in PyTorch material shader functions to their stage ports,
    so ``set_fragment_shader(phong_shader)`` resolves to ``STAGE_PHONG``.
    """
    from algan.rendering.shaders.material_shaders import (
        basic_material_shader,
        lambert_shader,
        phong_shader,
        physical_shader,
        standard_shader,
    )
    from algan.rendering.shaders.pbr_shaders import default_shader, null_shader

    return {
        default_shader: STAGE_DEFAULT,
        null_shader: STAGE_UNLIT,
        basic_material_shader: STAGE_UNLIT,
        lambert_shader: STAGE_LAMBERT,
        phong_shader: STAGE_PHONG,
        standard_shader: STAGE_STANDARD,
        physical_shader: STAGE_PHYSICAL,
    }


_SHADER_TO_STAGE = None


def resolve_stage(shader):
    """Resolve a user-supplied shader to a :class:`FragmentStage`.

    Accepts a :class:`FragmentStage` (returned as-is) or a built-in PyTorch
    material shader function (mapped to its stage port). Raises for anything
    else -- a custom fragment shader must be wrapped in a ``FragmentStage`` so
    its parameters are known.
    """
    global _SHADER_TO_STAGE
    if isinstance(shader, FragmentStage):
        return shader
    if _SHADER_TO_STAGE is None:
        _SHADER_TO_STAGE = _builtin_shader_to_stage()
    if shader in _SHADER_TO_STAGE:
        return _SHADER_TO_STAGE[shader]
    raise TypeError(
        f"{shader!r} is not a valid fragment shader. Pass a FragmentStage "
        "(a @ti.func stage + its param specs) or a built-in material shader "
        "such as phong_shader / standard_shader."
    )


# --- Pipeline registry (session-global; ids are stable so the packed pipeline
# --- ids stay consistent with the injected pipeline tuple). ------------------
# key -> (pipeline_id, composed_func); _PIPELINE_LIST[pid - _USER_PIPELINE_BASE].
# _PIPELINE_SCATTERS parallels _PIPELINE_LIST: the pipeline's custom scatter
# ``@ti.func`` (the last stage-supplied one), or None for the default scatter.
_PIPELINE_REGISTRY = {}
_PIPELINE_LIST = []
_PIPELINE_SCATTERS = []


def register_pipeline(stages):
    """Register a pipeline (list of :class:`FragmentStage`) for in-kernel use.

    Returns ``(pipeline_id, total_width, layout)`` where ``layout`` is a list of
    ``(name, slot, width, default)`` for every parameter across all stages (with
    ``slot`` the absolute offset into the per-primitive param block). Identical
    pipelines (same stage funcs + widths + scatter) reuse the same id and
    composed func.
    """
    stages = list(stages)
    offsets = []
    off = 0
    for s in stages:
        offsets.append(off)
        off += s.width
    total_width = off
    scatters = [s.scatter for s in stages if getattr(s, "scatter", None)]
    scatter = scatters[-1] if scatters else None

    key = (
        tuple(id(s.ti_func) for s in stages),
        tuple(s.width for s in stages),
        id(scatter) if scatter is not None else None,
    )
    if key in _PIPELINE_REGISTRY:
        pid = _PIPELINE_REGISTRY[key][0]
    else:
        pid = _USER_PIPELINE_BASE + len(_PIPELINE_LIST)
        composed = make_pipeline_func([s.ti_func for s in stages], offsets)
        _PIPELINE_REGISTRY[key] = (pid, composed)
        _PIPELINE_LIST.append(composed)
        _PIPELINE_SCATTERS.append(scatter)

    layout = []
    for s, base in zip(stages, offsets):
        slot = base
        for name, width, default in s.param_specs:
            layout.append((name, slot, width, default))
            slot += width
    return pid, total_width, layout


def _select_by_pid(entries, pids):
    """``entries`` (a registry list indexed by ``pid - _USER_PIPELINE_BASE``)
    narrowed to the pipeline ids in ``pids``, or the whole list when ``pids``
    is None.

    Slots outside ``pids`` become ``None`` -- the position of a pipeline IS its
    id, so the slots cannot be closed up without renumbering every packed
    per-primitive material id -- and the tuple is then trimmed after its last
    live entry. Both kernel-side dispatches already compile a ``None`` slot out
    (``ti.static(bool(fn))``), so a narrowed tuple costs the kernel nothing it
    would not have paid anyway.
    """
    if pids is None:
        selected = list(entries)
    else:
        keep = {int(pid) - _USER_PIPELINE_BASE for pid in pids}
        selected = [entry if i in keep else None for i, entry in enumerate(entries)]
    while selected and selected[-1] is None:
        selected.pop()
    return tuple(selected)


def build_frag_pipelines(pids=None):
    """Composed pipeline funcs to inject as the shade kernel's
    ``frag_pipelines`` template argument, indexed by ``pid -
    _USER_PIPELINE_BASE`` and ordered by id.

    ``pids`` is the set of material pipeline ids the batch being rendered
    actually carries; everything else is dropped (see :func:`_select_by_pid`).
    Pass it. **The registry is process-global and append-only, and Taichi
    specialises the shade kernels on this tuple**, so handing over the whole
    registry puts every render in a process that ever registered a pipeline
    onto its own uncached kernel variant -- including renders with no custom
    shader at all, which is both the pathology this argument exists to close
    and the reason a batch-narrowed tuple is what the tracer passes. ``None``
    (the whole registry) is the conservative fallback for a batch whose merged
    scene cannot enumerate its ids.
    """
    return _select_by_pid(_PIPELINE_LIST, pids)


def build_frag_scatters(pids=None):
    """Per-pipeline custom scatter funcs (None = default scatter), ordered by
    id, for the monolithic wavefront's per-material continuation dispatch.

    Narrowed by ``pids`` exactly as :func:`build_frag_pipelines` is, and for
    the same reason.
    """
    return _select_by_pid(_PIPELINE_SCATTERS, pids)


class FragmentPipelineShader:
    """Marker ``shader`` for a mob with a custom fragment pipeline.

    Carries the pipeline metadata (id, per-param slot layout, total width) so it
    reaches the ray-traced primitive through the ordinary ``shader=`` handoff
    (see ``RayTracedTrianglePrimitive._pack_frag_pipeline``). As a *vertex*
    shader it is a no-op returning the raw albedo, so if the vertex path ever
    runs it leaves the colour raw for the in-kernel pipeline to shade.
    """

    def __init__(self, pipeline_id, layout, total_width):
        self._frag_pipeline_id = pipeline_id
        self._frag_param_layout = layout  # [(name, slot, width, default)]
        self._frag_total_width = total_width

    def __call__(
        self, memory, vertex_location, vertex_normal, albedo_color, *args, **kwargs
    ):
        return albedo_color


def build_fragment_pipeline(shader):
    """Resolve + register a fragment shader (a stage or a list of stages) and
    return ``(marker, param_specs)``.

    ``marker`` is a :class:`FragmentPipelineShader` to assign to the mob's
    ``shader``; ``param_specs`` is an ordered ``[(name, default)]`` of the
    pipeline's animatable parameters (duplicate names across stages are
    suffixed) for the mob to register as animatable attributes.
    """
    stages_in = shader if isinstance(shader, (list, tuple)) else [shader]
    frag_stages = [resolve_stage(s) for s in stages_in]
    pid, total_width, raw_layout = register_pipeline(frag_stages)

    # Namespace duplicate param names across stages so animatable attrs don't
    # clash (a single occurrence keeps its bare name).
    seen = {}
    layout = []
    param_specs = []
    for name, slot, width, default in raw_layout:
        if name in seen:
            seen[name] += 1
            fname = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
            fname = name
        layout.append((fname, slot, width, default))
        param_specs.append((fname, default))

    marker = FragmentPipelineShader(pid, layout, total_width)
    return marker, param_specs


# ---------------------------------------------------------------------------
# Example custom stage.
# ---------------------------------------------------------------------------


@ti.func
def _stage_cosine_color(
    pos,
    view_dir,
    n_interp,
    face_n,
    in_rgb,
    in_glow,
    params: ti.template(),
    f,
    prim,
    off,
    light_pos: ti.template(),
    light_col: ti.template(),
    num_lights,
    shadows: ti.template(),
    vis,
):
    """Modulate the albedo by an RGB-phase-shifted cosine of world x (rainbow
    banding). Params: ``frequency`` (slot 0), ``phase`` (slot 1).
    """
    tm = f % params.shape[0]
    freq = params[tm, prim, off + 0]
    phase = params[tm, prim, off + 1]
    w = pos[0] * freq + phase
    r = 0.5 + 0.5 * ti.cos(w)
    g = 0.5 + 0.5 * ti.cos(w + 2.0943951)  # +2pi/3
    b = 0.5 + 0.5 * ti.cos(w + 4.1887902)  # +4pi/3
    return ti.math.vec4(in_rgb[0] * r, in_rgb[1] * g, in_rgb[2] * b, in_glow)


#: Example custom fragment stage: recolours each fragment with a cosine wave.
#: Compose before a lighting stage, e.g. ``[cosine_color, phong_shader]``.
cosine_color = FragmentStage(
    _stage_cosine_color,
    [("frequency", 1, 4.0), ("phase", 1, 0.0)],
)


# ---------------------------------------------------------------------------
# Example custom scatter (ray-bouncing behaviour). See the scatter contract in
# ``algan.rendering.raytracing.shading_taichi``.
# ---------------------------------------------------------------------------


@ti.func
def _scatter_forced_mirror(
    rd,
    n_interp,
    face_n,
    hit_point,
    shaded,
    albedo,
    alpha,
    reflectivity,
    ior,
    transmission,
    params: ti.template(),
    f,
    prim,
    bounces_left,
    refraction: ti.template(),
):
    """Treat the surface as a 85% mirror regardless of its per-vertex
    reflectivity: commit 15% of the shaded colour and bounce the remaining
    throughput along the mirror direction (no transmission). Branch weights
    are vec3 per-channel throughput multipliers (colour transport); this
    scatter reflects achromatically, so all three channels match.
    """
    n = n_interp.normalized()
    if n.dot(rd) > 0.0:
        n = -n
    refl_dir = (rd - 2.0 * rd.dot(n) * n).normalized()
    refl_orig = hit_point + n * 1e-3  # 10 * MIN_HIT_DISTANCE
    contrib = (alpha * 0.15) * shaded
    zero3 = ti.math.vec3(0.0, 0.0, 0.0)
    rw = 0.85 * alpha
    if bounces_left <= 0:  # out of bounces: absorb instead of reflecting
        rw = 0.0
    refl_w = ti.math.vec3(rw, rw, rw)
    pass_w = ti.math.vec3(1.0 - alpha, 1.0 - alpha, 1.0 - alpha)
    return (contrib, pass_w, refl_orig, refl_dir, refl_w, zero3, zero3, zero3)


#: Example custom scatter: forces mirror bouncing regardless of the mob's
#: reflectivity. Attach to a stage, e.g.
#: ``mob.set_fragment_shader(FragmentStage(_stage, specs, scatter=...))`` or
#: compose with a built-in stage: ``FragmentStage(STAGE_PHONG.ti_func,
#: STAGE_PHONG.param_specs, scatter=forced_mirror_scatter.scatter)``.
forced_mirror_scatter = FragmentStage(
    _stage_unlit, _BUILTIN_MAT_SPECS, scatter=_scatter_forced_mirror
)
