"""Custom and composable per-fragment shaders shared by the renderers.

A **fragment stage** is a Taichi ``@ti.func`` with the uniform stage contract
(see :mod:`algan.rendering.raytracing.shading_taichi`): it shades one surface hit
from the running color, the interpolated surface attributes, its slice of the
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
``mob.set_fragment_shader([cosine_color, phong_shader])`` recolors each fragment
with a cosine wave and then lights the result with Blinn-Phong.
"""

from algan.rendering.raytracing.raytrace_kernels_taichi import min_alpha
from algan.rendering.raytracing.settings import _MAT_COLOR_SLOT_NAMES
from algan.rendering.raytracing.shading_taichi import (
    _USER_PIPELINE_BASE,
    _reflect_frame,
    _stage_lambert,
    _stage_manim,
    _stage_phong,
    _stage_physical,
    _stage_standard,
    _stage_unlit,
    make_pipeline_func,
)
from algan.rendering.raytracing.transport_taichi import _offset_ray_origin
from algan.taichi_compat import ti


class FragmentStage:
    """A fragment shader stage: a Taichi ``@ti.func`` plus its parameter specs.

    ``ti_func`` must follow the stage contract in
    :func:`~algan.rendering.raytracing.shading_taichi._stage_phong`. ``param_specs``
    is an ordered list of ``(name, width, default)`` for the stage's animatable
    parameters; the stage reads them from ``params[tm, prim, off + slot]`` where
    ``slot`` is that parameter's cumulative offset within the stage (so the first
    param is at ``off + 0``). ``width`` is 1 for a scalar, 3 for an RGB triple.

    ``scatter`` optionally customises how a ray *continues* after this stage's
    pipeline shades a surface hit (reflection / refraction / pass-through): a
    ``@ti.func`` following the scatter contract documented in
    :mod:`algan.rendering.raytracing.shading_taichi`. When no stage of a
    pipeline supplies one, the default scatter applies the classic
    opacity/reflectivity/Fresnel-glass behaviour. When several stages supply
    one, the last stage's scatter wins.

    Both renderers honour it. The deterministic wavefront splits into the
    reflected and transmitted branches the scatter returns; the path tracer
    (``samples_per_pixel > 1``) never splits, so it picks one of the three
    branches at random -- weighted by the branch weights -- and continues
    along it as a delta lobe (weight 1, no MIS coverage, exactly what
    refraction and a tinted pane get there).

    **What ``num_lights`` counts under the path tracer.** A stage's contract is
    a loop ``for li in range(num_lights)`` over the packed light rows, and with
    ``samples_per_pixel == 1`` those are every row in the scene. With
    ``samples_per_pixel > 1`` and a light count past ``max_shadow_lights``
    (``SETTINGS.raytracing.experimental.pt_authored_light_sampling``, ``"auto"``
    by default) the path tracer hands the stage a small SAMPLED SUBSET instead:
    the direction-less rows, then a few rows drawn from the rig, each with the
    weight of the rows it stands for folded into the RGB that ``_light_eval``
    returns. A stage that multiplies by that colour -- every built-in one does,
    in both its reflection and its energy budget -- is therefore an unbiased
    estimator of the sum it would have computed. A stage that uses a light's
    *direction* (or its mere existence) without multiplying by its colour is
    not: it sees an unweighted sum over the sampled rows. Set the switch to
    ``"off"`` if a stage needs every row, at the deterministic renderer's cost
    model and its 16-shadow cap.

    **Colour parameters are linear.** The renderer works in linear light, and
    a stage receives its parameters exactly as they were set: nothing decodes
    a width-3 colour from sRGB the way a Mob's colour or a material's colours
    are decoded. So ``(0.5, 0.5, 0.5)`` is half of white's light, which
    displays as about 0.74, not 0.5. That holds for every stage you write and
    for the looks in :mod:`~algan.rendering.shaders.fragment_stage_library`.
    The built-in material stages (``STAGE_STANDARD`` and the rest) are the
    exception: their ``emissive``, ``specular``, ``specular_color`` and
    ``sheen_color`` are decoded wherever the stage sits in a pipeline, so they
    mean what they mean on the matching material.
    """

    #: Parameter names the renderer decodes from display-referred sRGB to
    #: linear light before the stage reads them. Empty for a custom stage
    #: (see the class docstring); the built-in material stages name their
    #: material colours (``_MAT_COLOR_SLOT_NAMES``).
    _decoded_colors = ()

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


def _builtin_stage(ti_func, param_specs):
    """A built-in material stage, whose material colours are decoded exactly
    as ``set_material`` decodes them (``scene_builder``'s colour boundary), so
    a stage inside a composed pipeline renders the colour it renders alone.
    """
    stage = FragmentStage(ti_func, param_specs)
    names = {name for name, _w, _d in stage.param_specs}
    stage._decoded_colors = tuple(n for n in _MAT_COLOR_SLOT_NAMES if n in names)
    return stage


# Built-in material stages (12-slot canonical params; physical 26). ``manim``
# and ``unlit`` ignore most slots but share the layout so offsets stay uniform
# in a pipeline.
STAGE_MANIM = _builtin_stage(_stage_manim, _BUILTIN_MAT_SPECS)
STAGE_UNLIT = _builtin_stage(_stage_unlit, _BUILTIN_MAT_SPECS)
STAGE_LAMBERT = _builtin_stage(_stage_lambert, _BUILTIN_MAT_SPECS)
STAGE_PHONG = _builtin_stage(_stage_phong, _BUILTIN_MAT_SPECS)
STAGE_STANDARD = _builtin_stage(_stage_standard, _BUILTIN_MAT_SPECS)
STAGE_PHYSICAL = _builtin_stage(_stage_physical, _PHYSICAL_MAT_SPECS)


def _builtin_shader_to_stage():
    """Map the built-in PyTorch material shader functions to their stage ports,
    so ``set_fragment_shader(phong_shader)`` resolves to ``STAGE_PHONG``.
    """
    from algan.rendering.shaders.material_shaders import (
        basic_material_shader,
        lambert_shader,
        manim_shader,
        phong_shader,
        physical_shader,
        standard_shader,
    )
    from algan.rendering.shaders.pbr_shaders import null_shader

    return {
        manim_shader: STAGE_MANIM,
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
# _PIPELINE_COLOR_SLOTS parallels it too: ``((slot, width), ...)`` of the
# parameters its built-in stages declare as colours (see pipeline_color_slots).
_PIPELINE_REGISTRY = {}
_PIPELINE_LIST = []
_PIPELINE_SCATTERS = []
_PIPELINE_COLOR_SLOTS = []


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
    color_slots = []
    for s, base in zip(stages, offsets):
        decoded = getattr(s, "_decoded_colors", ())
        slot = base
        for name, width, _default in s.param_specs:
            if name in decoded:
                color_slots.append((slot, width))
            slot += width
    color_slots = tuple(color_slots)

    # The colour slots are part of the key: the same func at the same width
    # wrapped as a custom stage keeps its parameters linear, so it is not the
    # same pipeline as the built-in stage.
    key = (
        tuple(id(s.ti_func) for s in stages),
        tuple(s.width for s in stages),
        id(scatter) if scatter is not None else None,
        color_slots,
    )
    if key in _PIPELINE_REGISTRY:
        pid = _PIPELINE_REGISTRY[key][0]
    else:
        pid = _USER_PIPELINE_BASE + len(_PIPELINE_LIST)
        composed = make_pipeline_func([s.ti_func for s in stages], offsets)
        _PIPELINE_REGISTRY[key] = (pid, composed)
        _PIPELINE_LIST.append(composed)
        _PIPELINE_SCATTERS.append(scatter)
        _PIPELINE_COLOR_SLOTS.append(color_slots)

    layout = []
    for s, base in zip(stages, offsets):
        slot = base
        for name, width, default in s.param_specs:
            layout.append((name, slot, width, default))
            slot += width
    return pid, total_width, layout


def pipeline_color_slots(pid):
    """``((slot, width), ...)`` of the parameters in pipeline ``pid``'s block
    that are display-referred colours: the material colours of its built-in
    stages, at their offsets in the composed block. Empty for a pipeline of
    custom stages only, and for an id the registry does not hold.

    ``pid`` is a GLOBAL pipeline id (``register_pipeline``'s), not a batch's
    dense one. ``scene_builder._decode_material_block_colors`` decodes these
    slots at the render boundary, as it decodes a built-in material's.
    """
    i = int(pid) - _USER_PIPELINE_BASE
    if 0 <= i < len(_PIPELINE_COLOR_SLOTS):
        return _PIPELINE_COLOR_SLOTS[i]
    return ()


def _select_by_pid(entries, pids):
    """``entries`` (a registry list indexed by ``pid - _USER_PIPELINE_BASE``)
    gathered into the slot order ``pids`` asks for, or the whole list when
    ``pids`` is None.

    ``pids`` is the batch's GLOBAL pipeline ids **in the order the batch packs
    them**, so slot ``i`` of the result is the pipeline the batch packed as
    ``_USER_PIPELINE_BASE + i``. The scene merge renumbers a batch's ids to
    exactly that dense range (``scene_builder._densify_frag_pipeline_ids``),
    which is what makes the result's shape a function of the batch alone
    rather than of how many pipelines the process happened to register first
    -- and Taichi specialises the shade kernels on this tuple, so that
    difference is the difference between reusing a compiled kernel and
    compiling a new one.

    ``None`` means "the whole registry, indexed by global id": the fallback
    for a merged scene that carries no dense id list, whose packed ids are
    therefore still global.
    """
    if pids is None:
        return tuple(entries)
    return tuple(entries[int(pid) - _USER_PIPELINE_BASE] for pid in pids)


def build_frag_pipelines(pids=None):
    """Composed pipeline funcs to inject as the shade kernel's
    ``frag_pipelines`` template argument, indexed by ``pid -
    _USER_PIPELINE_BASE`` and ordered by id.

    ``pids`` is the batch's own pipeline ids in packing order; everything else
    is dropped (see :func:`_select_by_pid`). Pass it. **The registry is
    process-global and append-only, and Taichi specialises the shade kernels
    on this tuple**, so handing over the whole registry puts every render in a
    process that ever registered a pipeline onto its own uncached kernel
    variant -- including renders with no custom shader at all, which is both
    the pathology this argument exists to close and the reason a batch-narrowed
    tuple is what the tracer passes. ``None`` (the whole registry) is the
    conservative fallback for a batch whose merged scene cannot enumerate its
    ids.
    """
    return _select_by_pid(_PIPELINE_LIST, pids)


def build_frag_scatters(pids=None):
    """Per-pipeline custom scatter funcs (None = default scatter), ordered by
    id, for the monolithic wavefront's per-material continuation dispatch --
    and for the path tracer's, which takes the same tuple and continues along
    one sampled branch of it (``pt_shade``).

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
    runs it leaves the color raw for the in-kernel pipeline to shade.
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
    cam_pos,
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


#: Example custom fragment stage: recolors each fragment with a cosine wave.
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
    reflectivity: commit 15% of the shaded color and bounce the remaining
    throughput along the mirror direction (no transmission). Branch weights
    are vec3 per-channel throughput multipliers (color transport); this
    scatter reflects achromatically, so all three channels match.

    The bounce is set up the way the built-in scatter sets up its own:
    :func:`~algan.rendering.raytracing.shading_taichi._reflect_frame` picks the
    mirror direction and the side to leave from (so a smooth-shaded silhouette
    cannot launch its reflection into the solid), and
    :func:`~algan.rendering.raytracing.transport_taichi._offset_ray_origin`
    lifts the origin by a scale-aware epsilon rather than a fixed distance.

    Coverage is honoured: the ``1 - alpha`` that misses the surface continues
    through it. The scatter contract uses ``pass_w`` only when ``refl_w`` is
    zero, so where a partly covering surface has both continuations the
    reflection takes the split slot (``trans_*``) and the pass-through stays
    the primary ray -- what the built-in scatter does for a semi-transparent
    reflector. Without the split pool (``refraction == 0``) only the heavier
    of the two is traced.
    """
    refl_dir, offset_n = _reflect_frame(rd, n_interp.normalized(), face_n)
    refl_orig = _offset_ray_origin(hit_point, offset_n)
    contrib = (alpha * 0.15) * shaded
    zero3 = ti.math.vec3(0.0, 0.0, 0.0)
    rw = 0.85 * alpha
    if bounces_left <= 0:  # out of bounces: absorb instead of reflecting
        rw = 0.0
    through = 1.0 - alpha  # the share that misses the surface
    refl_w = ti.math.vec3(rw, rw, rw)
    pass_w = ti.math.vec3(through, through, through)
    trans_orig = zero3
    trans_dir = zero3
    trans_w = zero3
    if through > min_alpha:
        if ti.static(refraction != 0):
            trans_orig = refl_orig
            trans_dir = refl_dir
            trans_w = refl_w
            refl_w = zero3
        elif through > rw:
            refl_w = zero3
    return (
        contrib,
        pass_w,
        refl_orig,
        refl_dir,
        refl_w,
        trans_orig,
        trans_dir,
        trans_w,
    )


#: Example custom scatter: forces mirror bouncing regardless of the mob's
#: reflectivity. Attach to a stage, e.g.
#: ``mob.set_fragment_shader(FragmentStage(_stage, specs, scatter=...))`` or
#: compose with a built-in stage: ``FragmentStage(STAGE_PHONG.ti_func,
#: STAGE_PHONG.param_specs, scatter=forced_mirror_scatter.scatter)``.
forced_mirror_scatter = FragmentStage(
    _stage_unlit, _BUILTIN_MAT_SPECS, scatter=_scatter_forced_mirror
)
