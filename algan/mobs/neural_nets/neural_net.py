from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import STRUCTURE_VERSION, attr_ranges_for_mob
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Lag, Off, Seq, Sync
from algan.constants.rate_funcs import delay_fade, identity, pulse_fade
from algan.constants.spatial import *  # ORIGIN, OUTWARD, RIGHT
from algan.environment import env_flag
from algan.geometry.geometry import (
    get_orthonormal_vector,
    map_global_to_local_coords,
    map_local_to_global_coords,
)
from algan.mobs.shapes_3d import Cylinder, Sphere
from algan.mobs.text import Tex
from algan.rendering.shaders.materials import (
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)
from algan.settings._startup import _ANIMATION_DEVICE
from algan.utils.tensor_utils import dot_product, squish, unsquish

# Synapses jitter their colour for visual variety. Draw that jitter from a
# dedicated, fixed-seed generator (reseeded per net in NeuralNetMLP.__init__)
# rather than the global RNG: otherwise every render produced different synapse
# colours, so the same scene rendered twice differed by tens of code values --
# which reads as nondeterministic ("order-sensitive") output and makes the
# frame-comparison tests impossible to satisfy. A private generator keeps the
# synapse-to-synapse variety while making each render byte-reproducible, and
# leaves the global torch RNG untouched for everything else.
COLOR_JITTER_SEED = 0xA76A
_color_rng = torch.Generator(device=_ANIMATION_DEVICE).manual_seed(COLOR_JITTER_SEED)

# The idle walk is authored as a loop of deterministic random waypoints. An
# updater must be a pure function of elapsed time because Algan may materialize
# frame windows out of playback order; drawing a fresh random increment per
# invocation would make the path depend on batching and render order.
_IDLE_WALK_SEED = 0x1D1E
_IDLE_WAYPOINT_COUNT = 16
_IDLE_SECONDS_PER_WAYPOINT = 8
_IDLE_DESIRED_RADIUS_PER_SPACING = 0.5
_IDLE_CLEARANCE_RADIUS_FRACTION = 0.5
_IDLE_PARALLEL_RADIUS_FRACTION = 0.2
_idle_rng = torch.Generator(device=_ANIMATION_DEVICE).manual_seed(_IDLE_WALK_SEED)


def _single_location(mob):
    """Return the one world-space center represented by a network component."""
    return mob.location.reshape(-1, 3)[0]


def _neuron_collision_radius(neuron):
    """Radius of a sphere enclosing the neuron's visible core and shell."""
    center = _single_location(neuron)
    bounds = []
    for component in (neuron.core, neuron.shell):
        component_center = _single_location(component)
        scale = component.scale_coefficient.reshape(-1, 3).amax()
        base_radius = torch.as_tensor(
            component.radius,
            device=component_center.device,
            dtype=component_center.dtype,
        )
        bounds.append((component_center - center).norm() + base_radius * scale)
    return torch.stack(bounds).amax()


def _idle_radii_for_layers(layers, neuron_spacing):
    """Return per-neuron walk and collision radii in layer-flattened order.

    If two original centers are distance ``d`` apart and their enclosing
    radii are ``a`` and ``b``, giving every neuron in the layer a walk radius
    below ``(d - a - b) / 2`` makes overlap impossible for every pair of
    points in their walk spheres. The 0.45 multiplier leaves a strict margin.
    """
    first_neuron = next(neuron for layer in layers for neuron in layer)
    reference = _single_location(first_neuron)
    desired_radius = (
        torch.as_tensor(
            neuron_spacing, device=reference.device, dtype=reference.dtype
        ).abs()
        * _IDLE_DESIRED_RADIUS_PER_SPACING
    )
    walk_radii = []
    collision_radii = []
    for layer in layers:
        centers = torch.stack([_single_location(neuron) for neuron in layer])
        radii = torch.stack([_neuron_collision_radius(neuron) for neuron in layer])
        layer_radius = desired_radius
        if len(layer) > 1:
            distances = torch.cdist(centers, centers)
            clearances = distances - radii[:, None] - radii[None, :]
            pair_mask = torch.triu(
                torch.ones_like(clearances, dtype=torch.bool), diagonal=1
            )
            minimum_clearance = clearances[pair_mask].amin().clamp_min(0)
            layer_radius = torch.minimum(
                layer_radius,
                minimum_clearance * _IDLE_CLEARANCE_RADIUS_FRACTION,
            )
        walk_radii.append(layer_radius.expand(len(layer)))
        collision_radii.append(radii)
    return torch.cat(walk_radii), torch.cat(collision_radii)


def _make_idle_waypoints(walk_radii, direction, *, dtype, device):
    """Sample deterministic points in ellipsoids flattened along ``direction``."""
    _idle_rng.manual_seed(_IDLE_WALK_SEED)
    shape = (walk_radii.numel(), _IDLE_WAYPOINT_COUNT - 1, 3)
    directions = torch.randn(shape, dtype=dtype, device=device, generator=_idle_rng)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radial_scale = torch.rand(
        (*shape[:-1], 1), dtype=dtype, device=device, generator=_idle_rng
    ).pow(1 / 3)
    random_points = directions * radial_scale
    network_direction = torch.as_tensor(direction, dtype=dtype, device=device)
    network_direction = network_direction / network_direction.norm().clamp_min(1e-8)
    parallel_components = (random_points * network_direction).sum(
        dim=-1, keepdim=True
    ) * network_direction
    random_points = (
        random_points
        - parallel_components
        + parallel_components * _IDLE_PARALLEL_RADIUS_FRACTION
    )
    unit_waypoints = torch.cat(
        [torch.zeros((shape[0], 1, 3), dtype=dtype, device=device), random_points],
        dim=1,
    )
    return unit_waypoints * walk_radii.view(-1, 1, 1)


def _interpolate_idle_waypoints(time_elapsed, waypoints):
    """Evaluate the looping, smooth random-waypoint walk at arbitrary times."""
    elapsed = time_elapsed.reshape(-1)
    progress = elapsed / _IDLE_SECONDS_PER_WAYPOINT
    segment = torch.floor(progress).long()
    alpha = (progress - segment).view(-1, 1, 1)
    # Smoothstep gives every random waypoint zero arrival/departure velocity,
    # avoiding a visible direction snap while retaining a convex interpolation.
    alpha = alpha * alpha * (3 - 2 * alpha)
    current_index = segment.remainder(waypoints.shape[1])
    next_index = (current_index + 1).remainder(waypoints.shape[1])
    current = waypoints[:, current_index].permute(1, 0, 2)
    following = waypoints[:, next_index].permute(1, 0, 2)
    return torch.lerp(current, following, alpha)


class _IdleBatchUnsupported(Exception):
    """The net's structure is not one the batched idle path can replicate."""


class _IdleBatchPlan:
    """Static row map and geometry constants of one net's idle updater.

    Everything the batched path needs that does not change between frames:
    which buffer rows each idle neuron, synapse, tube grid and cap owns, the
    per-synapse endpoint wiring of the four loops, and the shared (u, v)
    grids. Rebuilt whenever the global structure version moves.
    """

    __slots__ = (
        "neurons",
        "neuron_own",
        "subtree_rows",
        "subtree_seg",
        "syn_loc",
        "syn_basis",
        "grid_rows",
        "syn_owner",
        "start_src",
        "end_src",
        "variant",
        "synapses",
        "tube_uv",
        "radius",
        "height",
        "v_range0",
        "v_range1",
        "trace_mobs",
    )


def _idle_batch_plan(net, neurons):
    """Build (or return the cached) :class:`_IdleBatchPlan` for ``net``."""
    version = STRUCTURE_VERSION[0]
    cache = getattr(net, "_idle_batch_plan_cache", None)
    if cache is not None and cache[0] == version and cache[1].neurons == neurons:
        return cache[1]
    plan = _build_idle_batch_plan(net, neurons)
    object.__setattr__(net, "_idle_batch_plan_cache", (version, plan))
    return plan


def _rows_of(attr_timeline, mob):
    return attr_ranges_for_mob(attr_timeline, mob).tensor()


def _same(a, b):
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)
    return a_t.shape == b_t.shape and torch.equal(a_t, b_t)


def _build_idle_batch_plan(net, neurons):
    tl = net.scene.timeline_manager
    loc_tl = tl.attr_to_timeline.get("location")
    bas_tl = tl.attr_to_timeline.get("basis")
    if loc_tl is None or bas_tl is None:
        raise _IdleBatchUnsupported

    neuron_index = {}
    for index, neuron in enumerate(neurons):
        neuron_index[id(neuron)] = index

    def global_index(mob):
        got = neuron_index.get(id(mob))
        if got is None:
            raise _IdleBatchUnsupported
        return got

    # The four loops' endpoint wiring, in loop order: (synapse, start source,
    # end source), where a source is a global neuron index or -1 for "the
    # synapse's own current end". Loop 2 replaces only ends; loop 4 only
    # starts; loop 3 replaces both; loop 2/4 use the basis-row offset, loop 3
    # the normalized-direction-times-scale one.
    wiring = []
    for i, n in enumerate(net.layers[0]):
        for synapse in n.synapses:
            wiring.append((synapse, -1, i, 0))
    for previous_layer, layer in zip(net.layers[:-1], net.layers[1:-1]):
        for neuron in layer:
            j = global_index(neuron)
            for source, synapse in zip(previous_layer, neuron.synapses):
                wiring.append((synapse, global_index(source), j, 1))
    for neuron in net.layers[-1]:
        for source, synapse in zip(net.layers[-2], neuron.synapses):
            wiring.append((synapse, global_index(source), -1, 0))
    if not wiring:
        raise _IdleBatchUnsupported

    plan = _IdleBatchPlan()
    plan.neurons = neurons
    plan.neuron_own = torch.cat([_rows_of(loc_tl, n) for n in neurons])

    rows_parts = []
    seg_parts = []
    trace_mobs = []
    for index, neuron in enumerate(neurons):
        trace_mobs.append(neuron)
        count = 0
        for mob in neuron.get_descendants(include_self=True):
            if mob is not neuron and "location" in getattr(
                mob, "_excluded_from_parent_attrs", ()
            ):
                continue
            rows_parts.append(_rows_of(loc_tl, mob))
            count += int(rows_parts[-1].numel())
        seg_parts.append(torch.full((count,), index, dtype=torch.long))
    plan.subtree_rows = torch.cat(rows_parts)
    plan.subtree_seg = torch.cat(seg_parts)
    plan.trace_mobs = trace_mobs

    synapses = [w[0] for w in wiring]
    plan.synapses = synapses
    plan.start_src = torch.tensor([w[1] for w in wiring], dtype=torch.long)
    plan.end_src = torch.tensor([w[2] for w in wiring], dtype=torch.long)
    plan.variant = torch.tensor([w[3] for w in wiring], dtype=torch.bool)
    # A synapse hangs under the neuron that owns it, whose subtree loop 1's
    # recursive move shifts. Every owner outside the idle set is an
    # output-layer neuron (loop 4), and nothing ever moves those: they get a
    # zero change, handled through the padded column below.
    owner_by_syn = {}
    for layer in net.layers:
        for n2 in layer:
            for s in n2.synapses:
                owner_by_syn[id(s)] = neuron_index.get(id(n2), -1)
    if any(id(w[0]) not in owner_by_syn for w in wiring):
        raise _IdleBatchUnsupported
    owners = [owner_by_syn[id(w[0])] for w in wiring]
    plan.syn_owner = torch.tensor(owners, dtype=torch.long)

    plan.syn_loc = torch.cat([_rows_of(loc_tl, s) for s in synapses])
    plan.syn_basis = torch.cat([_rows_of(bas_tl, s) for s in synapses])
    plan.grid_rows = torch.cat([_rows_of(loc_tl, s.grid) for s in synapses])

    # These synapses are open tubes (no end discs); a capped cylinder would
    # also rewrite its caps in ``_move_between_points`` via ``_place_bases``.
    if any(getattr(s, "bottom_cap", None) is not None for s in synapses):
        raise _IdleBatchUnsupported

    first = synapses[0]
    tube_uv = squish(first.get_base_grid(), -3, -2).unsqueeze(0)
    for s in synapses:
        uv = squish(s.get_base_grid(), -3, -2).unsqueeze(0)
        if uv.shape != tube_uv.shape or not torch.equal(uv, tube_uv):
            raise _IdleBatchUnsupported
    plan.tube_uv = tube_uv

    radius = first.radius
    height = first.height
    v_range0 = first.v_range[0]
    v_range1 = first.v_range[1]
    for s in synapses:
        if (
            type(s.radius) is not type(radius)
            or s.radius != radius
            or s.height != height
            or not _same(s.v_range[0], v_range0)
            or not _same(s.v_range[1], v_range1)
        ):
            raise _IdleBatchUnsupported
    plan.radius = radius
    plan.height = height
    plan.v_range0 = v_range0
    plan.v_range1 = v_range1
    return plan


def _cylinder_coord_offsets(uv, radius, height, v_range0, v_range1, right, up, fwd):
    """``Cylinder.coord_function`` verbatim, batched over stacked tubes.

    ``right``/``up``/``fwd`` are the tube's (scaled) basis rows with shape
    ``[B, S, 1, 3]``; the result is offsets from each tube's centre.
    """
    uv = uv.clone()
    uv[..., 1:] /= uv[..., 1:].amax()
    u = -(v_range0 + uv[..., :1] * (v_range1 - v_range0))
    v = uv[..., 1:]
    return u.sin() * radius * right + (v - 0.5) * height * up + u.cos() * radius * fwd


def _update_idle_loops_batched(net, scalar_time, neurons, world_positions):
    """The four per-mob loops of :func:`_update_neural_net_idle`, batched.

    Writes exactly what the per-mob loops write -- same expressions evaluated
    over all synapses at once, same per-row arithmetic in the same order --
    into three timeline writes instead of hundreds of per-mob reads and
    writes. Every read and every computation happens before any write, so an
    unsupported structure can fall back to the loops without having touched
    state.

    Per-loop semantics replicated bit for bit:

    * neuron ``move_to``: a recursive add of ``(target - own location)`` to
      every location row of the neuron's subtree;
    * ``set_end_point`` / ``set_start_point``: offset from the raw basis row;
      the interpolated endpoint passes through unchanged at interpolation 1;
    * ``move_between_points``: offset from the normalized direction times the
      scale coefficient;
    * the common tail writes location (midpoint, via the setter's
      change-then-add on every shifted row), basis directly, then re-evaluates
      each tube's coordinate function against its new basis and midpoint,
      landing through the same setter arithmetic.

    These synapses are open tubes: ``_move_between_points``'s
    ``_place_bases`` step only runs on a capped cylinder, so there is nothing
    else to replicate (and a capped synapse raises
    :class:`_IdleBatchUnsupported` at plan time).
    """
    plan = _idle_batch_plan(net, neurons)
    tl = net.scene.timeline_manager
    loc_tl = tl.attr_to_timeline["location"]
    bas_tl = tl.attr_to_timeline["basis"]

    targets = world_positions.unsqueeze(0) if scalar_time else world_positions

    for mob in plan.trace_mobs:
        tl.trace_updater_mob_access(mob, True)

    # ---- reads (all before any write) ------------------------------------
    own_cur = loc_tl.get(plan.neuron_own, copy=False)  # [B, N, 3]
    sub_cur = loc_tl.get(plan.subtree_rows, copy=False)  # [B, R, 3]
    syn_loc_pre = loc_tl.get(plan.syn_loc, copy=False)  # [B, S, 3]
    syn_bas = bas_tl.get(plan.syn_basis, copy=False)  # [B, S, 9]
    grid_pre = loc_tl.get(plan.grid_rows, copy=False)  # [B, S*Gv, 3]

    batch = own_cur.shape[0]
    n_neurons = len(neurons)
    n_syn = len(plan.synapses)

    # ---- loop 1: recursive neuron moves ----------------------------------
    changes = targets.to(own_cur.device) - own_cur  # [B, N, 3]
    neu_loc = own_cur + changes  # the value write 1 lands on the own row
    new_sub = sub_cur + changes.index_select(1, plan.subtree_seg)
    # Synapses under a moved neuron inherit its change; synapses under an
    # output-layer neuron (never moved) take the padded zero column.
    padded_changes = torch.cat((changes, changes.new_zeros((batch, 1, 3))), dim=1)
    owner_gather = torch.where(
        plan.syn_owner >= 0,
        plan.syn_owner,
        torch.full_like(plan.syn_owner, n_neurons),
    )
    syn_change = padded_changes.index_select(1, owner_gather)  # [B, S, 3]
    syn_loc = syn_loc_pre + syn_change
    g_view_shape = (batch, n_syn, -1, 3)
    grid_loc = grid_pre.view(*g_view_shape) + syn_change.unsqueeze(2)

    # ---- loops 2-4: per-synapse endpoints ---------------------------------
    unsq = unsquish(syn_bas, -1, 3)
    up_row = unsq[..., 1, :]
    scale = unsq.norm(p=2, dim=-1, keepdim=False)
    off_a = up_row * 0.5
    off_b = (F.normalize(up_row, p=2, dim=-1) * scale[..., 1].unsqueeze(-1)) * 0.5
    variant = plan.variant.view(1, n_syn, 1).to(off_a.device)
    offset = torch.where(variant, off_b, off_a)
    current_end = syn_loc + offset
    current_start = syn_loc - offset

    start_arg = neu_loc.index_select(1, plan.start_src.clamp(min=0))
    end_arg = neu_loc.index_select(1, plan.end_src.clamp(min=0))
    has_start = (plan.start_src >= 0).view(1, n_syn, 1).to(start_arg.device)
    has_end = (plan.end_src >= 0).view(1, n_syn, 1).to(end_arg.device)
    interpolation = 1.0
    interp_start = current_start * (1 - interpolation) + interpolation * start_arg
    interp_end = current_end * (1 - interpolation) + interpolation * end_arg
    start_ = torch.where(has_start, interp_start, current_start)
    end_ = torch.where(has_end, interp_end, current_end)

    # ---- common tail of Cylinder._move_between_points ---------------------
    sep = end_ - start_
    up_b = F.normalize(sep, p=2, dim=-1)
    right_b = get_orthonormal_vector(up_b)
    forward_b = torch.cross(right_b, up_b, dim=-1)
    mid = (start_ + end_) * 0.5
    d_mid = mid - syn_loc
    new_syn_loc = syn_loc + d_mid
    new_basis = torch.cat(
        (right_b * scale[..., :1], sep, forward_b * scale[..., 2:]), -1
    )

    d_mid_g = d_mid.unsqueeze(2)  # [B, S, 1, 3]
    nb = unsquish(new_basis, -1, 3)
    rb = nb[..., 0, :].unsqueeze(2)
    ub = nb[..., 1, :].unsqueeze(2)
    fb = nb[..., 2, :].unsqueeze(2)

    tube_offs = _cylinder_coord_offsets(
        plan.tube_uv,
        plan.radius,
        plan.height,
        plan.v_range0,
        plan.v_range1,
        rb,
        ub,
        fb,
    )
    grid_target = tube_offs + new_syn_loc.unsqueeze(2)
    grid_shifted = grid_loc + d_mid_g
    grid_new = grid_shifted + (grid_target - grid_shifted)

    # ---- plain (non-timeline) attributes ----------------------------------
    for j, synapse in enumerate(plan.synapses):
        synapse.direction = up_b[:, j : j + 1]
        synapse.coord_function_active = synapse.coord_function

    # ---- writes (all after every read and every computation) --------------
    fused_rows = torch.cat((plan.syn_loc, plan.grid_rows))
    fused_vals = torch.cat(
        (
            new_syn_loc,
            grid_new.reshape(batch, -1, 3),
        ),
        dim=1,
    )
    tl.capture_updater_write("location", plan.subtree_rows, new_sub)
    loc_tl.modify(plan.subtree_rows, new_sub)
    tl.capture_updater_write("location", fused_rows, fused_vals)
    loc_tl.modify(fused_rows, fused_vals)
    tl.capture_updater_write("basis", plan.syn_basis, new_basis)
    bas_tl.modify(plan.syn_basis, new_basis)


def _update_neural_net_idle(net, time_elapsed, local_origins, waypoints):
    """Updater for bounded neuron drift and synapses that follow their ends."""
    scalar_time = time_elapsed.ndim == 0
    local_positions = local_origins.unsqueeze(0) + _interpolate_idle_waypoints(
        time_elapsed, waypoints
    )
    if scalar_time:
        local_positions = local_positions[0]
        net_location = net.location
        net_basis = net.basis
    else:
        frame_count = time_elapsed.numel()
        net_location = net.location.reshape(frame_count, -1, 3)
        net_basis = net.basis.reshape(frame_count, -1, 9)
    world_positions = map_local_to_global_coords(
        net_location, net_basis, local_positions
    )

    neurons = [neuron for layer in net.layers[:-1] for neuron in layer]
    if scalar_time:
        world_positions = world_positions.reshape(-1, len(neurons), 3)[0]
    else:
        world_positions = world_positions.reshape(frame_count, len(neurons), 3)

    if env_flag("ALGAN_BATCHED_IDLE_UPDATER", True):
        # Default on: proved bit-identical to the loops below on this scene's
        # attribute buffers (all timelines plus the non-timeline directions)
        # across frame windows and layer sizes -- see
        # scratch_perf/r2/parity_idle_updater.py. Set 0 to restore the
        # per-mob loops.
        try:
            _update_idle_loops_batched(net, scalar_time, neurons, world_positions)
            return net
        except _IdleBatchUnsupported:
            # Unsupported structure: nothing has been written (every write
            # happens after all reads and arithmetic), so fall through to the
            # per-mob loops.
            pass

    for index, neuron in enumerate(neurons):
        target = (
            world_positions[index]
            if scalar_time
            else world_positions[:, index : index + 1]
        )
        neuron.move_to(target)

    for neuron in net.layers[0]:
        for synapse in neuron.synapses:
            synapse.set_end_point(neuron.location)

    for previous_layer, layer in zip(net.layers[:-1], net.layers[1:-1]):
        for neuron in layer:
            for source, synapse in zip(previous_layer, neuron.synapses):
                synapse.move_between_points(source.location, neuron.location)

    for neuron in net.layers[-1]:
        for source, synapse in zip(net.layers[-2], neuron.synapses):
            synapse.set_start_point(source.location)
    return net


def tweak_color(c, strength=0.2, min_strength=0.0):
    t = torch.rand((1,), generator=_color_rng).item()
    t = t * strength + (1 - t) * min_strength
    # m = torch.randint(0, 2, (1,), generator=_color_rng)
    # target_c = WHITE * m + (1 - m) * BLACK
    target_c = c.set_rgb(
        torch.rand(
            c.rgb.shape, device=c.rgb.device, dtype=c.rgb.dtype, generator=_color_rng
        )
    )
    return c * (1 - t) + t * target_c


gs = 0.75


class Synapse(Cylinder):
    def __init__(self, grid_height=5, *args, **kwargs):
        # grid_height = 20  # None
        # grid_width = 12
        grid_height = None
        grid_width = None
        if "color" in kwargs:
            c = kwargs["color"]
            kwargs["color"] = tweak_color(c, strength=0.25, min_strength=0.25)
        super().__init__(grid_height=grid_height, grid_width=grid_width, **kwargs)
        self.scale(0.02)


class Neuron(Mob):
    synapse_cls = Synapse

    def __init__(self, input_locs, direction, neuron_color, **kwargs):
        super().__init__(**kwargs)
        grid_height = None
        self.core = self._make_core(grid_height, neuron_color).move_to(self.location)
        self.shell = (
            self._make_shell(grid_height, neuron_color)
            .move_to(self.location)
            .look(direction, axis=1)
        )
        self.synapses = [
            self.synapse_cls(
                grid_height, scene=self.scene, color=neuron_color
            ).move_between_points(input_location, self.location)
            for input_location in input_locs
        ]
        self.add_children(self.core, self.shell, self.synapses)

    def _make_core(self, grid_height, neuron_color):
        return Sphere(
            scene=self.scene,
            grid_height=grid_height,
            grid_width=grid_height,
            color=neuron_color,
        ).scale(0.17)

    def _make_shell(self, grid_height, neuron_color):
        return (
            Sphere(
                scene=self.scene,
                opacity=0.5,
                grid_width=grid_height,
                grid_height=grid_height,
                color=neuron_color,
            )
            .set_shader(None)
            .scale(0.2)
        )


# class Layer(Mob):
#    def __init__(self, input_locs, neuron_locs, **kwargs):
#        super().__init__(**kwargs)
#        self.neurons = [Neuron(input_locs, location=l) for l in neuron_locs]
#        self.add_children(self.neurons)


# The V2/V3 fill light comes from envMapIntensity (ambient = albedo * 0.1 * env)
# rather than emissive, so it tracks the albedo during wave_color pulses instead
# of tinting them with the resting colour. Metalness stays 0 (dielectric): the
# glossy look comes from low roughness, and -- in V3 -- clearcoat and sheen.


class SynapseV2(Cylinder):
    """Improved synapse: a thin filament lit from within (emissive) with a
    glossy dielectric surface, so pulses read as light travelling down a wire.
    """

    def __init__(self, grid_height=5, *args, **kwargs):
        grid_height = 20
        grid_width = 12
        c = kwargs.get("color")
        if c is not None:
            c = tweak_color(c, strength=0.25, min_strength=0.25)
            kwargs["color"] = c
        else:
            c = WHITE
        super().__init__(grid_height=grid_height, grid_width=grid_width, **kwargs)
        # Fill light comes from env_map_intensity (ambient = albedo * 0.1 * env)
        # rather than emissive, so it tracks the albedo during colour-wave
        # pulses instead of tinting them with the resting colour.
        self.set_material(
            MeshStandardMaterial(
                color=c.set_glow(0.04),
                roughness=0.3,
                metalness=0.0,
                envMapIntensity=4.5,
            )
        )
        self.scale(0.02)


class NeuronV2(Neuron):
    """Improved neuron: a glossy self-lit core (crisp specular highlight over
    an emissive base) inside a soft translucent halo shell, replacing the
    flat-shaded spheres of :class:`Neuron`.
    """

    synapse_cls = SynapseV2

    def _make_core(self, grid_height, neuron_color):
        material = MeshStandardMaterial(
            color=neuron_color.set_glow(0.08),
            roughness=0.2,
            metalness=0.0,
            envMapIntensity=4.0,
        )
        return (
            Sphere(
                scene=self.scene,
                grid_height=grid_height,
                grid_width=grid_height,
                color=neuron_color,
            )
            .set_material(material)
            .scale(0.17)
        )

    def _make_shell(self, grid_height, neuron_color):
        material = MeshStandardMaterial(
            color=neuron_color.set_opacity(0.3),
            roughness=0.4,
            metalness=0.0,
            envMapIntensity=2.5,
        )
        return (
            Sphere(
                scene=self.scene,
                opacity=0.3,
                grid_width=grid_height,
                grid_height=grid_height,
                color=neuron_color,
            )
            .set_material(material)
            .scale(0.21)
        )


k = 1


def zap(mob1, mob2, color=BLUE, direction=UP, num_points=3):
    with Off(animation_manager=mob1.animation_manager):
        p1 = mob1.get_points_evenly_along_direction(direction)
        p2 = mob2.get_points_evenly_along_direction(direction)
        syns = [
            Synapse(scene=mob1.scene).move_between_points(p1[i], p2[i])
            for i in range(num_points)
        ]
        for s in syns:
            for _ in s.get_descendants():
                if not _.is_primitive:
                    continue
                _.color = _.color.set_opacity(0)
            s.spawn(animate=False)
    with Sync(run_time=1, animation_manager=mob1.animation_manager):
        for s in syns:
            s.wave_color(
                color + GLOW,
                direction=s.get_upwards_direction(),
                opacity=1,
                wave_length=1.5,
            )
    with Off(animation_manager=mob1.animation_manager):
        [s.despawn(animate=False) for s in syns]
    return


class NeuralNetMLP(Mob):
    neuron_cls = Neuron

    def __init__(
        self,
        dims,
        direction=RIGHT,
        orth_direction=UP,
        layer_spacing=1,
        neuron_spacing=0.5,
        input_locs=None,
        neuron_color=GREEN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Reseed the synapse colour jitter so each net build is reproducible
        # regardless of how many nets (or other RNG users) preceded it.
        _color_rng.manual_seed(COLOR_JITTER_SEED)
        self.look(direction)
        start = ORIGIN if input_locs is None else sum(input_locs) / len(input_locs)

        def proj(x):
            return x - dot_product(x, direction) * direction

        def rng(size):
            return torch.rand(size) * 2 - 1

        # orth_direction = get_orthonormal_vector(direction)
        # neuron_locs = [start + proj(rng((d,3))*0.4) + direction * rng((1,)) * 0.1 + direction*(i)*layer_spacing for i, d in enumerate(dims)]
        neuron_locs = [
            start
            + (torch.arange(d).unsqueeze(-1) - (d // 2))
            * neuron_spacing
            * orth_direction
            + direction * (i) * layer_spacing
            for i, d in enumerate(dims)
        ]
        if input_locs is not None:
            neuron_locs[0] = input_locs
        self.input_synapse_offset = -layer_spacing * 0.5
        with Off(animation_manager=self.animation_manager):
            self.layers = [
                [
                    self.neuron_cls(
                        [location + direction * self.input_synapse_offset],
                        direction,
                        location=location,
                        neuron_color=neuron_color,
                        scene=self.scene,
                    )
                    for location in neuron_locs[0]
                ]
            ] + [
                [
                    self.neuron_cls(
                        neuron_locs[i],
                        direction,
                        location=location,
                        neuron_color=neuron_color,
                        scene=self.scene,
                    )
                    for location in neuron_locs[i + 1]
                ]
                for i in range(len(neuron_locs) - 1)
            ]
        # self.layers = [[Neuron(neuron_locs[i], location=l) for l in neuron_locs[i+1]] for i in range(len(neuron_locs)-1)]

        self.add_children(self.layers)
        self._idle_neurons = [neuron for layer in self.layers[:-1] for neuron in layer]
        original_locations = torch.stack(
            [_single_location(neuron) for neuron in self._idle_neurons]
        )
        self._idle_origins_local = map_global_to_local_coords(
            self.location, self.basis, original_locations
        )
        self._idle_walk_radii, self._idle_collision_radii = _idle_radii_for_layers(
            self.layers[:-1], neuron_spacing
        )
        local_network_direction = map_global_to_local_coords(
            self.location, self.basis, self.location + direction
        )
        self._idle_waypoints = _make_idle_waypoints(
            self._idle_walk_radii,
            local_network_direction,
            dtype=original_locations.dtype,
            device=original_locations.device,
        )
        self.idle_updater_id = self.add_updater(
            _update_neural_net_idle,
            self._idle_origins_local,
            self._idle_waypoints,
        )

    def increment_weight(self):
        weight = self.layers[1][0].synapses[0]
        weight.orig_color = weight.color
        with Seq(animation_manager=self.animation_manager):
            weight.color = weight.color + GLOW * 0.2
            self.increment_label = (
                Tex("w := w + 0.001", scene=self.scene)
                .move_next_to(weight, LEFT + DOWN, buffer=0.05)
                .spawn()
            )
        self.incremented_weight = weight
        return weight

    def unincrement_weight(self):
        with Sync(animation_manager=self.animation_manager):
            self.increment_label.despawn()
            self.incremented_weight.color = self.incremented_weight.orig_color

    def train_step(
        self,
        input_values,
        output_generator,
        label,
        run_time=3,
        forward_color=PURE_RED * k + (1 - k) * WHITE,
        backward_color=PURE_BLUE * k + (1 - k) * WHITE,
    ):
        with Seq():
            o = self.forward(
                input_values,
                output_generator,
                run_time,
                reset=False,
                color=forward_color,
            )  # .get_component_mobs())
            # o.move_next_to(label, -self.get_right_direction())
            self.backward(o, label, color=backward_color, run_time=run_time)
            o.despawn()
        return self

    def forward(self, inputs, output_generator=None, run_time=3, reset=True, **kwargs):
        if isinstance(inputs, Mob):
            inputs = [
                [_]
                for _ in inputs.get_points_evenly_along_direction(
                    -(self.get_forward_direction() + self.get_upwards_direction()),
                    len(self.layers[0]),
                )
            ]
        else:
            inputs = [[_.location for _ in inputs] for _ in range(len(self.layers[0]))]
        with Seq(run_time=run_time, animation_manager=self.animation_manager):
            with Sync(run_time=1, animation_manager=self.animation_manager):
                for neuron, neuron_inputs in zip(self.layers[0], inputs):
                    for syn, inp in zip(neuron.synapses, neuron_inputs):
                        syn.set_start_point(inp)  # , n.location)
            out = self.activate(
                run_time=run_time, output_generator=output_generator, **kwargs
            )
            with Sync(animation_manager=self.animation_manager):
                if reset:
                    self.reset_input_synapses()
                out.move(self.get_forward_direction() * 0.25)
            return out

    def reset_input_synapses(self):
        with Sync(run_time=1, animation_manager=self.animation_manager):
            for n in self.layers[0]:
                for syn in n.synapses:
                    syn.move_between_points(
                        n.location
                        + self.get_forward_direction() * self.input_synapse_offset,
                        n.location,
                    )  # , n.location)
            """with Off():
                for n in self.layers[0]:
                    for syn in n.synapses:
                        syn.location = n.location + self.get_forward_direction() * self.input_synapse_offset * 0.5
                        syn.basis = torch.cat([-self.get_upwards_direction() * syn.scale_coefficient[...,:1],
                                               self.get_forward_direction() * self.input_synapse_offset,
                                               -self.get_right_direction() * syn.scale_coefficient[...,2:]], -1)
                        syn.set_location_by_function(syn.coord_function)"""

    def backward(
        self, output=None, label=None, color=PURE_BLUE * k + (1 - k) * WHITE, run_time=3
    ):
        # with Seq():
        #    self.activate(reverse=True, color=color, run_time=run_time)
        #    self.reset_input_synapses()
        # return self
        # with Seq(run_time=run_time, animation_manager=self.animation_manager):
        with Lag(0.9, run_time=run_time):
            if label is not None:
                with Lag(0.65, run_time=1, animation_manager=self.animation_manager):
                    zap(label, output, color=color)
                    zap(output, self.layers[-1][0].shell, color=color)
            # self.animation_manager.context.timespan.current_time = (
            #    self.animation_manager.context.timespan.current_time - 1.5
            # )
            with Seq():
                self.activate(reverse=True, color=color)
                self.reset_input_synapses()
        return self

    def activate(
        self,
        output_generator=None,
        color=PURE_RED * k + (1 - k) * WHITE,
        run_time=1,
        reverse=False,
    ):
        layers = self.layers

        def pulse_synapses(neuron):
            with Sync(
                rate_func=pulse_fade, animation_manager=neuron.animation_manager
            ):  # ease_out_expo):
                for synapse in neuron.synapses:
                    synapse.wave_color(
                        color + GLOW * 1,
                        0.7,
                        reverse,
                        direction=self.get_forward_direction(),
                        new_color=tweak_color(synapse.color, 0.33) if reverse else None,
                    )

        def pulse_neuron(neuron):
            with Sync(
                run_time=1.1,
                rate_func=delay_fade,
                animation_manager=neuron.animation_manager,
            ):  # lambda t: pulse_fade(t, inflection=1.0)):
                for n, w in [[neuron.core, 0.15], [neuron.shell, 0]]:
                    with Seq(run_time=1):
                        n.wait(w)
                        n.wave_color(
                            (color + GLOW * 0.8),  # .set_opacity(
                            # 1 / neuron.shell.opacity.clamp_min(1e-5)
                            # ),
                            1,
                            reverse,
                            lag_duration=0.5,
                            direction=self.get_forward_direction(),
                        )
                        n.wait(w)

        pulse_funcs = [pulse_synapses, pulse_neuron]
        if reverse:
            pulse_funcs = list(reversed(pulse_funcs))
            layers = list(reversed(layers))

        with (
            Seq(animation_manager=self.animation_manager),
            Lag(
                0.55, rate_func=identity, animation_manager=self.animation_manager
            ),  # , run_time=run_time):
        ):
            for layer in layers:
                with Sync(animation_manager=self.animation_manager):
                    for neuron in layer:
                        with Lag(0.5, animation_manager=self.animation_manager):
                            for f in pulse_funcs:
                                f(neuron)
            if output_generator is None:
                return
            with Seq():
                # self.animation_manager.context.current_time = (
                #        self.animation_manager.context.current_time - 1.7
                # )
                with Off(animation_manager=self.animation_manager):
                    output = output_generator().move_next_to(
                        self.layers[-1][len(self.layers[-1]) // 2],
                        self.get_forward_direction(),
                        buffer=0,
                    )
                    output_colors = {
                        id(part): part.color.clone()
                        for part in output._wave_pulsed_parts()
                    }
                    for part in output._wave_pulsed_parts():
                        # Hide through the per-sample color alpha, not the
                        # primitive's scalar opacity. A filled circuit can
                        # then materialize behind the same spatial wave as
                        # its glow instead of globally brightening as its
                        # one opacity value ramps up. Written non-recursively,
                        # like the wave that restores it: a part whose helper
                        # children carry colour rows of their own (a Text's
                        # texture points) has more rows under a recursive set
                        # than its own colour getter returns.
                        part.set_non_recursive(color=part.color.set_opacity(0))
                    output.spawn(animate=False)

                def authored_color(part):
                    # wave_color refines a part that is sampled too coarsely
                    # to show the wave (a Surface's vertex grid, say), which
                    # leaves the color captured above indexed by the old
                    # sampling. Its first row still broadcasts over the new
                    # one, which is exactly right for the uniformly colored
                    # parts that refinement applies to.
                    authored = output_colors[id(part)]
                    if authored.shape[-2] != part.color.shape[-2]:
                        authored = authored.reshape(-1, authored.shape[-1])[:1]
                    return authored

                with Seq(run_time=1.5, animation_manager=self.animation_manager):
                    output.wave_color(
                        color + GLOW,
                        direction=self.get_forward_direction(),
                        wave_length=0.5,
                        # The output can be a multi-colored composite. Each
                        # part must settle to its own authored color, while
                        # the shared pulse supplies the uniform glow peak.
                        new_color=authored_color,
                        # An output materializes at whatever resolution it
                        # was authored with. Refining it here judges the
                        # sampling against the output's own extent rather
                        # than its size on screen, so a mob a few dozen
                        # pixels wide is pushed to the 64-per-axis ceiling
                        # for a wave that spans a handful of pixels -- and
                        # with restore_resolution False it keeps that
                        # geometry for the rest of the video.
                        refine_resolution=False,
                        # Restoring a refined Code panel creates a second
                        # coplanar incarnation in a different render batch.
                        # It can cover already-materialized glyphs until the
                        # handoff frame. This output is newly created, so
                        # retain its authored color grid as its stable
                        # topology and avoid the handoff altogether.
                        restore_resolution=False,
                    )
                return output


class SynapseV3(Cylinder):
    """V3 synapse: a thin filament with a lacquered (clearcoat) surface, so the
    wires pick up crisp light streaks on top of their colour-tracking fill.
    """

    def __init__(self, grid_height=5, *args, **kwargs):
        grid_height = None
        grid_width = None
        c = kwargs.get("color")
        if c is not None:
            c = tweak_color(c, strength=0.25, min_strength=0.25)
            kwargs["color"] = c
        else:
            c = WHITE
        super().__init__(grid_height=grid_height, grid_width=grid_width, **kwargs)
        """self.set_material(MeshPhysicalMaterial(
            color=c.set_glow(0.04),
            roughness=0.25,
            metalness=0.0,
            clearcoat=0.6,
            clearcoatRoughness=0.15,
            envMapIntensity=5.0,
        ))"""
        self.color = c
        # self.set_material(MeshBasicMaterial(color=c.set_glow(0.04)))
        self.set_shader(None)
        self.scale(0.01)


class NeuronV3(Neuron):
    """V3 neuron: the full physical-material design -- a lacquered clearcoat
    core inside a translucent glass shell with a soft sheen rim, shaded per
    fragment by the physical material's in-kernel port.
    """

    synapse_cls = SynapseV3

    def _make_core(self, grid_height, neuron_color):
        material = MeshPhysicalMaterial(
            color=neuron_color.set_glow(0.08),
            roughness=0.18,
            metalness=0.0,
            clearcoat=1.0,
            clearcoat_roughness=0.08,
            env_map_intensity=4.0,
            # reflectivity=1.0
        )
        return (
            Sphere(
                scene=self.scene,
                grid_height=grid_height,
                grid_width=grid_height,
                color=neuron_color,
                opacity=1.0,
            )
            .set_material(material)
            .scale(0.15)
        )

    def _make_shell(self, grid_height, neuron_color):
        rim_color = neuron_color * 0.8 + WHITE * 0.2
        # The rim still needs a boosted sheen: at shell opacity 0.25 the alpha
        # composite mutes it, and the shell's dark limb shading fights it.
        material = MeshPhysicalMaterial(
            color=neuron_color.set_opacity(0.25),
            roughness=0.12,
            metalness=0.0,
            clearcoat=1.0,
            clearcoat_roughness=0.05,
            sheen=8.0,
            sheen_roughness=0.1,
            sheen_color=rim_color,
            env_map_intensity=5.0,
            transmission=0.0,
            ior=5,
        )
        return (
            Sphere(
                scene=self.scene,
                opacity=1.0,
                grid_width=grid_height,
                grid_height=grid_height,
                color=neuron_color,
            )
            .set_material(material)
            .scale(0.21)
        )


class NeuralNetMLPV2(NeuralNetMLP):
    """Drop-in replacement for :class:`NeuralNetMLP` with upgraded visuals:
    glossy self-lit neuron cores inside translucent halo shells and filament
    synapses, built from MeshStandardMaterial. Same constructor and animation
    API.
    """

    neuron_cls = NeuronV2


class NeuralNetMLPV3(NeuralNetMLP):
    """Drop-in replacement for :class:`NeuralNetMLP` built from
    MeshPhysicalMaterial: lacquered clearcoat neuron cores inside glass shells
    with sheen rims, and clearcoat filament synapses. Same constructor and
    animation API.
    """

    neuron_cls = NeuronV3
