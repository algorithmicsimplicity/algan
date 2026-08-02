from __future__ import annotations

import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Lag, Off, Seq, Sync
from algan.constants.rate_funcs import delay_fade, identity, pulse_fade
from algan.constants.spatial import *  # ORIGIN, OUT, RIGHT
from algan.geometry.geometry import (
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
from algan.utils.tensor_utils import dot_product

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
_IDLE_SECONDS_PER_WAYPOINT = 4
_IDLE_DESIRED_RADIUS_PER_SPACING = 1
_IDLE_CLEARANCE_RADIUS_FRACTION = 1
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
    """Sample deterministic points uniformly inside each neuron's unit ball."""
    _idle_rng.manual_seed(_IDLE_WALK_SEED)
    shape = (walk_radii.numel(), _IDLE_WAYPOINT_COUNT - 1, 3)
    directions = torch.randn(
        shape, dtype=dtype, device=device, generator=_idle_rng
    )
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radial_scale = torch.rand(
        (*shape[:-1], 1), dtype=dtype, device=device, generator=_idle_rng
    ).pow(1 / 3)
    random_points = directions * radial_scale
    unit_waypoints = torch.cat(
        [torch.zeros((shape[0], 1, 3), dtype=dtype, device=device), random_points],
        dim=1,
    )
    unit_waypoints = unit_waypoints - dot_product(unit_waypoints, direction) * direction
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
    for index, neuron in enumerate(neurons):
        target = (
            world_positions[index]
            if scalar_time
            else world_positions[:, index : index + 1]
        )
        neuron.move_to(target)

    forward = net.get_forward_direction()
    for neuron in net.layers[0]:
        for synapse in neuron.synapses:
            synapse.move_between_points(
                neuron.location + forward * net.input_synapse_offset,
                neuron.location,
            )
    for previous_layer, layer in zip(net.layers[:-1], net.layers[1:-1]):
        for neuron in layer:
            for source, synapse in zip(previous_layer, neuron.synapses):
                synapse.move_between_points(source.location, neuron.location)
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
        #grid_height = 20  # None
        #grid_width = 12
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
        self._idle_waypoints = _make_idle_waypoints(
            self._idle_walk_radii,
            direction,
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
        o = self.forward(
            input_values, output_generator, run_time, reset=False, color=forward_color
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
        with Seq(run_time=run_time, animation_manager=self.animation_manager):
            if label is not None:
                with Lag(0.65, run_time=6, animation_manager=self.animation_manager):
                    zap(label, output, color=color)
                    zap(output, self.layers[-1][0].shell, color=color)
                self.animation_manager.context.timespan.current_time = (
                    self.animation_manager.context.timespan.current_time - 1.5
                )
            self.activate(reverse=True, color=color, run_time=run_time)
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

        with Seq(animation_manager=self.animation_manager):
            with Lag(
                0.55, rate_func=identity, animation_manager=self.animation_manager
            ):  # , run_time=run_time):
                for layer in layers:
                    with Sync(animation_manager=self.animation_manager):
                        for neuron in layer:
                            with Lag(0.5, animation_manager=self.animation_manager):
                                for f in pulse_funcs:
                                    f(neuron)
                if output_generator is None:
                    return
                with Seq():
                    #self.animation_manager.context.current_time = (
                    #        self.animation_manager.context.current_time - 1.7
                    #)
                    with Off(animation_manager=self.animation_manager):
                        output = output_generator().move_next_to(
                            self.layers[-1][len(self.layers[-1]) // 2],
                            self.get_forward_direction(),
                            buffer=0,
                        )
                        for _ in output.get_descendants():
                            if not _.is_primitive:
                                continue
                            _.set(color=_.color.set_opacity(0))
                        output.spawn(animate=False)
                    with Seq(run_time=3, animation_manager=self.animation_manager):
                        output.wave_color(
                            color + GLOW,
                            direction=self.get_forward_direction(),
                            opacity=1,
                            wave_length=1.5,
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
            clearcoatRoughness=0.08,
            envMapIntensity=4.0,
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
            clearcoatRoughness=0.05,
            sheen=8.0,
            sheenRoughness=0.1,
            sheenColor=rim_color,
            envMapIntensity=5.0,
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
