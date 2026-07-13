import torch

from algan.rendering.shaders.pbr_shaders import default_shader
from algan.animation.animation_contexts import Off, Sync, Seq, Lag
from algan.constants.spatial import *  # ORIGIN, OUT, RIGHT
from algan.mobs.mob import Mob
from algan.mobs.shapes_3d import Sphere, Cylinder
from algan.constants.rate_funcs import identity, ease_in_expo, ease_out_expo
from algan.rendering.shaders.pbr_shaders import null_shader
from algan.rendering.shaders.materials import MeshStandardMaterial
from algan.utils.tensor_utils import dot_product, unsquish, squish
from algan.mobs.text import Tex
from algan.constants.rate_funcs import smooth, pulse_fade, delay_fade


# Synapses jitter their colour for visual variety. Draw that jitter from a
# dedicated, fixed-seed generator (reseeded per net in NeuralNetMLP.__init__)
# rather than the global RNG: otherwise every render produced different synapse
# colours, so the same scene rendered twice differed by tens of code values --
# which reads as nondeterministic ("order-sensitive") output and makes the
# frame-comparison tests impossible to satisfy. A private generator keeps the
# synapse-to-synapse variety while making each render byte-reproducible, and
# leaves the global torch RNG untouched for everything else.
COLOR_JITTER_SEED = 0xA76A
_color_rng = torch.Generator(device=COMPUTING_DEFAULTS.animation_device).manual_seed(COLOR_JITTER_SEED)


def tweak_color(c, strength=0.2, min_strength=0.0):
    t = torch.rand((1,), generator=_color_rng).item()
    t = t * strength + (1-t) * min_strength
    #m = torch.randint(0, 2, (1,), generator=_color_rng)
    #target_c = WHITE * m + (1 - m) * BLACK
    target_c = c.set_rgb(torch.rand(c.rgb.shape, device=c.rgb.device, dtype=c.rgb.dtype, generator=_color_rng))
    return c * (1 - t) + t * target_c


gr = 0.15
gs = 0.5

class Synapse(Cylinder):
    def __init__(self, grid_height=5, *args, **kwargs):
        grid_height = 20#None
        grid_width = 12
        if 'color' in kwargs:
            c = kwargs['color']
            kwargs['color'] = tweak_color(c, strength=0.25, min_strength=0.25)
        super().__init__(grid_height=grid_height, grid_width=grid_width, glow_radius=gr, **kwargs)
        self.scale(0.02)


class Neuron(Mob):
    synapse_cls = Synapse

    def __init__(self, input_locs, direction, neuron_color, **kwargs):
        super().__init__(**kwargs)
        grid_height = 12
        self.core = self._make_core(grid_height, neuron_color).move_to(self.location)
        self.shell = (
            self._make_shell(grid_height, neuron_color)
            .move_to(self.location)
            .look(direction, axis=1)
        )
        self.synapses = [
            self.synapse_cls(grid_height, color=neuron_color).move_between_points(l, self.location)
            for l in input_locs
        ]
        self.add_children(self.core, self.shell, self.synapses)

    def _make_core(self, grid_height, neuron_color):
        return Sphere(
            grid_height=grid_height, grid_width=grid_height, color=neuron_color
        ).scale(0.17)

    def _make_shell(self, grid_height, neuron_color):
        return (
            Sphere(opacity=0.5, grid_width=grid_height, grid_height=grid_height,
                   color=neuron_color, glow_radius=gr)
            .set_shader(None)
            .scale(0.2)
        )


# class Layer(Mob):
#    def __init__(self, input_locs, neuron_locs, **kwargs):
#        super().__init__(**kwargs)
#        self.neurons = [Neuron(input_locs, location=l) for l in neuron_locs]
#        self.add_children(self.neurons)


# NOTE: only the material shaders with in-kernel fragment ports render
# correctly on the deterministic renderer (basic / lambert / phong / standard;
# see _build_core_shader_ids). MeshPhysicalMaterial (clearcoat/sheen) falls back
# to the per-vertex path, which currently produces flat unlit grey -- so the V2
# looks below are built from MeshStandardMaterial only. Metalness is kept at 0:
# metalness > 0 routes mirror reflectivity into the ray tracer.


class SynapseV2(Cylinder):
    """Improved synapse: a thin filament lit from within (emissive) with a
    glossy dielectric surface, so pulses read as light travelling down a wire."""

    def __init__(self, grid_height=5, *args, **kwargs):
        grid_height = 20
        grid_width = 12
        c = kwargs.get('color', None)
        if c is not None:
            c = tweak_color(c, strength=0.25, min_strength=0.25)
            kwargs['color'] = c
        else:
            c = WHITE
        super().__init__(grid_height=grid_height, grid_width=grid_width, glow_radius=gr, **kwargs)
        # Fill light comes from env_map_intensity (ambient = albedo * 0.1 * env)
        # rather than emissive, so it tracks the albedo during colour-wave
        # pulses instead of tinting them with the resting colour.
        self.set_material(MeshStandardMaterial(
            color=c.set_glow(0.04),
            roughness=0.3,
            metalness=0.0,
            envMapIntensity=4.5,
        ))
        self.scale(0.02)


class NeuronV2(Neuron):
    """Improved neuron: a glossy self-lit core (crisp specular highlight over
    an emissive base) inside a soft translucent halo shell, replacing the
    flat-shaded spheres of :class:`Neuron`."""

    synapse_cls = SynapseV2

    def _make_core(self, grid_height, neuron_color):
        material = MeshStandardMaterial(
            color=neuron_color.set_glow(0.08),
            roughness=0.2,
            metalness=0.0,
            envMapIntensity=4.0,
        )
        return (
            Sphere(grid_height=grid_height, grid_width=grid_height, color=neuron_color)
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
            Sphere(opacity=0.3, grid_width=grid_height, grid_height=grid_height,
                   color=neuron_color, glow_radius=gr)
            .set_material(material)
            .scale(0.21)
        )

k = 1


def zap(mob1, mob2, color=BLUE, direction=UP, num_points=3):
    with Off():
        p1 = mob1.get_points_evenly_along_direction(direction)
        p2 = mob2.get_points_evenly_along_direction(direction)
        syns = [Synapse().move_between_points(p1[i], p2[i]) for i in range(num_points)]
        for s in syns:
            for _ in s.get_descendants():
                if not _.is_primitive:
                    continue
                _.color = _.color.set_opacity(0)
            s.spawn(animate=False)
    with Sync(run_time=1):
        for s in syns:
            s.wave_color(
                color + GLOW,
                direction=s.get_upwards_direction(),
                opacity=1,
                wave_length=1.5,
            )
    with Off():
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
        with Off():
            self.layers = [
                [
                    self.neuron_cls(
                        [l + direction * self.input_synapse_offset],
                        direction,
                        location=l,
                        neuron_color=neuron_color
                    )
                    for l in neuron_locs[0]
                ]
            ] + [
                [
                    self.neuron_cls(neuron_locs[i], direction, location=l, neuron_color=neuron_color)
                    for l in neuron_locs[i + 1]
                ]
                for i in range(len(neuron_locs) - 1)
            ]
        # self.layers = [[Neuron(neuron_locs[i], location=l) for l in neuron_locs[i+1]] for i in range(len(neuron_locs)-1)]

        self.add_children(self.layers)

    def increment_weight(self):
        weight = self.layers[1][0].synapses[0]
        weight.orig_color = weight.color
        with Seq():
            weight.color = weight.color + GLOW * 0.2
            self.increment_label = Tex('w := w + 0.001').move_next_to(weight, LEFT+DOWN, buffer=0.05).spawn()
        self.incremented_weight = weight
        return weight

    def unincrement_weight(self):
        with Sync():
            self.increment_label.despawn()
            self.incremented_weight.color = self.incremented_weight.orig_color

    def train_step(
        self,
        input,
        output_generator,
        label,
        run_time=3,
        forward_color=PURE_RED * k + (1 - k) * WHITE,
        backward_color=PURE_BLUE * k + (1 - k) * WHITE,
    ):
        o = self.forward(
            input, output_generator, run_time, reset=False, color=forward_color
        )  # .get_component_mobs())
        #o.move_next_to(label, -self.get_right_direction())
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
        with Seq(run_time=run_time):
            with Sync(run_time=1):
                for neuron, neuron_inputs in zip(self.layers[0], inputs):
                    for syn, inp in zip(neuron.synapses, neuron_inputs):
                        syn.set_start_point(inp)  # , n.location)
            out = self.activate(
                run_time=run_time, output_generator=output_generator, **kwargs
            )
            with Sync():
                if reset:
                    self.reset_input_synapses()
                out.move(self.get_forward_direction() * 0.25)
            return out

    def reset_input_synapses(self):
        with Sync(run_time=1):
            for n in self.layers[0]:
                for syn in n.synapses:
                    syn.move_between_points(
                        n.location + self.get_forward_direction() * self.input_synapse_offset,
                        n.location
                    )  # , n.location)
            '''with Off():
                for n in self.layers[0]:
                    for syn in n.synapses:
                        syn.location = n.location + self.get_forward_direction() * self.input_synapse_offset * 0.5
                        syn.basis = torch.cat([-self.get_upwards_direction() * syn.scale_coefficient[...,:1],
                                               self.get_forward_direction() * self.input_synapse_offset,
                                               -self.get_right_direction() * syn.scale_coefficient[...,2:]], -1)
                        syn.set_location_by_function(syn.coord_function)'''


    def backward(
        self, output=None, label=None, color=PURE_BLUE * k + (1 - k) * WHITE, run_time=3
    ):
        with Seq(run_time=run_time):
            if label is not None:
                with Lag(0.65, run_time=6):
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
        color=PURE_RED * k + (1 - k) * WHITE,
        run_time=1,
        reverse=False,
        output_generator=None,
    ):
        layers = self.layers

        def pulse_synapses(neuron):
            with Sync(rate_func=pulse_fade):#ease_out_expo):
                for synapse in neuron.synapses:
                    synapse.wave_color(color + GLOW * gs, 0.7, reverse,
                                       direction=self.get_forward_direction(),
                                       new_color=tweak_color(synapse.color, 0.33) if reverse else None)

        def pulse_neuron(neuron):
            with Sync(run_time=1.1, rate_func=delay_fade):#lambda t: pulse_fade(t, inflection=1.0)):
                for n in [neuron.core, neuron.shell]:
                    n.wave_color(
                        (color + GLOW * gs),#.set_opacity(
                            #1 / neuron.shell.opacity.clamp_min(1e-5)
                        #),
                        1,
                        reverse,
                        lag_duration=0.5,
                        direction = self.get_forward_direction()
                    )

        pulse_funcs = [pulse_synapses, pulse_neuron]
        if reverse:
            pulse_funcs = list(reversed(pulse_funcs))
            layers = list(reversed(layers))

        with Seq():
            with Lag(0.70, rate_func=identity):  # , run_time=run_time):
                for layer in layers:
                    with Sync():
                        for neuron in layer:
                            with Lag(0.5):
                                for f in pulse_funcs:
                                    f(neuron)
            self.animation_manager.context.timespan.current_time = (
                self.animation_manager.context.timespan.current_time - 1.7
            )
            if output_generator is None:
                return
            with Off():
                output = output_generator().move_next_to(
                    self.layers[-1][len(self.layers[-1]) // 2], self.get_forward_direction(), buffer=0
                )
                for _ in output.get_descendants():
                    if not _.is_primitive:
                        continue
                    _.set_opacity(0)
                output.spawn(animate=False)
            with Seq(run_time=3):
                output.wave_color(
                    color + GLOW, direction=self.get_forward_direction(), opacity=1, wave_length=1.5
                )
            return output


class NeuralNetMLPV2(NeuralNetMLP):
    """Drop-in replacement for :class:`NeuralNetMLP` with upgraded visuals:
    lacquered self-lit neuron cores inside glass shells (clearcoat + sheen rim)
    and emissive filament synapses. Same constructor and animation API."""

    neuron_cls = NeuronV2
