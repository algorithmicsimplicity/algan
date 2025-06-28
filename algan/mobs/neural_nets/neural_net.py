from algan.animation.animation_contexts import Off, Sync, Seq, Lag
from algan.constants.spatial import *#ORIGIN, OUT, RIGHT
from algan.mobs.mob import Mob
from algan.mobs.shapes_3d import Sphere, Cylinder
from algan.constants.rate_funcs import identity, ease_in_expo, ease_out_expo
from algan.utils.tensor_utils import dot_product


class Synapse(Cylinder):
    def __init__(self, grid_height=10, *args, **kwargs):
        super().__init__(grid_height=grid_height, grid_aspect_ratio=2)
        self.scale(0.02)

class Neuron(Mob):
    def __init__(self, input_locs, direction, **kwargs):
        super().__init__(**kwargs)
        grid_height = 10
        self.core = Sphere(grid_height=grid_height, grid_aspect_ratio=2).scale(0.15).move_to(self.location)
        self.shell = Sphere(opacity=0.5, grid_height=grid_height, grid_aspect_ratio=2).scale(0.2).move_to(self.location).look(direction, axis=1)
        self.synapses = [Synapse(grid_height).move_between_points(l, self.location) for l in input_locs]
        self.add_children(self.core, self.shell, self.synapses)


#class Layer(Mob):
#    def __init__(self, input_locs, neuron_locs, **kwargs):
#        super().__init__(**kwargs)
#        self.neurons = [Neuron(input_locs, location=l) for l in neuron_locs]
#        self.add_children(self.neurons)


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
            s.wave_color(color + GLOW, direction=s.get_upwards_direction(), opacity=1, wave_length=1.5)
    return


class NeuralNetMLP(Mob):
    def __init__(self, dims, direction=RIGHT, orth_direction=UP, layer_spacing=1, neuron_spacing=0.5, input_locs=None, **kwargs):
        super().__init__(**kwargs)
        start = ORIGIN if input_locs is None else sum(input_locs) / len(input_locs)

        def proj(x):
            return x - dot_product(x, direction) * direction
        def rng(size):
            return torch.rand(size)*2-1
        #orth_direction = get_orthonormal_vector(direction)
        #neuron_locs = [start + proj(rng((d,3))*0.4) + direction * rng((1,)) * 0.1 + direction*(i)*layer_spacing for i, d in enumerate(dims)]
        neuron_locs = [start + (torch.arange(d).unsqueeze(-1)-(d//2))*neuron_spacing*orth_direction + direction*(i)*layer_spacing for i, d in enumerate(dims)]
        if input_locs is not None:
            neuron_locs[0] = input_locs
        self.input_synapse_offset = -layer_spacing*0.5
        with Off():
            self.layers = [[Neuron([l + direction * self.input_synapse_offset], direction, location=l) for l in neuron_locs[0]]] +\
                     [[Neuron(neuron_locs[i], direction, location=l) for l in neuron_locs[i+1]] for i in range(len(neuron_locs)-1)]
        #self.layers = [[Neuron(neuron_locs[i], location=l) for l in neuron_locs[i+1]] for i in range(len(neuron_locs)-1)]
        self.add_children(self.layers)
        self.direction = direction

    def train_step(self, input, output_generator, label, run_time=3, forward_color=PURE_RED, backward_color=BLUE):
        o = self.forward(input, output_generator, run_time, color=forward_color)  # .get_component_mobs())
        o.move_next_to(label, -self.get_right_direction())
        self.backward(o, label, color=backward_color, run_time=run_time)
        o.despawn()
        return self

    def forward(self, inputs, output_generator=None, run_time=3, **kwargs):
        if isinstance(inputs, Mob):
            inputs = [[_] for _ in inputs.get_points_evenly_along_direction(-(self.get_right_direction() + self.get_upwards_direction()), len(self.layers[0]))]
        else:
            inputs = [[_.location for _ in inputs] for _ in range(len(self.layers[0]))]
        with Seq(run_time=run_time):
            with Sync(run_time=1):
                for neuron, neuron_inputs in zip(self.layers[0], inputs):
                    for syn, inp in zip(neuron.synapses, neuron_inputs):
                        syn.set_start_point(inp)#, n.location)
            return self.activate(run_time=run_time, output_generator=output_generator, **kwargs)

    def backward(self, output=None, label=None, color=BLUE, run_time=3):
        with Seq(run_time=run_time):
            if label is not None:
                with Lag(0.6, run_time=6):
                    zap(label, output, color=color)
                    zap(output, self.layers[-1][0].shell, color=color)
                self.animation_manager.context.current_time = self.animation_manager.context.current_time - 1.5
            self.activate(reverse=True, color=color, run_time=run_time)
            with Sync(run_time=1):
                for n in self.layers[0]:
                    for syn in n.synapses:
                        syn.set_start_point(n.location + RIGHT * self.input_synapse_offset)#, n.location)
        return self

    def activate(self, color=PURE_RED, run_time=1, reverse=False, output_generator=None):
        layers = self.layers

        def pulse_synapses(neuron):
            with Sync(rate_func=ease_out_expo):
                for synapse in neuron.synapses:
                    synapse.wave_color(color + GLOW * 0.8, 0.9, reverse)

        def pulse_neuron(neuron):
            with Seq(run_time=2):
                neuron.shell.wave_color(color + GLOW, 1, reverse, lag_duration=0.5)

        pulse_funcs = [pulse_synapses, pulse_neuron]
        if reverse:
            pulse_funcs = list(reversed(pulse_funcs))
            layers = list(reversed(layers))

        with Seq():
            with Lag(0.6, rate_func=identity):#, run_time=run_time):
                for layer in layers:
                    with Sync():
                        for neuron in layer:
                            with Lag(0.5):
                                for f in pulse_funcs:
                                    f(neuron)
            self.animation_manager.context.current_time = self.animation_manager.context.current_time - 1.3
            if output_generator is None:
                return
            with Off():
                output = output_generator().move_next_to(self.layers[-1][len(self.layers[-1])//2], self.direction, buffer=0)
                for _ in output.get_descendants():
                    if not _.is_primitive:
                        continue
                    _.set_opacity(0)
                output.spawn(animate=False)
            with Seq(run_time=3):
                output.wave_color(color + GLOW, direction=self.direction, opacity=0, wave_length=1.5)
            return output

