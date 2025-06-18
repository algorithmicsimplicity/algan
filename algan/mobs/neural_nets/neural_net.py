from algan.animation.animation_contexts import Off, Sync, Seq, Lag
from algan.constants.spatial import *#ORIGIN, OUT, RIGHT
from algan.mobs.mob import Mob
from algan.mobs.shapes_3d import Sphere, Cylinder
from algan.constants.rate_funcs import identity
from algan.utils.tensor_utils import dot_product


class Neuron(Mob):
    def __init__(self, input_locs, direction, **kwargs):
        super().__init__(**kwargs)
        grid_height = 10
        self.core = Sphere(grid_height=grid_height).scale(0.15).move_to(self.location)
        self.shell = Sphere(opacity=0.5, grid_height=grid_height).scale(0.2).move_to(self.location).look(direction, axis=1)
        self.synapses = [Cylinder(grid_height=grid_height).scale(0.04).move_between_points(l, self.location) for l in input_locs]
        self.add_children(self.core, self.shell, self.synapses)


#class Layer(Mob):
#    def __init__(self, input_locs, neuron_locs, **kwargs):
#        super().__init__(**kwargs)
#        self.neurons = [Neuron(input_locs, location=l) for l in neuron_locs]
#        self.add_children(self.neurons)


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

    def forward(self, inputs, output_generator=None, color=PURE_GREEN, run_time=1, **kwargs):
        with Seq(run_time=run_time):
            with Sync(run_time=3):
                for n in self.layers[0]:
                    for syn, inp in zip(n.synapses, inputs):
                        syn.set_start_point(inp.location)#, n.location)
            return self.activate(color=color, run_time=run_time, output_generator=output_generator, **kwargs)

    def backward(self, color=BLUE+GLOW, run_time=1):
        with Seq(run_time=run_time):
            self.activate(reverse=True, color=color, run_time=run_time)
            with Sync(run_time=3):
                for n in self.layers[0]:
                    for syn in n.synapses:
                        syn.set_start_point(n.location + RIGHT * self.input_synapse_offset)#, n.location)

    def activate(self, color=PURE_RED+GLOW, run_time=1, reverse=False, output_generator=None):
        layers = self.layers

        def pulse_synapses(neuron):
            with Sync():
                for synapse in neuron.synapses:
                    synapse.wave_color(color, 0.33, reverse)

        def pulse_neuron(neuron):
            neuron.shell.wave_color(color, 0.1, reverse)

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
            self.animation_manager.context.current_time -= 1.5
            if output_generator is None:
                return
            with Off():
                output = output_generator().move_next_to(self.layers[-1][len(self.layers[-1])//2], self.direction, buffer=0)
                for _ in output.get_descendants():
                    if not _.is_primitive:
                        continue
                    _.set_opacity(0)
                output.spawn(animate=False)
            output.wave_color(color, direction=self.direction, set_opaque=True)
            return output

