================================
An Overview of Algan's Internals
================================

Algan consists of two main systems: the animation system and the rendering system.

The animation system is responsible for creating intermediate states of Mobs by
interpolating between the Mob state at the beginning and end of each animation. In
Algan, animating is not done as animations are created, but rather Algan internally
makes a record of each new animation as it is created. Once a command to render has been given,
the animating system then reads through the recorded data and produces interpolated states.
These interpolated states are then fed to the rendering system to produce the video.

The rendering system then uses the states of Mobs at each point in time, along
with the camera location and orientation, light sources, and render settings,
to produce the final output video.

The Animation System
********************

The basic workhorse of the animation system is the :func:`.animated_function` decorator.
This decorator transforms a regular function into an animated function.
The way it works is, whenever an animated_function is called, the decorator will
make a record of the fact that the function was called, what time the function was
called at, and with what parameters the function was called, and write this information
to the global animation timeline (an :class:`.AnimationTimeline`, accessed through
:class:`.TimelineManager`). The timeline stores, per animatable attribute, one shared
buffer holding every Mob's current values (each Mob owns a set of rows, keyed by its
id) together with the history of edits made to those rows, plus the record of all
animated function applications and each Mob's spawn/despawn interval (its
:class:`.Lifespan`, exposed as :attr:`.Animatable.lifespan`).

Note that all of the attribute modification animations are implemented as animated_functions
under the hood, by decorating the attribute setters for animatable attributes.

When the animated_function decorator records a function application, it uses
the settings of the current animation context. Notably, the end of the animation
is set as the context's current time + run_time_unit.

After writing the animation data, the animated_function decorator then calls the original
function, and finally it updates the current_time of the current context to the place
on the timeline where the next animation should take place. Note that the update
depends on the type of context, for Seq it will be one run_time_unit in the future,
for Sync it will be the same time.

At animation time (:meth:`.AnimationTimeline.set_state_to_times`), Algan first
materializes every attribute buffer at the requested frame times from the recorded
edits, in one batched pass per attribute, and then re-applies the recorded functions
with their animated parameters interpolated per frame. This is done in batches of
time steps, sized to fit in the animation memory budget.


The Rendering System
********************

The main logic for the rendering system is contained in :class:`.Primitive`.
The rendering system takes as input a :class:`.Primitive` object (representing
a batch of mob states), a camera, and render settings.
One of the unique features of the Algan rendering system is that it renders
frames in batches, instead of one at a time. The number of frames rendered
in each batch is defined by DEFAULT_BATCH_SIZE_FRAMES.
For example, if the rendering primitives are :class:`.TrianglePrimitive` s, the input to the
rendering system would be a tensor of shape [T, N, 3, 3], where T is the number
of time steps animated (aka the number of frames in the batch), N is the number
of triangles in the scene, 3 is the number of corners per triangle, and 3 is the
number of spatial dimensions (we are in 3D).
:class:`.BezierCircuitWithBorderThenFill` batches are of size [T,N,S,4,3], where S is
the number of segments per circuit, 4 is the number of control points per
Bezier (Cubic).


