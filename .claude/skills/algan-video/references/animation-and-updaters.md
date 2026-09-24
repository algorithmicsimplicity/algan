# Animation, updaters, and custom motion

## Recorded animation, not a live simulation

Constructing and editing a scene records what should happen. Algan computes
states later at requested times, often in frame batches. The authoring-time
Python value of an attribute is not a stream of future frame values. Functions
used during rendering must work when time is batched or revisited.

Create geometry and configure materials before spawning. Unspawned changes are
instantaneous. After spawn, ordinary transforms and animatable assignments become
timeline changes. The common operations are:

```python
shape = Square().scale(0.5).move_to(LEFT * 2)
shape.spawn()
shape.move(RIGHT)                  # displacement, not a target location
shape.move_to(UP)                  # absolute destination
shape.rotate(90, OUT)              # native Algan angles are degrees
shape.scale(1.2)
shape.color = BLUE
shape.opacity = 0.5
```

Each top-level animated action normally takes one second. `spawn()` itself
normally animates appearance. For an instant initial state:

```python
with Off():
    shape = Square().scale(0.5).move_to(LEFT * 2).spawn()
```

An object outside its spawned lifespan is not visible. Use `despawn()` to end its
presence, with `Off()` for an instant removal. Do not rely on an object becoming
visible merely because its constructor ran. An unspawned morph target should be
created with `add_to_scene=False` so it is not treated as an omitted visible actor.

## Timing and nesting

| Context | Timing |
|---|---|
| `Seq()` | Children play one after another |
| `Sync()` | Children start together |
| `Lag(ratio)` | Each child starts after the preceding child's duration multiplied by `ratio` |
| `Off()` | Changes take no animation time |

`Lag(0)` corresponds to simultaneous starts and `Lag(1)` to sequential starts.
Use `runtime` for the entire context and `runtime_per_part` for each child.
The outer total duration wins when both are set. Durations are finite,
nonnegative seconds.

```python
with Sync(runtime=2.0):
    shape.move(RIGHT * 2)
    shape.rotate(90, OUT)
    shape.opacity = 0.6

with Seq(runtime=2.0):
    shape.move(UP)                 # first half of the two-second sequence
    shape.move(DOWN)               # second half

Scene.wait(0.5)                    # consumes video time; does not sleep Python
```

Nested contexts count as one child of the parent. A plain helper function does
not create a context: a helper containing three transforms still records three
separate actions unless it opens a context. For example, two internally
sequential routines can run together:

```python
with Sync():
    with Seq(runtime=2):
        first.move(RIGHT)
        first.move(UP)
    with Seq(runtime=2):
        second.move(LEFT)
        second.move(DOWN)
```

A context's `runtime` rescales its whole content, and a `Speech`/`Audio` block
rescales its content to the clip. Inside a plain `Sync`, each child keeps its own
relative duration, so a 1-second move beside a 3-second sequence finishes early.
To make one change last a whole block while other changes run in sequence beside
it, use `Sync(equalize_runtimes=True)`, which stretches every direct child to the
duration of the longest one:

```python
with Speech("An exact segment of the user's narration."):
    with Sync(equalize_runtimes=True):
        camera_move()                       # stretched to the content's duration
        with Seq():                         # the content defines the block's timing
            with Seq(runtime=1.0):
                first.spawn()
            with Seq(runtime=2.0):
                second.move(RIGHT)
```

Every direct child is stretched, not only the one meant to fill, and the longest
child sets the length. Give the spanning change no explicit runtime so that it
stretches rather than stretching the content, and put all other actions,
including short accents that must stay short, inside a single `Seq`/`Sync` child.
The stretch applies only to direct children of that `Sync`. The Speech clip then
scales the whole block to the spoken duration.

Use `easing=` for the specified time mapping. `easings.identity` gives linear
progress; `easings.smooth` is the default nonlinear mapping. A custom easing
accepts a tensor and returns a tensor. Do not prescribe an easing as a creative
rule. Avoid unintentionally applying a nonlinear easing to a parent containing
several carefully timed subactions.

Timing belongs on contexts, not ordinary calls:

```python
# Correct ordinary transform timing:
with Seq(runtime=2):
    shape.move(RIGHT)

# Not Algan's ordinary transform API:
# shape.move(RIGHT, run_time=2)
# shape.move(RIGHT, runtime=2)
```

A few convenience APIs explicitly accept timing, including text `write()` and
imported-model `play_animation()`. Use their documented signatures rather than
assuming all Mob methods accept those keywords. Algan's spelling is `runtime`,
not `duration` or `run_time`, and `easing`, not `rate_func`.

## Morphs, groups, and reusable mechanics

Use a compatible target for `become`:

```python
with Seq(runtime=1.0):
    shape.become(Triangle(add_to_scene=False))
```

Do not assume arbitrary mesh topology, shader replacement, or imported skeletal
animation can be expressed by `become`. A surface's UV-preserving shape change
uses `set_shape_to`; see the geometry reference. Check a short render at the
start, middle, and end of any nontrivial morph.

A `Group([a, b])` is a transformable hierarchy. A Python list is not. Child Mobs
follow parent transforms; use an updater for relationships that should not
inherit every transform, such as a label that follows position without rotation.
Indexing a packed Mob accesses parts of its existing data and lifespan; it is
not an independent copy. `clone(spawn=False)` creates a separate unspawned Mob.
Do not expect historical animation or updater ownership to transfer to a
replacement merely because its shape looks the same.

Prefer a normal function wrapping public calls for reusable finite sequences.
Check installed built-in animations before implementing an equivalent custom
motion. Do not borrow Manim's `self.play(...)` or `.animate` conventions: adapted
geometry still uses Algan's authoring timeline.

## Updater contract

Attach a continuing rule with `mob.add_updater(function, *args, **kwargs)`. The
callback receives the Mob and elapsed time since attachment:

```python
import torch

def offset_over_time(mob, t, amplitude, frequency):
    mob.move(UP * amplitude * torch.sin(2 * PI * frequency * t))

updater_id = shape.add_updater(offset_over_time, 0.4, 1.0)
Scene.wait(2.0)
shape.remove_updater(updater_id)
Scene.wait(0.5)
```

The numeric values are demonstrations, not motion recommendations. Algan first
materializes normal timeline animation and then applies the updater on top of
that state. The offset above is recomputed for every requested time; it is not
accumulated from the previous frame. Thus it can compose with a separately
recorded horizontal move.

`t` is a torch tensor with shape `[frames, 1, 1]`. It is **elapsed seconds**, not
frame delta, frame index, or normalized 0–1 progress. Use `torch.sin`,
`torch.cos`, `torch.where`, and broadcasting. Do not use Python `math` functions,
`.item()`, or `if t > ...` on a frame batch. For piecewise behavior, use tensor
selection so different frames can take different branches.

Every updater must accept `t`, even when unused. `add_updater` returns an integer
ID, not the Mob. Keep the Mob reference and ID separately. Removing an updater
stops the continuing rule and preserves its attained state rather than undoing
it; inspect the removal boundary when combining it with other animation. The
updater alone adds no timeline duration: animate something or call `Scene.wait`.

### Following another object

Read the target's state *inside* the callback so the dependency is evaluated at
the materialized frame time:

```python
def follow(mob, t, target):
    mob.move_next_to(target, DOWN, buffer=0.2)

follower_id = follower.add_updater(follow, target)
```

Capturing `target.location` once during authoring freezes that position and does
not track the target. When attaching callbacks in a loop, pass the target as an
explicit updater argument or bind it in the callback's defaults; do not leave
all closures pointing at the loop's last target.

The same contract applies to the camera:

```python
tracking_id = Scene.get_camera().add_updater(
    lambda camera, t: camera.look_at(target.location)
)
```

Avoid cyclic dependencies and conflicting writers to the same attribute. For
several objects driven by one quantity, derive their states from the same
well-defined time function rather than incrementing a shared Python variable.

### Replay-safe callback checklist

The function should derive output from frame time, the current materialized
inputs, and stable parameters. Do not keep a mutable previous-frame accumulator,
use wall-clock time, open files, request network data, synthesize audio, create
new Mobs, or randomly sample on every evaluation. Precompute data and deterministic
random choices outside callbacks. Do not allocate tensors on a hard-coded CPU
when operands may be on another device; use operations on the incoming tensors,
`zeros_like`, or an explicitly matched device and dtype.

Shapes must broadcast over arbitrary frame batches. Never assume a callback is
executed only once, only for a single frame, or only in chronological order.
Changing geometry topology or text glyph construction inside an updater is not
a substitute for a verified geometry-changing API. Animate `DecimalNumber.value`
for numerical displays rather than constructing `Text` per frame.

## Custom finite-duration animation

An `@animated_function` describes the state at a parameter value. The decorator
interpolates explicitly named arguments and records the function application:

```python
import torch

@animated_function(animated_args={'u': 0.0})
def move_on_curve(mob, u):
    mob.location = RIGHT * (4.0 * u - 2.0) + UP * torch.sin(PI * u)

with Off():
    shape = Square().scale(0.3).move_to(LEFT * 2).spawn()
with Seq(runtime=2.0, easing=easings.identity):
    move_on_curve(shape, 1.0)
```

The first argument is a Mob. Keys in `animated_args` name function arguments;
values specify their initial scalar values. The call supplies their target
values. Those parameters can become batched tensors during replay, so use torch
operations. Additional arguments can carry fixed data. Keep the function body
free of external side effects just like an updater.

Assignments inside the body describe that evaluation's state; they are not
separate subanimations. Use several interpolated arguments when needed. A normal
helper function, a decorated finite animation, and an updater solve different
timing problems; choose by the user's requested behavior, not by coding habit.

## Explicit timestamps

Prefer contexts when they express the requested timing. For data-driven start
times, use a context you explicitly enter and exit:

```python
with Seq() as timeline:
    start = timeline.current_time
    for mob, offset_seconds in scheduled_objects:
        timeline.current_time = start + offset_seconds
        with Seq(runtime=0.5):
            mob.move(UP)
    timeline.current_time = timeline.end_time
```

Context timestamps resolve when contexts exit. Do not write events manually
against the unentered default root, and restore `current_time` to `end_time`
before resuming sequential authoring. An enclosing `runtime` can rescale these
times, so do not accidentally apply one when offsets must remain absolute.
Avoid overlapping contradictory edits unless the intended composition has been
verified at intermediate times.

Source basis: [timeline and callback sources](api-sources.md#animation-and-updaters).
