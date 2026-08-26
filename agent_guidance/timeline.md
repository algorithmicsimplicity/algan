# Timeline and animation

How recorded animation is stored, replayed and materialized. Read this before touching
`algan/animation_timeline/` or anything that records or materializes state.

## Animation and timeline architecture

The animation implementation lives under `../algan/animation_timeline` and the animatable/mob base implementation lives under `../algan/animatable_base`.

Each Scene's `TimelineManager` owns all recorded animation data for mobs in that Scene:

- one `AttributeTimeline` per animatable attribute;
- shared attribute buffers keyed by mob id and row ranges;
- timestamped attribute edit records;
- recorded animated-function applications;
- updater events and their dependency traces;
- mob lifespans represented as `[spawn, despawn)` intervals;
- materialization and replay state for a requested frame-time batch.

Mobs do not carry independent per-attribute animation histories. Their getters and setters read and write rows in their Scene's timeline.

### The function timeline

Alongside the attribute timelines, the `TimelineManager` holds `FunctionApplicationEvent`s recorded by the
`@animated_function` decorator, and `UpdaterEvent`s from `Mob.add_updater`.

### Animation contexts

`AnimationManager` owns the active context tree for one Scene. The main contexts are:

- `Seq`: sequential child animations;
- `Sync`: simultaneous child animations;
- `Lag(ratio)`: partially overlapping child animations;
- `Off`: instantaneous/non-animated changes;
- `Audio` and `Speech`: timing contexts that also register Scene-owned audio effects.

Mob methods decorated with `@animated_function` bind their Scene's `AnimationManager` automatically.

Critical timeline rule: events must be recorded against a context that is entered and exited. Context exit finalizes retroactively rescaled timestamps. Do not manually record events against the top-level context's raw timespan. `add_updater` and `remove_updater` demonstrate the correct pattern by opening an `Off(record_funcs=False, ...)` context.

The context classes live in `../algan/animation_timeline/animation_contexts.py`. `AnimationContext`s nest and inherit
unset parameters, and `run_time` rescales all child timestamps retroactively on `__exit__`.

**CRITICAL:** only `__exit__` syncs a context's rescaled timestamps, so events recorded against the top-level context
all evaluate to time 0. The `animated_function` wrapper enters a child context automatically; anything recording events
manually must wrap itself, e.g.:

```python
with Off(record_funcs=False) as context:
    ...
```

Timestamps are lazy because parent contexts can rescale child timing on exit. Treat an event's final start/end as unresolved until the relevant context tree has closed.

Overlapping edits to the same timeline rows are replayed in execution order using resolved replay windows. Do not simplify this to ordinary independent interpolation without preserving same-row overlap behavior.

### Attribute storage and materialization

One `AttributeTimeline` exists per animatable attribute (location, basis, color, opacity, ...): a shared `[1, N, W]` buffer of every mob's current values (each mob owns rows, keyed by its `id` in `mob_id_to_inds`) plus the log of timestamped edits to those rows (`EditRecord`: rows, pre-modification values, end time).

`set_state_to_times(times)` materializes all buffers at the requested frame times in one batched pass per attribute (`generate_array_states`, a flat `torch.searchsorted` over a per-row composite key on the animation device — deliberately **not** a Taichi kernel, which would stage the whole buffer through VRAM from the batch-prep worker), then re-executes recorded function applications with per-frame interpolated arguments, then applies updaters.

Edits of the same rows may overlap in time. `_resolve_replay_windows` extends each edit's effective end over the replay windows of earlier-executed edits that overlap it (transitively, unified per function application); the base state at time t is the pre-value of a row's earliest-executed edit still unfinished at t; and functions replay through their extended window (held at final parameters past their own end), so overlapping and same-end edits rematerialize in execution order.

### Lifespans

Every mob has a `Lifespan` — a `[spawn, despawn)` interval exposed as `Animatable.lifespan` and queried via `is_spawned()` / `is_despawned()`. Sub-mobs created by indexing (`mob[i]`) share their source's id and therefore its rows and lifespan; clones get a new id. Opacity is zeroed outside a mob's lifespan during materialization.

### Why `reset=False` is safe

`get_frames` calls `timeline_manager.clear_buffers()` when it finishes, returning `active_state` to `current_state`. That is what leaves the timeline queryable after a render, and what makes `save_video(reset=False)` — the default — non-destructive.

### Structural batch rewrites

Structural batch rewrites (e.g. `become`'s batch expansion) go through `_setattr_and_rebatch_without_record`, which re-allocates a mob's rows. Recorded history stays with the old rows, so this is only valid on mobs with fresh history (`detach_history` provides that).

## Audio

Audio is Scene-owned. `AudioManager` stores the Scene's speech source and transcript. `Audio`/`Speech` contexts add `AudioEffect` objects to the owning Scene and derive timing from that Scene's animation manager.

Do not add process-global transcript or speech-generator state. When constructing `Speech` or `Audio` contexts in low-level code, bind the relevant Scene animation manager explicitly.
