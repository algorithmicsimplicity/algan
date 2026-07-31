================================
An Overview of Algan's Internals
================================

Algan is a lazy animation system coupled to a batched 3-D raytrace renderer. Authoring
code records changes rather than evaluating every frame immediately. Rendering
later materializes the owning Scene's timeline at a batch of frame times,
converts the resulting mob states into render primitives, and sends those
primitives through the raytracer to be rendered.

Scene containment
=================

The central architectural unit is :class:`~algan.scene.Scene`. A Scene owns:

* its actor registry, camera, lights, effects, and environment map;
* one :class:`~algan.animation_timeline.timeline.TimelineManager`;
* one :class:`~algan.animation_timeline.animation_contexts.AnimationManager`;
* one :class:`~algan.sound.audio_effect.AudioManager`;
* video settings, render memory, and the frame/render loop.

These managers are regular instances. They are recreated when that Scene is
reset. :class:`~algan.scene_manager.SceneManager` is the only singleton and has
a narrower job: maintain the stack of active Scenes. Its ``current_scene`` is
used only when authoring code omits an explicit owner.

A new Animatable resolves its Scene once in
``Animatable.__init__``. Its ID, timeline rows, lifespan, and animation manager
all come from that Scene. Existing mobs never consult the global active Scene
for subsequent animation operations.

The animation system
====================

Every Scene's :class:`~algan.animation_timeline.timeline.AnimationTimeline`
contains shared attribute buffers. A mob owns rows in each relevant buffer,
keyed by its Scene-local ID. Attribute modifications record edit history for
those rows; animated-function calls record function-application events; spawn
and despawn operations record the mob's lifespan.

:class:`~algan.animation_timeline.animation_contexts.AnimationContext` objects
control event timing. ``Seq``, ``Sync``, ``Lag``, and ``Off`` are not global
contexts: each belongs to one Scene's AnimationManager. Implicit contexts use
the active Scene manager, while methods on initialized Animatables bind their
owning manager through a context-variable override.

A context cannot combine managers from multiple Scenes. Hierarchy and Group
mutations enforce the same ownership boundary.

At materialization time,
``AnimationTimeline.set_state_to_times(times)`` reconstructs every attribute
buffer for the requested frame times, replays animated functions with
interpolated arguments, and runs active updaters. Materialization is batched
according to the animation-memory budget in ``SETTINGS.computing``.

Rendering entry points
======================

The public APIs are :meth:`~algan.scene.Scene.save_frame` and
:meth:`~algan.scene.Scene.save_video`.

``save_frame`` resolves the output path, optionally installs temporary video
settings, materializes one or more timestamps, writes PNG images, and restores
all derived render state. It does not reset the Scene.

``save_video`` temporarily activates the target Scene and binds its
AnimationManager before delegating to ``_render_scene_to_file``, which carries
the implementation while ``Scene.save_video`` carries the user-facing signature
and documentation. It renders audio, streams video frames, and returns a
``RenderResult``. Preflight failures and ``overwrite=False`` skips are
observational and preserve authored state.

By default (``reset=False``) the Scene is left exactly as authored, so mobs
stay valid and the timeline can keep growing. Two pieces of finalization are
therefore conditional:

* the end-of-scene despawn of every actor runs when a fade-out was requested,
  or when ``reset=True``;
* ``render_to_video`` closes the camera and light lifespans only when the Scene
  is being finalized (``despawn_camera_and_lights``).

Both lifespans extend past the last rendered frame index either way, so output
is unaffected by the choice. With ``reset=True`` the Scene's timeline,
animation and audio managers are rebuilt in a ``finally`` block on both success
and failure, and mobs created before the render must not be reused.

Frame batches and scene assembly
================================

``RenderLoopMixin.get_frames`` divides the Scene timeline into windows sized by
animation and rendering memory. For each window it materializes actors, builds
render primitives, snapshots camera/light/environment state, and prepares the
next batch concurrently unless prefetching is disabled.

Scene assembly merges each geometry class into contiguous tensor arrays and
builds acceleration structures spanning the batch's frame interval. Render
out-of-memory errors reduce the frame window and retry rather than permanently
changing public video settings.

Renderer paths
==============

The primary entry point is ``render_batch_raytraced``. The renderer selects a
path from the live ``SETTINGS.raytracing`` configuration and scene features:

* the deterministic wavefront path for single-sample rendering and features
  such as deterministic reflection/refraction;
* the Monte Carlo path tracer for multi-sample stochastic rendering where its
  capability set is sufficient;
* the hybrid primary-raster path when enabled, with continuation rays handled
  by the wavefront machinery.

Renderer kernels and helper functions live in ``*_taichi.py`` modules. Keep
those filenames because lint tooling excludes them from transformations that
can break Taichi compilation.

Settings lifecycle
==================

:doc:`SETTINGS <../reference_index/settings>` contains runtime-adjustable sections with stable object
identity. Mutable sections update in place through ``set``; immutable presets
return modified copies. Initialization-only device and kernel-layout choices
live in ``algan.settings._startup`` and are selected through environment
variables before importing Algan.

Internal code should read settings through ``SETTINGS`` at use time. Do not
capture mutable settings fields into module-level value imports unless the
value is intentionally fixed for the process lifetime.
