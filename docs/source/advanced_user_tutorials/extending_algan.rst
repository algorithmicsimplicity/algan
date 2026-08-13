===============
Extending Algan
===============

Most things you might want are reachable without touching Algan's internals. Before
writing a new class, check whether one of these does the job:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - You want
     - Use
   * - A new animation
     - :func:`~.animated_function`, or a composition of existing ones
       (:doc:`../new_user_tutorials/basic_animations`)
   * - A rule that holds continuously
     - An updater (:doc:`../new_user_tutorials/updaters`)
   * - A new shape
     - :class:`~.Surface` with your own coordinate function, or
       :class:`~.Polygon` / :class:`~.Polyhedron` with your own vertices
   * - A new look under light
     - A material, a vertex shader, or a fragment-shader pipeline
       (:doc:`shaders_and_materials`)
   * - New ray-continuation behaviour
     - A custom scatter function on a fragment stage
       (:doc:`shaders_and_materials`)
   * - A new full-frame effect
     - A post-process pass
       (:doc:`backgrounds_and_post_processing`)
   * - Geometry from another library
     - :class:`~.ManimMob` or :class:`~.ThreeDModelMob`

What is left after that is genuinely new *geometry* -- a shape that cannot be built
out of triangles or cubic Bezier curves -- which is what this page is about.

How Rendering Reaches Your Mob
==============================

Two things have to be true for a Mob to appear on screen.

**It must be registered with the Scene.** The render loop iterates over the Scene's
actors; it does not walk the Mob hierarchy looking for things to draw. Ordinary Mob
construction registers the Mob automatically. If you construct something with
``add_to_scene=False`` -- as internal machinery sometimes does for intermediate
objects -- it will never render, no matter how correct its geometry is or whether you
spawned it.

**It must produce render primitives.** A renderable Mob defines
``get_render_primitives()``, returning geometry in one of the forms the renderer
understands:

* **Flat triangles** -- the general case, and what 3-D shapes and imported models
  reduce to.
* **PN (curved) triangles** -- triangles with per-vertex normals that are diced
  adaptively at render time, so a :class:`~.Surface` stays smooth as the camera moves
  in.
* **Cubic Bezier circuits** -- closed outlines with exact curved edges, used for 2-D
  shapes, text and LaTeX.

Each batch of frames is assembled by packing every primitive into contiguous
per-geometry-type arrays and building a spatio-temporal bounding volume hierarchy
over them, covering all the frames in the batch at once. That is why moving geometry
costs more than static geometry, and why the classes in
``algan/rendering/primitives/`` are about primitive construction and batching rather
than being a separate renderer.

Adding a Mob Class
==================

The usual case is a new Mob built from existing primitives -- which is most new
shapes. Subclass the Mob type whose geometry you want and supply your own points:

* For a 2-D outline, subclass :class:`~.BezierCircuitCubic`. Look at
  :class:`~.Circle` and :class:`~.Polygon` for the pattern.
* For a curved 3-D surface, subclass :class:`~.Surface` and pass a coordinate
  function. :class:`~.Sphere` and :class:`~.Cylinder` are both a few lines each.
* For an explicit mesh, look at :class:`~.Polyhedron`.

If your class has properties of its own that should be animatable, register them:

.. code-block:: python

    class Ribbon(Surface):
        def __init__(self, twist=0.0, **kwargs):
            super().__init__(self._shape, **kwargs)
            self.register_attrs_as_animatable(["twist"])
            self.twist = twist

``register_attrs_as_animatable`` is what gives an attribute the interpolation,
recording and per-frame materialization every built-in attribute has -- after that,
``ribbon.twist = 2.0`` animates like ``ribbon.color`` does.
:class:`~.BezierCircuitCubic` does exactly this for ``border_width`` and
``border_color``.

.. important::

    Register the attributes **before** assigning them. Assigning first stores a plain
    Python value that the timeline knows nothing about.

Helpers Outside the Star Import
===============================

``from algan import *`` is deliberately curated: it carries the names you need to
author a scene, and leaves out lower-level helpers that would otherwise spend a name
in every user's namespace. Those helpers are still public and still supported --
import them from the module that defines them:

.. code-block:: python

    from algan.geometry.geometry import (
        get_orthonormal_vector,
        get_rotation_around_axis,
        get_rotation_between_bases,
        map_global_to_local_coords,
        map_local_to_global_coords,
        project_onto_basis,
        rotate_vector_around_axis,
    )
    from algan.utils.animation_utils import animate_lagged_by_location
    from algan.utils.mob_utils import batch_mobs

These are the ones worth knowing about when writing a custom animation or Mob:
the ``algan.geometry.geometry`` functions convert between world and Mob-local
coordinate frames and build rotation matrices, ``animate_lagged_by_location``
staggers an animation across a batch by where each element sits in space, and
``batch_mobs`` packs several Mobs into one batched Mob so they animate as a single
recorded operation.

Adding a Render Primitive
=========================

If your geometry genuinely cannot be expressed as triangles or cubic Bezier circuits,
you need a new primitive type, which means touching the renderer. This is a
substantial change rather than a subclass: a primitive type has to be packed into the
merged per-batch arrays, given a bounding volume hierarchy, and handled in the Taichi
traversal, shading and shadow kernels -- each of which is compile-time specialised on
which geometry types are present.

Start by reading, in this order:

1. ``algan/rendering/raytracing/primitives.py`` -- how triangle and Bezier circuit
   primitives declare their per-vertex and per-surface data.
2. ``algan/rendering/raytracing/scene_builder.py`` -- how a batch's primitives are
   packed and how the hierarchies are built.
3. ``algan/rendering/raytracing/tracer.py`` -- the entry point and how it dispatches
   between the deterministic wavefront pipeline and the Monte Carlo path tracer.

Then see the developer documentation: :doc:`../developer_tutorials/index` covers the
internals, and ``AGENTS_DETAILED.md`` in the repository is the detailed architecture
and contract reference.

Working on Algan's Kernels
==========================

If you do end up in the Taichi kernels, three things will cost you time if you do not
know them:

* **The offline kernel cache does not invalidate on ``@ti.func`` edits.** Clear it
  with ``clear_cache(taichi_kernels=True)`` before benchmarking or A/B-testing a
  kernel change, or you will be measuring the old kernel.
* **Never edit a ``*_taichi.py`` file while a render process or warm daemon is
  running.** The JIT reads sources at first launch and can compile half-edited code.
* **Kernel files must keep the ``_taichi`` filename suffix.** Ruff is configured to
  leave those files alone, because the ``from __future__ import annotations`` it
  would otherwise insert breaks Taichi kernel compilation. Note also that a plain
  ``ruff check`` rewrites files in this repository -- use ``ruff check --no-fix``
  unless you mean to apply fixes.

Contributing
============

If you build something generally useful, contributions are welcome -- see
:doc:`../contributing`. Two expectations worth knowing up front:

* **Optimizations are held to byte-identical output**, validated by an A/B parity
  script, and new behaviour goes behind a setting so the default path stays identical.
* **Rendering changes are validated pixel-wise** against checked-in expected outputs,
  so a change that legitimately alters output comes with re-baselined videos.

See Also
========

- :doc:`shaders_and_materials` -- the extension points most "I need custom
  rendering" problems actually want.
- :doc:`../developer_tutorials/index` -- Algan's internals.
- :doc:`performance_and_quality` -- the constraints any renderer change is judged
  against.
