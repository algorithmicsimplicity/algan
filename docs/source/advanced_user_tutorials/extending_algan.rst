===============
Extending Algan
===============

Before diving into Algan's internals to create a new class from scratch, check
whether your goal can be achieved with existing extension hooks:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - What you want
     - What to use
   * - A custom mathematical animation
     - :func:`~.animated_function` (:doc:`custom_animations`)
   * - Continuous real-time behavior
     - Updaters (:doc:`../new_user_tutorials/updaters`)
   * - A new parametric 3-D surface
     - :class:`~.Surface` with a custom coordinate function
   * - Custom lighting / shading effects
     - Materials or fragment shader pipelines (:doc:`shaders_and_materials`)
   * - Custom post-processing image filters
     - Post-processing passes (:doc:`backgrounds_and_post_processing`)
   * - External 2-D/3-D assets
     - :class:`~.ManimMob` or :class:`~.Model3D`

Building Custom Mob Classes
===========================

If you need a new shape, subclass an existing Mob base class:

* **2-D Outlines & Shapes:** Subclass :class:`~.BezierCircuitCubic`. (See :class:`~.Circle` or :class:`~.Polygon` for reference implementations).
* **3-D Curved Surfaces:** Subclass :class:`~.Surface` and provide a parametric coordinate function ``(u, v) -> (x, y, z)``.
* **Polyhedral Meshes:** Subclass :class:`~.Polyhedron` and supply vertex and face arrays.

Registering Custom Animatable Attributes
----------------------------------------

If your custom Mob class introduces new attributes that should animate smoothly
over time on the timeline, register them during `__init__`:

.. code-block:: python

    class Ribbon(Surface):
        def __init__(self, twist=0.0, **kwargs):
            super().__init__(self._shape_func, **kwargs)
            # Register custom attribute with the timeline engine
            self.register_attrs_as_animatable(["twist"])
            self.twist = twist

.. important::

    Always call ``self.register_attrs_as_animatable(["attr_name"])`` **before**
    assigning the initial value in ``__init__``.

Batching Large Numbers of Mobs
==============================

If your scene features hundreds or thousands of identical shapes (like a point
cloud or a large particle grid), don't create thousands of separate Mob
instances. Instead, pack them into a single batched Mob:

.. algan:: ExtendingPackedSpheres

    from algan import *
    import torch

    grid = torch.arange(12) * 0.5 - 2.75
    centers = torch.stack(torch.meshgrid(grid, grid, indexing='ij'), -1)
    centers = torch.cat((centers.reshape(-1, 2),
                         torch.zeros(centers.numel() // 2, 1)), -1)

    # Builds 144 spheres in a single packed batch
    spheres = Sphere.from_batches(centers, radius=0.15, color=BLUE).spawn()

    spheres.move(UP * 0.5)   # Moves all 144 spheres at once
    spheres[7].move(OUT)     # Moves an individual sphere view

    Scene.save_video()

This stores all items in a single contiguous GPU buffer, keeping rendering
memory and CPU timeline overhead minimal.

Contributing Extensions
=======================

If you create new shapes or animations that would benefit other users, we'd love
to review your PR! Check out :doc:`../contributing` and
:doc:`../contributing/development` for development setup and testing guidelines.

See Also
========

* :doc:`custom_animations` -- creating animated functions with ``@animated_function``.
* :doc:`shaders_and_materials` -- custom shader stages and material models.
* :doc:`../developer_tutorials/overview_internals` -- Algan's internal architecture.
* :doc:`../contributing` -- contributing guidelines.
