=========
Rendering
=========

.. currentmodule:: algan

.. autosummary::
   :toctree: ../reference

   ~rendering.camera
   ~rendering.lights
   ~rendering.shaders.materials
   ~rendering.shaders.material_shaders
   ~rendering.shaders.pbr_shaders
   ~rendering.shaders.fragment_shaders
   ~rendering.raytracing.tracer
   ~rendering.raytracing
   ~rendering.raytracing.settings
   ~rendering.raytracing.truncation
   ~rendering.denoise
   ~rendering.primitives.primitive
   ~rendering.primitives.triangle_primitive
   ~rendering.primitives.bezier_circuit_primitive
   ~rendering.post_processing
   ~rendering.memory_model

Runtime Light Attributes
------------------------

These attributes are registered dynamically by
:class:`~algan.rendering.lights.Light` during construction, so they do not
appear as ordinary class properties in ``autoclass``. They are full parts of
the public Light API.

.. py:attribute:: algan.rendering.lights.Light.intensity

   The light's brightness: a dimensionless multiplier applied to the light's
   ``color`` every frame. Must be a finite number of at least ``0.0``;
   defaults to ``1.0``. Animatable like any Mob attribute.
