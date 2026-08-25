====
Core
====

.. autosummary::
   :toctree: ../reference

   ~algan.errors
   ~algan.constants.color
   ~algan.constants.material_presets
   ~algan.constants.spatial
   ~algan.constants.math
   ~algan.constants.rate_funcs
   ~algan.geometry.geometry
   ~algan.logging.logger

Helper Methods
--------------

.. _reference-color-set-opacity:

.. py:method:: algan.constants.color.Color.set_opacity

   Return a copy of this color with the specified opacity.

.. _reference-color-add-defaults:

.. py:method:: algan.constants.color.Color.add_defaults

   Pad color channels to Algan's 5-channel RGBA-plus-glow representation.
