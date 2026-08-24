Animation
=========

.. autosummary::
   :toctree: ../reference

   ~algan.animatable_base.animatable
   ~algan.animatable_base.mob
   ~algan.animatable_base.mob_hierarchy
   ~algan.animatable_base.mob_orientation
   ~algan.animatable_base.mob_movement
   ~algan.animatable_base.mob_layout
   ~algan.animatable_base.mob_morph
   ~algan.animatable_base.mob_materials
   ~algan.animation_timeline.animation_contexts
   ~algan.animation_timeline.timeline
   ~algan.animations.manim_animations
   ~algan.animations.movement
   ~algan.animations.changing
   ~algan.animations.indication

Runtime Mob attributes
----------------------

These attributes are registered dynamically by
:class:`~algan.animatable_base.mob.Mob` during
construction, so they do not appear as ordinary class properties in
``autoclass``. They are still part of the public Mob API.

.. _reference-mob-color:

.. py:attribute:: algan.animatable_base.mob.Mob.color

   The Mob's color, including its red, green, blue, glow, and opacity channels.

.. _reference-mob-opacity:

.. py:attribute:: algan.animatable_base.mob.Mob.opacity

   The Mob's independent opacity multiplier.

.. _reference-mob-glow:

.. py:attribute:: algan.animatable_base.mob.Mob.glow

   The Mob's glow intensity, consumed by the bloom post-processing pass.

.. _reference-mob-children:

.. py:attribute:: algan.animatable_base.mob.Mob.children

   The live list of child Mobs in the hierarchy.

.. _reference-text-character-mobs:

.. py:attribute:: algan.mobs.text.Text.character_mobs

   The lazy indexed sequence of individual glyph Mobs.
