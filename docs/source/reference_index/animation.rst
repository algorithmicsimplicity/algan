=========
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

Runtime Mob Attributes
----------------------

These attributes are registered dynamically by
:class:`~algan.animatable_base.mob.Mob` on construction, so they do not appear
as ordinary class properties in Sphinx ``autoclass``. They are still part of the
public Mob API.

.. _reference-mob-color:

.. py:attribute:: algan.animatable_base.mob.Mob.color

   The Mob's color, including its red, green, blue, glow, and opacity channels.

.. _reference-mob-opacity:

.. py:attribute:: algan.animatable_base.mob.Mob.opacity

   The Mob's opacity multiplier (0.0 is invisible, 1.0 is opaque).

.. _reference-mob-glow:

.. py:attribute:: algan.animatable_base.mob.Mob.glow

   The Mob's glow intensity, consumed by the bloom post-processing pass.

.. _reference-mob-children:

.. py:attribute:: algan.animatable_base.mob.Mob.children

   The list of direct child Mobs attached to this Mob.

.. _reference-mob-parents:

.. py:attribute:: algan.animatable_base.mob.Mob.parents

   The list of Mobs this Mob is attached to as a child, and whose changes it
   therefore follows. A Mob may have several, and then accumulates all of them.

.. _reference-text-character-mobs:

.. py:attribute:: algan.mobs.text.Text.character_mobs

   The sequence of individual glyph Mobs making up this text.
