"""Shader / material API for
:class:`~algan.animatable_base.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobMaterialsMixin` is mixed
into ``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from algan.rendering.shaders.pbr_shaders import default_shader
from algan.utils.tensor_utils import cast_to_tensor

if TYPE_CHECKING:
    from algan.animatable_base.mob import Mob


class ModifiedProtectedAttributeError(Exception):
    """Raised when a shader/material is (re)assigned after the mob has spawned."""


class MobMaterialsMixin:
    """``set_shader`` / ``set_fragment_shader`` / ``set_material`` -- all must
    be called *before* the mob is spawned.
    """

    def set_shader(self, shader) -> Mob:
        """Set the per-vertex lighting shader for this Mob and its descendants.

        The shader decides how the Mob responds to light. Its parameters become
        animatable attributes on the Mob, so a shader with a ``roughness``
        parameter gives you ``mob.roughness`` to animate. Most scenes should set a
        :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`
        instead, which picks the matching shader and
        fills in its values.

        Animation
        ---------
        Not animated, and **must be called before the Mob is spawned** -- the
        shader cannot be changed afterwards. To re-shade something already on
        screen, swap in a fresh clone:

        .. code-block:: python

            with Off():
                new_mob = mob.clone(spawn=False)
                new_mob.set_shader(new_shader)
                mob.despawn()
                new_mob.spawn()
                mob = new_mob

        Parameters
        ----------
        shader
            Shading function used at render time, e.g. ``phong_shader`` or
            ``standard_shader``. ``None`` clears the shader, leaving the Mob
            unlit.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`.ModifiedProtectedAttributeError`
            If called on a Mob that has already been spawned.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`
            Three.js-style materials, the usual entry point.
        :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`
            Shade per fragment instead of per vertex.
        """
        if self.is_spawned():
            raise ModifiedProtectedAttributeError(
                "You are attempting to change the shader "
                "of a mob that is already spawned. This is not allowed. "
                "See docs for help."
            )

        if shader is None:
            for d in reversed(self.get_descendants()):
                d.shader = shader
            return self

        shader_params = inspect.signature(shader).parameters
        num_shader_independent_params = len(
            inspect.signature(default_shader).parameters.keys()
        )
        shader_specific_param_names = list(shader_params.keys())[
            num_shader_independent_params:
        ]
        shader_specific_param_defaults = [
            shader_params[n].default
            if shader_params[n].default is not inspect._empty
            else 0
            for n in shader_specific_param_names
        ]

        for d in reversed(self.get_descendants()):
            d.register_attrs_as_animatable(shader_specific_param_names)
            d.set_non_recursive(**dict(zip(shader_specific_param_names, shader_specific_param_defaults)))
            #for n, v in zip(
            #
            #):
            #    d.__setattr__(n, v)
            d.shader = shader
            d.shader_specific_param_names = shader_specific_param_names
        return self

    def set_fragment_shader(self, shader) -> Mob:
        """Set a per-fragment shader for this Mob and its descendants.

        Shading runs once per rendered fragment rather than once per vertex as
        with
        :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader`,
        so effects can vary smoothly across a
        surface instead of being interpolated between its corners -- at a higher
        render cost.

        ``shader`` is a
        :class:`~algan.rendering.shaders.fragment_shaders.FragmentStage`
        (a Taichi ``@ti.func`` stage plus its parameter specs), a built-in
        material shader function (e.g. ``phong_shader``), or a **list** of
        these forming a *pipeline* run left-to-right -- each stage receives the
        previous stage's output colour. For example
        ``mob.set_fragment_shader([cosine_color, phong_shader])`` recolours each
        fragment with a cosine wave and then lights the result with Blinn-Phong.

        The stages' parameters become animatable attributes (duplicate names
        across stages are suffixed). Setting a fragment shader forces the
        deterministic renderer's per-fragment path on for any scene the mob
        appears in; it is ignored by the Monte Carlo / physical path tracer.

        Animation
        ---------
        Not animated, and **must be called before the Mob is spawned**. The stage
        parameters it registers *are* animatable afterwards, so
        ``mob.wave_speed = 3`` animates like any other attribute.

        Parameters
        ----------
        shader
            A fragment stage, a built-in material shader, or a list of these
            forming a pipeline. ``None`` clears the fragment shader.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`.ModifiedProtectedAttributeError`
            If called on a Mob that has already been spawned.
        """
        if self.is_spawned():
            raise ModifiedProtectedAttributeError(
                "You are attempting to change the fragment shader of a mob that "
                "is already spawned. This is not allowed. See docs for help.")

        if shader is None:
            for d in reversed(self.get_descendants()):
                d.shader = None
            return self

        from algan.rendering.shaders.fragment_shaders import (
            build_fragment_pipeline,
        )

        marker, param_specs = build_fragment_pipeline(shader)
        names = [n for n, _d in param_specs]
        for d in reversed(self.get_descendants()):
            d.register_attrs_as_animatable(names)
            for n, default in param_specs:
                d.__setattr__(n, default)
            d.shader_specific_param_names = names
            d.shader = marker
        return self

    def set_material(self, material) -> Mob:
        """Give this Mob a material, deciding how it looks under light.

        The Three.js-style entry point to appearance, and the one to reach for
        first: the material picks the lighting shader and fills in its values, so
        ``MeshStandardMaterial(metalness=1.0, roughness=0.2)`` gives you polished
        metal without touching a shader directly. Applies to this Mob and all its
        descendants.

        The material's numeric and colour properties land on the Mob as animatable
        attributes -- ``mob.roughness``, ``mob.emissive_intensity`` and so on -- so
        they can be animated afterwards like any other. Its colour drives the Mob's
        base colour and its ``opacity`` the Mob's maximum opacity.

        The material is also the sole public source of ray-transport
        properties. ``metalness`` and ``roughness`` drive reflections, while a
        transmissive
        :class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` supplies
        ``ior`` for
        refraction. This mirrors the Three.js material workflow; there are no
        separate mob-level reflectivity, roughness, or refractive-index setters.

        Animation
        ---------
        Not animated, and **must be called before the Mob is spawned**, since it
        sets the shader. The properties it installs are animatable from then on.

        Parameters
        ----------
        material
            A :class:`~algan.rendering.shaders.materials.Material` instance, e.g.
            ``MeshStandardMaterial(metalness=1.0, roughness=0.2)``. A material
            built with the default ``color=None`` leaves the Mob's own colour
            alone, so material and colour can be chosen independently.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`.ModifiedProtectedAttributeError`
            If called on a Mob that has already been spawned.

        Examples
        --------
        .. algan:: Example1MobSetMaterial

            from algan import *

            sphere = Sphere(color=BLUE)
            sphere.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.15))
            sphere.spawn()
            sphere.rotate(180, UP)

            Scene.save_video()
        """
        from algan.rendering.shaders.materials import _to_color5

        if self.is_spawned():
            raise ModifiedProtectedAttributeError(
                "You are attempting to set the material "
                "of a mob that is already spawned. This is not allowed. "
                "See docs for help."
            )

        # Register the lighting shader and its animatable parameters, then
        # override the signature defaults with this material's values.
        self.set_shader(material.shader)
        params = material.get_shader_param_values()
        # ``color=None`` (the default) means the material does not repaint the
        # mob -- only an explicitly supplied material colour overrides it.
        color5 = (_to_color5(material.color)
                  if material.applies_color and material.color is not None
                  else None)
        for d in reversed(self.get_descendants()):
            d.set_non_recursive(**params)
            #for name, value in params.items():
            #    d.__setattr__(name, value)
            if color5 is not None:
                d.color = color5
            d.opacity = cast_to_tensor(material.opacity)
            d.material = material

        material.emit_warnings()

        return self

    def get_shader_params(self) -> dict:
        """Get this Mob's current shader parameter values, by name.

        The parameters a shader or material installed as animatable attributes,
        e.g. ``{"roughness": ..., "metalness": ...}``. Useful for inspecting what a
        material actually set, or copying it onto another Mob.

        Returns
        -------
        dict
            Parameter names mapped to their current values; empty if the Mob has
            no shader-specific parameters.
        """
        if hasattr(self, "shader_specific_param_names"):
            return {
                _: self.__getattribute__(_) for _ in self.shader_specific_param_names
            }
        return {}

