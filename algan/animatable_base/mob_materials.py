"""Shader / material API for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobMaterialsMixin` is mixed
into ``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import inspect

from algan.rendering.shaders.pbr_shaders import default_shader
from algan.utils.tensor_utils import cast_to_tensor


class ModifiedProtectedAttributeError(Exception):
    """Raised when a shader/material is (re)assigned after the mob has spawned."""


class MobMaterialsMixin:
    """``set_shader`` / ``set_fragment_shader`` / ``set_material`` -- all must
    be called *before* the mob is spawned."""

    def set_shader(self, shader):
        """Sets the shader for this mob and all of its descendants. This MUST
        be called before the mob is spawned, the shader cannot be changed after spawn.
        If you need to change the shader for a spawned mob, create a new clone of it and
        despawn the original e.g.:
        with Off():
            new_mob = mob.clone(spawn=False)
            new_mob.set_shader(new_shader)
            mob.despawn()
            new_mob.spawn()
            mob = new_mob

        Parameters
        ----------
        shader
            The function to use for shading at render time.

        Returns
        -------
        :class:`~.Mob`
            The mob instance itself, allowing for method chaining.

        Raises
        ------
        :class:`.ModifiedProtectedAttributeError`
            If set_shader is used on an already spawned mob.

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
            d.set_non_recursive(**{k: v for k, v in zip(shader_specific_param_names, shader_specific_param_defaults)})
            #for n, v in zip(
            #
            #):
            #    d.__setattr__(n, v)
            d.shader = shader
            d.shader_specific_param_names = shader_specific_param_names
        return self

    def set_fragment_shader(self, shader):
        """Sets a custom **fragment shader** for this mob and its descendants,
        evaluated per fragment inside the deterministic ray tracer's shade
        kernel (rather than per vertex like :meth:`set_shader`). MUST be called
        before the mob is spawned.

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

        Parameters
        ----------
        shader
            A fragment stage, a built-in material shader, or a list of these
            (a pipeline). ``None`` clears the fragment shader.

        Returns
        -------
        :class:`~.Mob`
            The mob instance itself, allowing for method chaining.

        Raises
        ------
        :class:`.ModifiedProtectedAttributeError`
            If called on an already spawned mob.
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

    def set_material(self, material):
        """Applies a Three.js-style :class:`~algan.rendering.shaders.materials.Material`
        to this mob and all of its descendants.

        This configures the material's lighting shader (via :meth:`set_shader`)
        and copies its properties onto the mob: numeric/colour material
        properties become animatable attributes (e.g. ``mob.roughness``,
        ``mob.emissive_intensity``), the material colour drives the mob's base
        colour, and ``opacity`` drives its max opacity. Like :meth:`set_shader`,
        this MUST be called before the mob is spawned.

        The material is also the sole public source of ray-transport
        properties. ``metalness`` and ``roughness`` drive reflections, while a
        transmissive :class:`MeshPhysicalMaterial` supplies ``ior`` for
        refraction. This mirrors the Three.js material workflow; there are no
        separate mob-level reflectivity, roughness, or refractive-index setters.

        Parameters
        ----------
        material
            A :class:`~algan.rendering.shaders.materials.Material` instance, e.g.
            ``MeshStandardMaterial(metalness=1.0, roughness=0.2)``.

        Returns
        -------
        :class:`~.Mob`
            The mob instance itself, allowing for method chaining.

        Raises
        ------
        :class:`.ModifiedProtectedAttributeError`
            If used on an already spawned mob.
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

    def get_shader_params(self):
        if hasattr(self, "shader_specific_param_names"):
            return {
                _: self.__getattribute__(_) for _ in self.shader_specific_param_names
            }
        return dict()

