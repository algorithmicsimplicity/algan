"""Shader / material API for
:class:`~algan.animatable_base.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobMaterialsMixin` is mixed
into ``Mob`` and is not useful standalone (``self`` is always a Mob).
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING

from algan.rendering.shaders.material_shaders import SHADER_FIXED_PARAM_COUNT
from algan.utils.tensor_utils import cast_to_tensor

if TYPE_CHECKING:
    from algan.animatable_base.mob import Mob

from algan.errors import (
    AlganConfigurationError,
    ModifiedProtectedAttributeError,
    UnsupportedFeatureWarning,
    _user_stacklevel,
)


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
            mob_name = self._describe()
            raise ModifiedProtectedAttributeError(
                f"Cannot change shader on {mob_name} because it has already been spawned. "
                "Shaders and materials must be configured before calling .spawn(). "
                "To change shaders after spawning, create an unspawned clone with `with Off(): clone = mob.clone(spawn=False)`, "
                "configure its shader, despawn the old mob, and spawn the new one."
            )

        if shader is None:
            for d in reversed(self.get_descendants()):
                d.shader = shader
            return self

        shader_params = inspect.signature(shader).parameters
        num_shader_independent_params = SHADER_FIXED_PARAM_COUNT
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
            d.set_non_recursive(
                **dict(zip(shader_specific_param_names, shader_specific_param_defaults))
            )
            # for n, v in zip(
            #
            # ):
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
        previous stage's output color. For example
        ``mob.set_fragment_shader([cosine_color, phong_shader])`` recolors each
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
            mob_name = self._describe()
            raise ModifiedProtectedAttributeError(
                f"Cannot change fragment shader on {mob_name} because it has already been spawned. "
                "Shaders must be configured before calling .spawn(). "
                "To change fragment shaders after spawning, create an unspawned clone with `with Off(): clone = mob.clone(spawn=False)`, "
                "configure its fragment shader, despawn the old mob, and spawn the new one."
            )

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

        The material's numeric and color properties land on the Mob as animatable
        attributes -- ``mob.roughness``, ``mob.emissive_intensity`` and so on -- so
        they can be animated afterwards like any other. Its color drives the Mob's
        base color and its ``opacity`` the Mob's maximum opacity.

        The material is also the sole public source of ray-transport
        properties. ``metalness`` and ``roughness`` drive reflections, while a
        transmissive
        :class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` supplies
        ``ior`` for
        refraction. This mirrors the Three.js material workflow; there are no
        separate mob-level reflectivity, roughness, or refractive-index setters.

        Its texture maps are forwarded onto the geometry, which is what samples
        them: ``map``, ``normal_map``, ``roughness_map`` and ``metalness_map``
        each take a file path or an ``[H, W, C]`` image and are sampled
        bilinearly per fragment. That needs per-vertex UVs, so it reaches a
        :class:`~algan.mobs.surfaces.surface.Surface` (a
        :class:`~algan.mobs.shapes_3d.Sphere`, :class:`~algan.mobs.shapes_3d.Cylinder`, :class:`~.ImageMob`, ...) or a
        :class:`~algan.mobs.three_d_models.mesh.TriangleMesh` built with
        ``uvs``; on anything else the maps are ignored, with a warning. A
        forwarded map is **static** -- unlike the scalar properties above it is
        not an animatable attribute -- except ``map`` on a Surface, which lands
        on the animatable
        :attr:`~algan.mobs.surfaces.surface.Surface.color_texture`.

        Every built-in material class shades per fragment in the render kernel,
        so it sees every light type, receives shadows, and its look no longer
        depends on the mesh's tessellation. Only a *custom* per-vertex shader
        (``set_shader`` with a plain function) is baked into vertex colors
        before the frame renders -- lit only by a plain :class:`~.PointLight`
        and never receiving shadows. Applying one under a lighting rig that
        asks for more than that warns, rather than quietly dropping the
        difference.

        Animation
        ---------
        Not animated, and **must be called before the Mob is spawned**, since it
        sets the shader. The properties it installs are animatable from then on.

        Parameters
        ----------
        material
            A :class:`~algan.rendering.shaders.materials.Material` instance, e.g.
            ``MeshStandardMaterial(metalness=1.0, roughness=0.2)``. A material
            built with the default ``color=None`` leaves the Mob's own color
            alone, so material and color can be chosen independently.

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
        from algan.rendering.shaders.materials import Material, _to_color5

        if not isinstance(material, Material):
            # Reaching for ``set_material(GOLD)`` is a natural mistake: CHROME
            # and COPPER are materials while GOLD is a color. Unchecked, the
            # first thing this method touched was ``material.shader``, so the
            # answer was an AttributeError about a missing attribute on
            # whatever was passed.
            from algan.constants import material_presets

            presets = ", ".join(sorted(material_presets.__all__))
            raise AlganConfigurationError(
                f"set_material() expects a Material, got "
                f"{type(material).__name__}. Use one of the material classes "
                f"-- MeshBasicMaterial, MeshStandardMaterial, "
                f"MeshPhysicalMaterial and the rest -- or a preset "
                f"({presets}). To change only the color, set "
                f"mob.color instead."
            )

        if self.is_spawned():
            mob_name = self._describe()
            raise ModifiedProtectedAttributeError(
                f"Cannot set material on {mob_name} because it has already been spawned. "
                "Materials must be configured before calling .spawn(). "
                "To change materials after spawning, create an unspawned clone with `with Off(): clone = mob.clone(spawn=False)`, "
                "configure its material, despawn the old mob, and spawn the new one."
            )

        # Register the lighting shader and its animatable parameters, then
        # override the signature defaults with this material's values.
        self.set_shader(material.shader)
        params = material.get_shader_param_values()
        # ``color=None`` (the default) means the material does not repaint the
        # mob -- only an explicitly supplied material color overrides it.
        color5 = (
            _to_color5(material.color)
            if material.applies_color and material.color is not None
            else None
        )
        for d in reversed(self.get_descendants()):
            d.set_non_recursive(**params)
            # for name, value in params.items():
            #    d.__setattr__(name, value)
            if color5 is not None:
                d.color = color5
            d.opacity = cast_to_tensor(material.opacity)
            d.material = material

        material.emit_warnings(self._forward_material_textures(material))
        self._warn_lighting_beyond_vertex_bake(material)

        return self

    def _forward_material_textures(self, material):
        """Hand ``material``'s image slots to whichever descendants can sample
        them, and report back what landed.

        Returns ``{slot_name: is_animatable}`` over every descendant that took
        something -- what :meth:`Material.emit_warnings` needs to tell a map
        that is sampled from one that is dropped. A slot counts as animatable
        only where *every* geometry that took it made it animatable, so the
        caution is never understated.

        Run after the loop above, so a forwarded ``map`` wins over the
        material's flat ``color``: the kernel's color sampler replaces the
        per-vertex color rather than modulating it.
        """
        from algan.rendering.shaders.materials import _normalize_forwarded_maps

        if not material._textures:
            return {}
        targets, uncovered = self._texture_forwarding_targets()
        # All-or-nothing, because a partial application is the confusing case:
        # a Cube's body is TriangleVertices, which cannot be textured, while
        # the decorative Dot3D at each of its corners is a Sphere, which can.
        # Painting the corner dots and calling the map delivered would be worse
        # than saying it went nowhere.
        if uncovered or not targets:
            return {}
        # Decoded once here rather than inside each target: a Group of ten
        # Spheres would otherwise re-read the same file ten times.
        maps = _normalize_forwarded_maps(material._textures)
        if not maps:
            return {}
        forwarded = {}
        for d in targets:
            for slot, animatable in d._accept_material_textures(maps).items():
                forwarded[slot] = forwarded.get(slot, True) and animatable
        return forwarded

    def _texture_forwarding_targets(self):
        """The Mobs under this one that a material's texture maps should go to.

        Returns ``(targets, uncovered)``. A Mob that can take the maps owns its
        whole subtree, so the walk stops there -- a Surface's own grid child
        renders but is never a target of its own. ``uncovered`` is True when
        the walk passed a Mob that *renders* (``is_primitive``) and cannot take
        them, which is what separates a Group of Spheres (every rendering part
        textured) from a Cube (its faces cannot be, whatever its corner dots
        can do).
        """
        targets, uncovered, stack = [], False, [self]
        while stack:
            mob = stack.pop()
            if mob._can_accept_material_textures():
                targets.append(mob)
                continue
            if getattr(mob, "is_primitive", False):
                uncovered = True
            stack.extend(mob.children or ())
        return targets, uncovered

    def _can_accept_material_textures(self):
        """Whether :meth:`_accept_material_textures` would take anything.

        Asked before any image is loaded, so the decision costs nothing on the
        Mobs that cannot be textured -- which is most of them.
        """
        return False

    def _accept_material_textures(self, maps):
        """Take the texture maps of a material being applied to this Mob.

        The geometry, not the material, owns Algan's texture pipeline: the maps
        are sampled against per-vertex UVs in the trace kernel, and only a Mob
        that carries UVs can serve them. Overridden by
        :class:`~algan.mobs.surfaces.surface.Surface` and
        :class:`~algan.mobs.three_d_models.mesh.TriangleMesh`; every other Mob
        takes nothing, which is what makes ``set_material`` warn that the maps
        are ignored.

        Parameters
        ----------
        maps
            The forwardable slots, already decoded into the engine's texture
            layout by
            :func:`~algan.rendering.shaders.materials._normalize_forwarded_maps`.
            One dict is shared by every target, so treat the tensors as
            read-only.

        Returns
        -------
        dict
            ``{slot_name: is_animatable}`` for the slots this Mob took.
        """
        return {}

    def _warn_lighting_beyond_vertex_bake(self, material):
        """Warn when ``material`` can only be baked into vertex colors and the
        Scene's lighting rig asks for more than that bake can deliver.

        Checked against the lights registered *now*, so the usual authoring
        order -- material first, lights later -- is caught by the render's own
        pass over the whole scene instead (see
        :meth:`~algan.render_loop.RenderLoopMixin._warn_vertex_baked_lighting`).
        """
        from algan.rendering.shaders.materials import (
            _PER_FRAGMENT_ADVICE,
            _lighting_beyond_vertex_bake,
            _shades_per_fragment,
        )

        if _shades_per_fragment(material.shader):
            return
        scene = getattr(self, "scene", None)
        features = _lighting_beyond_vertex_bake(
            getattr(scene, "light_sources", None) or (),
            environment_map=getattr(scene, "environment_map", None),
        )
        if not features:
            return
        warnings.warn(
            f"{type(material).__name__}: shading is baked into vertex colors "
            f"(it has no in-kernel port), so {'; '.join(features)}. "
            f"{_PER_FRAGMENT_ADVICE}",
            UnsupportedFeatureWarning,
            stacklevel=_user_stacklevel(),
        )

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
