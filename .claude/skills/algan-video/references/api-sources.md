# API source map and version scope

This skill was source-checked on **2026-09-10** against
`algorithmicsimplicity/algan`, commit
`f9e6d73c12de35f7e14315c2d49df34de570c644` on `master`.
It is a snapshot, not a claim that every published package has that API.

Use the installed library's verified behavior for a production environment.
Prefer executable implementation over conflicting prose. Retrieve only the
relevant source when resolving a missing feature or changed signature; do not
load the whole renderer-development manual into an ordinary video task.

The links below are pinned for reproducibility. They are evidence and targeted
fallback references, not required reading for every scene.

## Setup and export

- [README: install, CLI, daemon](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/README.md)
- [Public API and settings guidance](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/agent_guidance/api_settings.md)
- [Saving videos and images, including premultiplied export](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/saving_videos_and_images.rst)
- [Root exports and compiler setup](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/__init__.py)
- [Scene implementation: targeted inspection for exact installed signatures](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/scene.py)

## Animation and updaters

- [Combining animations and timing](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/new_user_tutorials/combining_animations.rst)
- [Updater semantics and batched elapsed time](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/new_user_tutorials/updaters.rst)
- [Custom animations](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/custom_animations.rst)
- [Explicit timestamp scheduling](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/animating_out_of_order.rst)
- [AnimationContext implementation and accepted timing names](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/animation_timeline/animation_contexts.py)
- [Animatable implementation and decorated callback contract](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/animatable_base/animatable.py)

## Geometry, text, and assets

- [Positioning and layout](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/positioning_and_layout.rst)
- [Text, formula segments, glyphs, and numeric displays](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/text_and_math.rst)
- [Images and native surface textures](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/images_and_textures.rst)
- [Imported 3D models and rigid animation](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/three_d_models.rst)
- [Manim compatibility boundary and angle conventions](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/agent_guidance/manim_compat.md)
- [Bézier authoring: targeted reference for less common path requirements](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/bezier_curves.rst)

## Materials and shaders

- [Material and shader tutorial](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/shaders_and_materials.rst)
- [Material installation and texture forwarding implementation](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/animatable_base/mob_materials.py)
- [Nine-parameter vertex prefix and RGB+glow layout](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/rendering/shaders/material_shaders.py)
- [FragmentStage, parameter specs, complete cosine stage, and custom scatter example](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/rendering/shaders/fragment_shaders.py)
- [Scatter and lighting-stage contracts: inspect only for a requested custom continuation](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/rendering/raytracing/shading_taichi.py)
- [Path-tracer pipeline handoff implementation](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/algan/rendering/raytracing/path_tracer.py)
- [Lights and shadows](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/lighting_and_shadows.rst)
- [Camera controls](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/cameras.rst)

## Audio and projects

- [Audio/Speech and scene-local sources](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/audio_and_speech.rst)
- [Project rendering, subsets, screenshots, and concatenation](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/multi_scene_projects.rst)

## Backgrounds and compositing

- [Background and post-processing callback contracts](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/backgrounds_and_post_processing.rst)
- [Transparent/glow export, color decoding, and precision limits](https://github.com/algorithmicsimplicity/algan/blob/f9e6d73c12de35f7e14315c2d49df34de570c644/docs/source/advanced_user_tutorials/saving_videos_and_images.rst)

## Documentation conflicts resolved in this skill

The early note in the image-texture tutorial says material image slots are not
sampled. The later text and `MobMaterialsMixin.set_material` implementation
instead forward four supported slots to UV geometry; this skill describes that
implementation, not the obsolete note.

An older `set_fragment_shader` docstring says the path tracer ignores it.
`FragmentStage`'s current contract and the path-tracer pipeline handoff support
custom pipelines/scatters. This skill does not repeat the obsolete limitation.

The material tutorial also retains an early per-vertex-default description
alongside the current built-in fragment-path description. Built-in materials
and a plain custom PyTorch vertex shader must not be conflated.

The custom-animation tutorial contains NumPy examples. This skill uses torch
inside replayed functions to preserve batched time and device behavior.

## Skill packaging format

The package follows the [Agent Skills specification](https://agentskills.io/specification):
a named directory with a `SKILL.md` YAML header and on-demand references,
examples, and executable utilities. A `.skill` archive here is a ZIP copy for
hosts that accept that suffix, not a promise that any particular chat interface
can install it. Installation of the skill does not install Algan or grant tools.
