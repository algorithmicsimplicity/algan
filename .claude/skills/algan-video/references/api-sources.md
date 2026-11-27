# API source map and version scope

This skill was source-checked on **2026-11-26** against
`algorithmicsimplicity/algan`, commit
`423db6f5e0374b04874d1f09820893a6a348d58c` on `master`, whose package version
is 0.0.3. `algan-source-commit` and `algan-version` in `SKILL.md` name the same
commit and version, and every link below is pinned to it. It is a snapshot, not
a claim that every published package has that API: an older release on PyPI
can lack names described here.

Use the installed library's verified behavior for a production environment.
Prefer executable implementation over conflicting prose. Retrieve only the
relevant source when resolving a missing feature or changed signature; do not
load the whole renderer-development manual into an ordinary video task.

The links below are pinned for reproducibility. They are evidence and targeted
fallback references, not required reading for every scene.

## Setup and export

- [README: install, CLI, daemon](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/README.md)
- [Public API and settings guidance](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/agent_guidance/api_settings.md)
- [Saving videos and images, including premultiplied export](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/saving_videos_and_images.rst)
- [Root exports and compiler setup](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/__init__.py)
- [Scene implementation: targeted inspection for exact installed signatures](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/scene.py)

## Animation and updaters

- [Combining animations and timing](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/new_user_tutorials/combining_animations.rst)
- [Updater semantics and batched elapsed time](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/new_user_tutorials/updaters.rst)
- [Custom animations](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/custom_animations.rst)
- [Explicit timestamp scheduling](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/animating_out_of_order.rst)
- [AnimationContext implementation and accepted timing names](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/animation_timeline/animation_contexts.py)
- [Animatable implementation and decorated callback contract](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/animatable_base/animatable.py)

## Geometry, text, and assets

- [Positioning and layout](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/positioning_and_layout.rst)
- [Text, formula segments, glyphs, and numeric displays](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/text_and_math.rst)
- [Images and native surface textures](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/images_and_textures.rst)
- [Imported 3D models and rigid animation](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/three_d_models.rst)
- [Manim compatibility boundary and angle conventions](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/agent_guidance/manim_compat.md)
- [Bézier authoring: targeted reference for less common path requirements](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/bezier_curves.rst)

## Materials and shaders

- [Material and shader tutorial](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/shaders_and_materials.rst)
- [Material installation and texture forwarding implementation](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/animatable_base/mob_materials.py)
- [Nine-parameter vertex prefix and RGB+glow layout](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/rendering/shaders/material_shaders.py)
- [FragmentStage, parameter specs, complete cosine stage, and custom scatter example](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/rendering/shaders/fragment_shaders.py)
- [Scatter and lighting-stage contracts: inspect only for a requested custom continuation](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/rendering/raytracing/shading_taichi.py)
- [Path-tracer pipeline handoff implementation](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/algan/rendering/raytracing/path_tracer.py)
- [Lights and shadows](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/lighting_and_shadows.rst)
- [Camera controls](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/cameras.rst)

## Audio and projects

- [Audio/Speech and scene-local sources](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/audio_and_speech.rst)
- [Project rendering, subsets, screenshots, and concatenation](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/multi_scene_projects.rst)

## Backgrounds and compositing

- [Background and post-processing callback contracts](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/backgrounds_and_post_processing.rst)
- [Transparent/glow export, color decoding, and precision limits](https://github.com/algorithmicsimplicity/algan/blob/423db6f5e0374b04874d1f09820893a6a348d58c/docs/source/advanced_user_tutorials/saving_videos_and_images.rst)

## Documentation conflicts

The conflicts an earlier version of this skill listed (the image-texture note,
the `set_fragment_shader` docstring, the material tutorial's per-vertex default
and the custom-animation tutorial's NumPy examples) are fixed in the
documentation at the commit above. One detail is still easy to misread: a material
forwards four image slots (`map`, `normal_map`, `roughness_map`,
`metalness_map`) onto geometry that carries UVs, and `normal_scale` scales a
forwarded normal map; every other image slot is accepted and ignored with a
warning. Where a page and the source disagree, the source wins.

## Skill packaging format

The package follows the [Agent Skills specification](https://agentskills.io/specification):
a named directory with a `SKILL.md` YAML header and on-demand references,
examples, and executable utilities. A `.skill` archive here is a ZIP copy for
hosts that accept that suffix, not a promise that any particular chat interface
can install it. Installation of the skill does not install Algan or grant tools.
