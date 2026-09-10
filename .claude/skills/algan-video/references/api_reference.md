# Algan API reference for video authoring

Everything here is reachable with `from algan import *` unless an import line
says otherwise. Angles are degrees. Distances are world units (the default frame
is about 12.4 x 7 units at z = 0).

## Scene

`Scene.method(...)` acts on the active Scene; `scene.method(...)` on a specific
one. A default Scene is created when the first Mob is built.

| Call | Purpose |
| --- | --- |
| `Scene.save_video(path=None, video_settings=None, *, overwrite=True, reset=False, background=None, animate_fade_out=None, post_processes=None, codec=None, audio_codec=None, ffmpeg_params=None)` | Render the recording. Returns `RenderResult` (`status`, `output_path`, `walltime_seconds`, `render_plan`). |
| `Scene.save_frame(path=None, video_settings=None, at=None, *, overwrite, background, post_processes)` | Render a still at time `at` (seconds; negative counts back from now; a list writes one file per time and returns a list). |
| `Scene.wait(seconds=1)` | Pause on the timeline. |
| `Scene.get_camera()` | The `Camera` Mob. |
| `Scene.get_light_sources()` / `Scene.remove_light(light)` / `Scene.clear_lights()` | Light management. Index 0 is the default point light. |
| `Scene.set_background(color_or_path_or_callable)` | Whole-Scene background (not animatable). |
| `Scene.set_environment_map(path_or_array, intensity=1.0, ambient=True)` | 360-degree equirectangular skybox and image-based lighting; `None` removes it. |
| `Scene.set_video_settings(preset)` | Change the active Scene's settings after creation. |
| `Scene.set_premultiplied_over()` | Transparent export carrying additive glow (compositing intermediate). |
| `Scene.use_manim_defaults()` | Manim's frame height, camera, light, background and stroke conventions. Call before building anything. |
| `Scene.current()` | The active Scene instance (e.g. `Scene.current().audio_manager`). |
| `Scene.view(...)` | Interactive browser viewer. Blocks; not for unattended runs. |
| `with Scene(video_settings=PREVIEW) as scene:` | An explicit Scene; `scene.save_video(...)` inside. |

## Mob lifecycle and attributes

| Member | Notes |
| --- | --- |
| `spawn(animate=True)` | Required to appear. Returns the Mob. `spawn(False)` skips the fade-in. |
| `despawn(animate=True)` | Fade out and stop drawing. |
| `wait(seconds)` | Hold this Mob (and advance the timeline). |
| `location` | 3-vector. Assignment animates. |
| `basis` | 3x3 orientation and scale. Change through `rotate`/`scale`. |
| `color` | 5-channel `Color` (r, g, b, glow, opacity). |
| `glow`, `opacity` | Floats. Glow needs the bloom pass (default) to show. |
| `set(**attrs)` | Several attributes in one animation. |
| `add_updater(fn, *args, **kwargs) -> int` / `remove_updater(id)` | Per-frame rule; `fn(mob, t, *args, **kwargs)`, `t` a tensor `[frames, 1, 1]`. |
| `become(other, detach_history=True, minimize_movement=False)` | Morph into `other` (build with `add_to_scene=False`). Returns the resulting Mob; reassign. |
| `children`, `parents` (read-only), `add_children([...])`, `add_parent(p)`, `remove_child(c)`, `remove_parent(p)`, `replace_children([...])`, `get_descendants()` | Hierarchy. |
| `register_attrs_as_animatable(["name"])` | In a subclass `__init__`, before assigning, to add an animatable attribute. |

Reading an animatable attribute returns a copy; in-place edits of the copy are
discarded. Assign the whole value.

### Movement

| Method | Effect |
| --- | --- |
| `move(delta)` | Translate by a vector. |
| `move_to(point, arc_angle=None)` | Translate to a point; `arc_angle` swings along an arc. |
| `move_between(a, b)` | Midpoint of two points or Mobs. |
| `x`, `y`, `z` | Assign one axis. |
| `move_next_to(mob_or_point, direction, buffer=None, align_edge=None)` | Edge to edge beside another Mob. |
| `align_with(mob, direction, anchor="center" or "boundary" or "edge")` | Line up along one axis. |
| `move_to_screen_edge(direction, buffer=None)` | Rest against a screen edge (camera frame). |
| `move_to_screen_corner((UP, LEFT))` | Rest in a corner. |
| `move_to_screen_position(x_frac, y_frac)` | (0,0) bottom-left, (1,1) top-right. |
| `move_center_to_screen_position(x_frac, y_frac)` | Same, by centre. |
| `move_off_screen()` | Slide out and despawn. |
| `fit_to_screen(lower_left=(0,0), upper_right=(1,1), preserve_aspect_ratio=True)` | Scale and move into a screen rectangle. |

### Orientation and size

| Method | Effect |
| --- | --- |
| `rotate(angle, axis=OUT, about=None)` | Turn about an axis through the centre, or about a point (orientation travels with it). |
| `orbit(angle, axis=OUT, about=p)` | Travel around a point without turning. |
| `look_at(point)` | Face a point. |
| `reset_basis()` | Default orientation and scale. |
| `get_right_direction()`, `get_up_direction()`, `get_forward_direction()` | The Mob's own axes. |
| `scale(factor)` | Uniform scale. |
| `scale_to_height(h)`, `scale_to_width(w)` | Uniform scale to a size. |
| `get_width()`, `get_height()`, `get_depth()`, `get_center()`, `get_bounding_box()` | Measurements, read now. |

## Animation contexts

`Seq`, `Sync`, `Lag(ratio)`, `Off`, `Audio(clip, wait_at_end=0)`,
`Speech(text, wait_at_end=1)`. Keyword arguments on any of them:

| Argument | Meaning |
| --- | --- |
| `runtime` | Seconds for the whole block; inner animations rescale. Overrides `runtime_per_part`. |
| `runtime_per_part` | Seconds per animation inside. |
| `easing` | Rate function over the whole block. |
| `composed_easing` | Compose with the parent's easing instead of replacing it. |
| `lag_ratio` | For `Lag`; `Sync` is 0, `Seq` is 1. |

`with Seq() as ctx:` exposes `ctx.current_time` (assignable write pointer) and
`ctx.end_time`. Set `ctx.current_time = ctx.end_time` when done jumping around.

Animation functions (`Indicate`, `MoveAlongPath`, ...) accept `runtime=` directly.

## Easings (`easings.*`)

`smooth` (default), `identity`/`linear`, `ease_in_sine`, `ease_out_sine`,
`ease_in_out_sine`, `ease_in_quad`, `ease_out_quad`, `ease_in_out_quad`,
`ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`, `ease_in_quart`,
`ease_out_quart`, `ease_in_out_quart`, `ease_in_quint`, `ease_out_quint`,
`ease_out_quintic`, `ease_in_out_quint`, `ease_in_expo`, `ease_out_expo`,
`ease_out_exp`, `ease_in_circ`, `ease_out_circ`, `ease_in_out_circ`,
`ease_in_back`, `ease_out_back`, `ease_in_out_back`, `ease_in_elastic`,
`ease_out_elastic`, `ease_in_out_elastic`, `ease_in_bounce`, `ease_out_bounce`,
`ease_in_out_bounce`, `rush_into`, `rush_from`, `slow_into`, `delay_fade`,
`pulse_fade`, `inversed(f)`.

## Colours

`Color((r, g, b))` or `Color([r, g, b, glow, opacity])`, components in 0..1.
Methods: `set_glow(g)`, `set_opacity(a)`, `mult_opacity(k)`; properties `.rgb`,
`.glow`, `.opacity`. Constants: the Manim palette (`RED`, `RED_A`..`RED_E`,
`BLUE`, `GREEN`, `YELLOW`, `GOLD`, `TEAL`, `PURPLE`, `MAROON`, `ORANGE`, `PINK`,
`GREY`/`GRAY` with `_A`..`_E` shades, `DARK_BLUE`, `LIGHT_BROWN`, ...), `WHITE`,
`BLACK`, `TRANSPARENT`, `PURE_RED`, `PURE_GREEN`, `PURE_BLUE`, `CSS_COLORS`.
Material constructors also accept hex ints and strings. Multiplying a colour
scales its opacity too.

## Mobs

All take `color=` and `location=`; 2D circuits take `stroke_width=`,
`stroke_color=`, `filled=`, `grid_width=`, `grid_height=`. Pass
`add_to_scene=False` for a Mob that only serves as a `become`/`set_shape_to`
target.

### 2D shapes (Bezier circuits, unlit, exact edges)

| Class | Constructor highlights |
| --- | --- |
| `Circle(radius=1)`, `Dot()`, `Point()` | |
| `Square(size)`, `Rectangle(width, height)`, `RoundedRectangle`, `Quad(4 corners)` | |
| `RegularPolygon(n)`, `Triangle()`, `Polygon(*points)`, `Polygram`, `RegularPolygram` | |
| `Line(start=LEFT, end=RIGHT)`, `DashedLine`, `Arrow(start, end)`, `DoubleArrow`, `Vector`, `CurvedArrow`, `Elbow`, `Angle`, `RightAngle`, `TangentLine` | Arrow tips: `ArrowTriangleTip`, `ArrowCircleTip`, `StealthTip`, ... |
| `Arc(radius, start_angle, angle)`, `ArcBetweenPoints`, `Annulus(inner_radius, outer_radius)`, `Sector`, `AnnularSector`, `Ellipse(width, height)`, `Star(n, outer_radius, inner_radius)`, `Cross` | Manim-argument shapes. |
| `SurroundingRectangle(mob)`, `BackgroundRectangle`, `Underline`, `Cutout` | Decorations around other Mobs. |
| `BezierCurveCubic`, `BezierCircuitCubic`, `CubicBezier`, `ParametricFunction`, `FunctionGraph`, `ImplicitFunction` | Curves. |
| `SVGMob("file.svg")` | Path geometry only; children are the paths. |
| `DashedMob`, `Union`, `Intersection`, `Difference`, `Exclusion` | Boolean and dashed variants. |

### 3D shapes (triangle meshes, lit)

| Class | Constructor highlights |
| --- | --- |
| `Sphere(radius=1, resolution=None)` | Curved surface. |
| `Cylinder(radius=1, height=1, direction=UP, closed=False)`, `Cone(...)` | Curved. |
| `Torus(ring_radius, tube_radius)` | Curved. |
| `Dot3D`, `Line3D(start, end)`, `Arrow3D` | Markers. |
| `Cube(size)`, `Prism(width, height, depth)` | Faceted. |
| `Tetrahedron`, `Octahedron`, `Icosahedron`, `Dodecahedron` (`edge_length=`) | Faceted. |
| `Polyhedron(vertices, faces)`, `ConvexHull3D(points)`, `TriangleMesh` | Custom meshes. |
| `Surface(uv_function, color_texture=None, roughness_texture=None, reflectivity_texture=None, refractive_index_texture=None, normal_texture=None, glow_texture=None, grid_width=, grid_height=, render_tolerance_pixels=0.5)` | `uv_function(uv[..., 2] in [0,1]) -> xyz`, batched torch ops. |
| `ImageMob(path_or_array)` | Flat textured plane; a `Surface`. |
| `Model3D(path, fit_to_size=None, load_textures=True, smooth_normals=True, normal_maps=True, pbr_materials=True)` | glTF/GLB/OBJ/PLY/STL/DAE/OFF; FBX with `algan[fbx]` + assimp. `node_names`, `get_part(name)`, `animation_names`, `play_animation(name, runtime, loop, fps, easing)`. |
| `Sphere.from_batches(centers, radius, color)` | Hundreds of identical shapes as one packed Mob; index for individuals. |
| `DotCloud`, `TriangulatedBezierCircuit`, `TextTriangulated`, `TexTriangulated` | Point clouds; lit versions of 2D outlines. |

### Text and numbers

| Class | Notes |
| --- | --- |
| `Text(str, font_size=48, color=, font=, weight="BOLD", slant="ITALIC", color_map={}, font_map, slant_map, weight_map, line_spacing, gradient)` | `character_mobs` for glyphs; `.write(runtime, lag_ratio)`. |
| `Tex(*strings, font_size=)` | LaTeX math mode; segments via `get_segment(i)`, `tex_strings`; index glyphs `formula[3]`. |
| `MathTex`, `Title`, `Paragraph`, `BulletedList`, `MarkupText`, `Code`, `Label`, `LabeledDot`, `LabeledLine`, `LabeledArrow` | Manim-compatible text Mobs. |
| `DecimalNumber(value, decimal_places=, integer_places=)`, `Integer`, `Variable` | `.value` animates. |
| `Matrix`, `DecimalMatrix`, `IntegerMatrix`, `MobMatrix`, `Table`, `MathTable`, `DecimalTable`, `IntegerTable`, `MobTable` | Grids of content. |

### Diagrams (Manim-compatible, native)

`Axes(x_range, y_range, x_length, y_length)` with `.plot(f, color)`,
`.plot_parametric_curve`, `.get_axis_labels`, `.get_graph_label`, `.c2p`,
`.add_coordinates()`; `ThreeDAxes`, `NumberPlane`, `ComplexPlane`, `PolarPlane`,
`NumberLine`, `UnitInterval`, `BarChart`, `Graph`, `DiGraph`, `Brace`,
`BraceBetweenPoints`, `BraceLabel`, `BraceText`, `VectorField`,
`ArrowVectorField`, `StreamLines`, `SampleSpace`, `ScreenRectangle`,
`FullScreenRectangle`, `SlideShow`. What their methods return are Mobs to spawn
separately. Any other Manim Mobject: `ManimMob(mn.Something())` with
`import manim as mn`.

### Groups

`Group(*mobs_or_list)`: indexable, iterable, centre at the members' centre.
`arrange_in_line(direction=RIGHT, buffer=None, start_at_first=False, equal_widths=False, align_to=None)`,
`arrange_in_grid(num_rows=None, row_direction=RIGHT, column_direction=DOWN, row_buffer=None, column_buffer=None)`.

## Built-in animation functions

| Function | Signature highlights |
| --- | --- |
| `Indicate(mob, scale_factor=1.2, color=YELLOW, runtime=1)` | Scale up and tint. |
| `Circumscribe(mob, shape=None, fade_in=False, ...)` | Animated outline. |
| `Flash(point_or_mob, line_length=0.2, num_lines=12, ...)` | Radial burst. |
| `FocusOn(point_or_mob, opacity=0.2, color=GRAY, runtime=2)` | Dim the rest. |
| `Wiggle(mob, scale_value=1.1, rotation_angle=...)` | |
| `Blink(mob, time_on=0.5, time_off=0.5, blinks=...)` | |
| `ShowPassingFlash(mob, time_width=0.1, runtime=1)`, `ShowPassingFlashWithThinningStrokeWidth` | Travelling highlight along an outline. |
| `DrawBorderThenFill(mobs, runtime=None, lag_ratio=None)` | Handwriting for shapes; `spawn(False)` first. |
| `AnimatedBoundary(mob, max_stroke_width, cycle_rate, colors)` | A Mob; `.spawn()`, `.stop()`. |
| `MoveAlongPath(mob, path_mob, runtime)` | Path may stay unspawned. |
| `ApplyMatrix(mob, matrix_2x2_or_3x3, runtime)` | |
| `ApplyPointwiseFunction(mob, f(points))`, `ApplyComplexFunction(mob, f(z))` | |
| `Homotopy(mob, f(x, y, z, t) -> (x, y, z), runtime)`, `ComplexHomotopy` | Continuous deformation. |
| `PhaseFlow(mob, field(points) -> vectors, runtime, virtual_time, integration_steps)` | Flow along a vector field. |
| `ApplyWave(mob, direction=UP, amplitude=0.2, runtime)` | |
| `animated_function(animated_args={"name": start})` | Decorator; first parameter is the Mob, swept args are floats. |

Callbacks receive batched torch tensors and must return tensors of the same
shape; use torch operations, no Python loops over points.

## Camera (`Scene.get_camera()`)

A Mob: `move`, `rotate(deg, UP, about=ORIGIN)` (turntable), `orbit`, `look_at`,
`center_on(mob, buffer_portion=0.7)`, updaters. Animatable: `set_fov(deg)`,
`set_distance_to_screen(d)`. Configuration (not animatable, set before
spawning content): `set_near(d)`, `set_far(d)`, `set_near_orthographic()`.
Constructor: `Camera(orthographic=False, screen_distance=5, screen_half_height=2.5, fov=None, near=0, far=0)`.
Default: at `OUT * 7`, vertical FOV about 53 degrees.

## Lights

Every light: `location=`, `color=` (default white), `intensity=1.0`; all three
animate. Spawn inside `Off()`.

| Class | Extra arguments |
| --- | --- |
| `PointLight` | `decay=0` (2 for inverse-square), `distance=0`, `shadow_radius=0` (soft shadows). |
| `DirectionalLight` | `target=ORIGIN`, `shadow_angle=0` (degrees; the sun is about 0.5). |
| `AmbientLight` | none. |
| `HemisphereLight` | `ground_color=`, `up=UP`. Sky colour is `color`. |
| `SpotLight` | `target`, `cone_angle=30`, `penumbra=0`, `decay`, `distance`, `shadow_radius`. |
| `RectAreaLight` | `width=2`, `height=2`, `target`, `samples=4` (emitter grid, soft by nature). Leave `decay`/`distance` at defaults. |

Shadows: `SETTINGS.raytracing.set(shadows=True)`. At most 16 shadow-casting
light slots by default; each area-light sample uses one.

## Materials and shaders

See `three_d_and_shaders.md`. Quick list: `MeshBasicMaterial`,
`MeshLambertMaterial`, `MeshPhongMaterial`, `MeshStandardMaterial`,
`MeshPhysicalMaterial`, `MeshToonMaterial`, `MeshNormalMaterial`,
`MeshMatcapMaterial`, `MeshDepthMaterial`, `DiffuseMaterial` (default),
`UnlitMaterial`; presets `WOOD`, `GLASS`, `PLASTIC`, `RUBBER`, `CERAMIC`,
`STONE`, `MIRROR`, `BRUSHED_METAL`, `CHROME`, `COPPER`. Fragment stages
`STAGE_UNLIT`, `STAGE_LAMBERT`, `STAGE_PHONG`, `STAGE_STANDARD`,
`STAGE_PHYSICAL`, `STAGE_MANIM`, `cosine_color`, `fresnel_rim`, `glass_ball`,
`FragmentStage(ti_func, param_specs, scatter=None)`. Vertex shader functions
`basic_material_shader`, `lambert_shader`, `phong_shader`, `standard_shader`,
`physical_shader`, `toon_shader`, `normal_shader`, `matcap_shader`,
`depth_shader`, `basic_pbr_shader`.

## Procedural textures (`[W, H, 5]` tensors)

`get_checkerboard(colors=(WHITE, BLACK), ...)`, `get_stripes(colors)`,
`get_grid_lines(line_color, background_color)`, `get_polka_dots(dot_color, background_color)`,
`get_bricks(colors, mortar_color)`, `get_gradient(colors)`,
`get_radial_gradient(colors)`, `get_noise(colors)`. Each also takes a cell
count and `texture_resolution`.

## Presets and settings

Video presets: `SMOKE_TEST` (32x32, 2 fps), `PREVIEW` (704x396, 10),
`LD` (864x486, 15, the default), `MD` (1280x720, 30), `HD` (1920x1080, 30),
`PRODUCTION` (2560x1440, 60), `UHD` (3840x2160, 60), `THUMBNAIL` (1280x720, 1).
`VideoSettings(resolution=(w, h), frames_per_second=30, supersampling=2, fxaa=False)`
builds one from scratch; `preset.set(...)` returns a modified copy.

`SETTINGS` sections and fields:

| Section | Fields |
| --- | --- |
| `video` | `resolution`, `frames_per_second` (`fps`), `supersampling` (`ssaa`), `fxaa`, `audio_sample_rate` |
| `style` | `background`, `frame`, `text_color`, `buffer`, `fade_out_on_scene_end`, `default_material`, `shape_style_profile` ("algan"/"manim"), `border_placement` ("inward"/"centered"), `manim_stroke_width_ratio` |
| `paths` | `output_root`, `output_directory`, `output_filename`, `cache_directory`, `ffmpeg_binary` |
| `computing` | `render_device`, `rendering_memory_fraction`, `torch_compile`, `max_animation_batch_size`, ... |
| `raytracing` | `samples_per_pixel`, `max_bounces`, `shadows`, `denoise`, `analytic_aa`, `texture_antialiasing`, `glossy_reflection`, `glossy_prefilter`, `tonemapping`, `tonemap_method` ("neutral"/"agx"), `tonemap_exposure`, `linear_color_space`, `unsupported_feature_policy`; `experimental.*` for kernel switches |

`SETTINGS.video.set(HD, fps=60)`, `SETTINGS.video.fps = 60`,
`with SETTINGS.video.override(fps=12): ...`,
`with SETTINGS.override(video={...}, raytracing={...}): ...`,
`SETTINGS.snapshot()` / `SETTINGS.restore(snap)`.

## Utilities

`clear_cache()` (content caches), `clear_cached_kernels()` (everything),
`set_log_level(...)`, `set_progress_style(...)`, `PI`, `TAU`, `DEGREES`,
`RADIANS`, `DEFAULT_RUNTIME`, `CAMERA_ORIGIN`,
`algan.utils.file_utils.get_image(path)`,
`algan.utils.audio_utils.get_speech_generator_from_file(audio_file, transcript_file)`,
`algan.rendering.post_processing.bloom.bloom_filter`.
