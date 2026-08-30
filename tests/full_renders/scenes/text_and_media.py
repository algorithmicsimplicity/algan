"""Typesetting, imported media, textures and custom fragment shading.

``Text`` and ``Tex`` are bezier circuits produced by the (cached) Manim glyph
pipeline, so this scene is the reference for glyph geometry, per-glyph
addressing, and the hand-writing animations built on top of them.  It also
covers the two ways pixels enter a scene from outside the engine -- an
``ImageMob`` texture and an imported glTF model with PBR and normal maps -- and
the in-kernel fragment shader stack.

Assets are referenced relative to ``tests/full_renders``; the harness makes that
the working directory while a scene renders.
"""

from algan import *

# Pinned so the render does not depend on the host's fonts;
# tests/conftest.py registers the vendored faces.
FONT = "Algan Test Sans"
MONO_FONT = "Algan Test Mono"

Scene.set_background(DARKER_GRAY)

# --------------------------------------------------------------------------
# Act 1 -- the typesetting stack, one family per column.
# --------------------------------------------------------------------------
with Off():
    AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)

    title = Text(
        "TEXT AND MEDIA",
        font_size=44,
        weight="BOLD",
        color=WHITE,
        font=FONT,
    ).move(UP * 3.0)

    plain = Text("regular", font_size=34, color=BLUE_A, font=FONT).move(
        LEFT * 4.3 + UP * 1.8
    )
    bold = Text("bold", font_size=34, weight="BOLD", color=GREEN_A, font=FONT).move(
        LEFT * 4.3 + UP * 1.1
    )
    italic = Text("italic", font_size=34, slant="ITALIC", color=ORANGE, font=FONT).move(
        LEFT * 4.3 + UP * 0.4
    )
    markup = MarkupText(
        '<span foreground="#ffd700">markup</span>',
        font_size=34,
        font=FONT,
    ).move(LEFT * 4.3 + DOWN * 0.3)
    # The mesh-backed variants: same glyphs, triangulated instead of packed as
    # bezier circuits, so they exercise the flat-triangle path for text.
    mesh_text = (
        TextTriangulated("mesh", font_size=38, color=RED)
        .set_material(MeshBasicMaterial(color=RED))
        .move(LEFT * 4.3 + DOWN * 1.0)
    )
    mesh_tex = (
        TexTriangulated(r"\alpha\beta", font_size=38, color=TEAL_A)
        .set_material(MeshBasicMaterial(color=TEAL_A))
        .move(LEFT * 4.3 + DOWN * 1.8)
    )

    formula = Tex(r"\int_{0}^{1} x^{2}\,dx = \frac{1}{3}", font_size=42, color=WHITE)
    formula.move(RIGHT * 0.3 + UP * 1.6 - formula.get_center())
    matrix_tex = MathTex(
        r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}", font_size=42, color=TEAL_A
    )
    matrix_tex.move(RIGHT * 0.3 + DOWN * 0.15 - matrix_tex.get_center())

    paragraph = (
        Paragraph("wrapped lines", "share one Group", alignment="left", font=FONT)
        .scale(0.75)
        .move(RIGHT * 4.2 + UP * 1.5)
    )
    code = (
        Code(
            code_string="def f(x):\n    return x * 2",
            background="window",
            # Code defaults to the "Monospace" fontconfig alias, which
            # resolves to whatever the host installs; name the vendored
            # family so this block is reproducible too.
            paragraph_config={"font": MONO_FONT},
        )
        .scale(0.42)
        .move(RIGHT * 4.0 + DOWN * 0.4)
    )

    counter_label = Text(
        "Text / MarkupText / TextTriangulated / Tex / MathTex / Paragraph / Code",
        font_size=22,
        color=GRAY_A,
        font=FONT,
    ).move(DOWN * 2.65)

with Seq():
    title.spawn()
    with Lag(0.18, run_time=1.6):
        plain.spawn()
        bold.spawn()
        italic.spawn()
        markup.spawn()
        mesh_text.spawn()
        mesh_tex.spawn()
    with Sync(run_time=0.9):
        formula.spawn()
        matrix_tex.spawn()
    with Sync(run_time=0.9):
        paragraph.spawn()
        code.spawn()
    counter_label.spawn()

# --------------------------------------------------------------------------
# Act 2 -- hand-writing, per-glyph addressing and Tex-to-Tex morphing.
# --------------------------------------------------------------------------
with Off():
    written = Tex(r"\textrm{write}", font_size=52, color=YELLOW)
    written.move(LEFT * 0.2 + DOWN * 1.55 - written.get_center())
    glyphs = Text("GLYPHS", font_size=46, color=WHITE, font=FONT)
    glyphs.move(RIGHT * 3.9 + DOWN * 1.55 - glyphs.get_center())
    # ``text[i]`` is a lazy view onto glyph i of the packed batch.
    lifted = [glyphs[index] for index in range(3)]
    # ``become`` morphs position as well as shape, so the target is built where
    # the formula already sits.
    series = Tex(
        r"\sum_{n=1}^{\infty} \frac{1}{n^{2}} = \frac{\pi^{2}}{6}",
        font_size=42,
        add_to_scene=False,
    )
    series.move(formula.get_center() - series.get_center())

with Seq():
    with Sync(run_time=0.5):
        glyphs.spawn()
    written.spawn(False).write(run_time=1.6)
    # Glyphs are individually addressable, so single letters animate on their
    # own while the rest of the word stays put.
    with Lag(0.25, run_time=1.4):
        for glyph in lifted:
            glyph.move(UP * 0.35)
    with Lag(0.25, run_time=1.4):
        for glyph in lifted:
            glyph.move(DOWN * 0.35)
    # ``minimize_movement`` pairs each triangle with the closest one in the
    # target, so the glyphs deform in place instead of scattering.
    formula.become(series, minimize_movement=True)
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 3 -- imported media and the fragment shader stack.
# --------------------------------------------------------------------------
with Sync(run_time=0.8):
    for mob in (
        plain,
        bold,
        italic,
        markup,
        formula,
        matrix_tex,
        paragraph,
        code,
        written,
        glyphs,
        counter_label,
        mesh_text,
        mesh_tex,
    ):
        mob.despawn()

with Off():
    image = ImageMob("assets/world_map.jpg").scale(1.9).move(LEFT * 3.7 + UP * 0.3)
    image_frame = SurroundingRectangle(
        image,
        color=TEAL_A,
        stroke_width=4,
        filled=False,
        buffer=0.1,
    )
    flat_image = (
        ImageMob("assets/world_map.jpg", textured=False)
        .scale(1.1)
        .move(LEFT * 3.7 + DOWN * 2.1)
    )
    model = Model3D(
        "assets/textured_icosphere.glb",
        fit_to_size=2.6,
    ).move(UP * 0.2)
    shader_sphere = (
        Sphere(radius=0.95, color=BLUE)
        .set_fragment_shader([cosine_color, phong_shader])
        .move(RIGHT * 3.9 + UP * 0.3)
    )
    media_labels = Group(
        Text(
            "textured / per-pixel ImageMob", font_size=21, color=GRAY_A, font=FONT
        ).move(LEFT * 3.7 + DOWN * 1.35),
        Text("glTF + PBR + normal map", font_size=21, color=GRAY_A, font=FONT).move(
            DOWN * 1.9
        ),
        Text("composed fragment shader", font_size=21, color=GRAY_A, font=FONT).move(
            RIGHT * 3.9 + DOWN * 1.35
        ),
    )

with Seq():
    with Sync(run_time=0.8):
        image.spawn()
        image_frame.spawn()
        flat_image.spawn()
        model.spawn()
        shader_sphere.spawn()
        media_labels.spawn()
    with Sync(run_time=2.4):
        image.rotate(28, UP)
        model.rotate(300, UP)
        shader_sphere.rotate(210, UP + RIGHT)
        shader_sphere.frequency = 4.5
        shader_sphere.phase = 1.0
    ShowPassingFlash(image_frame, time_width=0.25, run_time=1.0)
    Circumscribe(model, color=YELLOW, run_time=0.9)
    Scene.wait(0.3)
