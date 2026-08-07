from algan import *

Scene.instance().set_background_color(DARKER_GRAY, True)

with Off():
    AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 3,
        target=ORIGIN,
        color=WHITE,
        intensity=1.2,
    ).spawn(animate=False)
    title = Text(
        "MEDIA + FRAGMENT PIPELINE",
        font_size=48,
        weight="BOLD",
        color=WHITE,
    ).move(UP * 1.55)
    image = ImageMob("assets/world_map.jpg").scale(1.6).move(LEFT * 1.75)
    image_frame = SurroundingRectangle(
        image,
        color=TEAL_A,
        border_width=4,
        filled=False,
        buffer=0.08,
    )
    shader_sphere = (
        Sphere(radius=0.72, color=BLUE)
        .set_fragment_shader([cosine_color, phong_shader])
        .move(RIGHT * 1.75)
    )
    media_labels = Group(
        Text("ImageMob texture", font_size=26, color=GRAY_A).move(
            LEFT * 1.75 + DOWN * 1.0
        ),
        Text("composed shader", font_size=26, color=GRAY_A).move(
            RIGHT * 1.75 + DOWN * 1.0
        ),
    )

with Seq():
    title.spawn()
    with Sync():
        image.spawn()
        image_frame.spawn()
        shader_sphere.spawn()
        media_labels.spawn()
    with Sync(run_time=2.0):
        image.rotate(22, UP)
        shader_sphere.rotate(210, UP + RIGHT)
        shader_sphere.frequency = 4.5
        shader_sphere.phase = 1.0
    ShowPassingFlash(image_frame, time_width=0.24, run_time=0.9)
    Scene.wait(0.25)

with Sync():
    image.despawn()
    image_frame.despawn()
    shader_sphere.despawn()
    media_labels.despawn()

with Off():
    model = ThreeDModelMob(
        "assets/textured_icosphere.glb",
        normalize=True,
        normalize_size=2.8,
    ).move(DOWN * 0.15)
    model_label = Tex(
        r"\mathrm{GLB}\ +\ \mathrm{PBR}\ +\ \mathrm{normal\ map}",
        font_size=33,
        color=TEAL_A,
    ).move(DOWN * 1.45)

with Seq():
    with Sync():
        model.spawn()
        model_label.spawn()
    with Sync(run_time=2.4):
        model.rotate(300, UP)
        Scene.instance().get_camera().rotate(18, UP, about_point=ORIGIN)
    Circumscribe(model, color=YELLOW, run_time=0.9)
    Scene.instance().get_camera().rotate(-18, UP, about_point=ORIGIN)
    model_label.color = ORANGE
    Scene.wait(0.35)

