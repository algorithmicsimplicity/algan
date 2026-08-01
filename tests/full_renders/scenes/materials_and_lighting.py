from algan import *

Scene.instance().set_background_color(DARKER_GRAY, True)

with Off():
    AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
    key_light = DirectionalLight(
        location=RIGHT * 4 + UP * 4 + OUT * 3,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
        shadow_angle=0.4,
    ).spawn(animate=False)
    title = Text(
        "MATERIAL STUDY",
        font_size=52,
        weight="BOLD",
        color=WHITE,
    ).move(UP * 1.55)
    first_row = Group(
        Sphere(radius=0.48).set_material(MeshBasicMaterial(color=BLUE)),
        Sphere(radius=0.48).set_material(MeshLambertMaterial(color=GREEN)),
        Sphere(radius=0.48).set_material(
            MeshPhongMaterial(color=ORANGE, specular=WHITE, shininess=80)
        ),
        Sphere(radius=0.48).set_material(
            MeshStandardMaterial(color=RED, roughness=0.18, metalness=0.75)
        ),
    ).arrange_in_line(RIGHT, buffer=0.55).move(DOWN * 0.05)
    first_labels = Group(
        Text("Basic", font_size=24, color=GRAY_A),
        Text("Lambert", font_size=24, color=GRAY_A),
        Text("Phong", font_size=24, color=GRAY_A),
        Text("Standard", font_size=24, color=GRAY_A),
    )
    for mob, label in zip(first_row, first_labels):
        label.move_next_to(mob, DOWN)

with Seq():
    title.spawn()
    with Lag(0.2, run_time=1.4):
        for mob in first_row:
            mob.spawn()
    with Lag(0.15, run_time=0.9):
        for label in first_labels:
            label.spawn()
    with Sync(run_time=1.8):
        for mob in first_row:
            mob.rotate(150, UP)
        first_row[2].shininess = 20
        first_row[3].roughness = 0.72
        first_row[3].metalness = 0.2
        key_light.move(LEFT * 2)
    Scene.wait(0.45)
    with Sync(run_time=0.8):
        first_row.scale(0.74)
        first_row.move(UP * 0.62)
        first_labels.scale(0.82)
        first_labels.move(UP * 0.62)

with Off():
    second_row = Group(
        Sphere(radius=0.43).set_material(
            MeshPhysicalMaterial(
                color=BLUE_A,
                roughness=0.12,
                clearcoat=0.8,
                transmission=0.45,
                ior=1.45,
            )
        ),
        Sphere(radius=0.4).set_material(
            MeshToonMaterial(color=GREEN, bands=4)
        ),
        Sphere(radius=0.4, color=WHITE).set_material(
            MeshNormalMaterial()
        ),
        Sphere(radius=0.4).set_material(
            MeshMatcapMaterial(color=ORANGE)
        ),
        Sphere(radius=0.4).set_material(
            MeshDepthMaterial(near=2.0, far=12.0)
        ),
    ).arrange_in_line(RIGHT, buffer=0.36).move(DOWN * 0.68)
    second_labels = Group(
        Text("Physical", font_size=21, color=GRAY_A),
        Text("Toon", font_size=21, color=GRAY_A),
        Text("Normal", font_size=21, color=GRAY_A),
        Text("Matcap", font_size=21, color=GRAY_A),
        Text("Depth", font_size=21, color=GRAY_A),
    )
    for mob, label in zip(second_row, second_labels):
        label.move_to(mob.location + DOWN * 0.72)

with Seq():
    with Lag(0.16, run_time=1.5):
        for mob in second_row:
            mob.spawn()
    with Lag(0.12, run_time=0.8):
        for label in second_labels:
            label.spawn()
    with Sync(run_time=2.0):
        for mob in second_row:
            mob.rotate(180, UP + RIGHT)
        second_row[0].clearcoat = 0.2
        second_row[0].transmission = 0.15
        key_light.move(RIGHT * 3)
        Scene.get_camera().rotate(12, UP, about_point=ORIGIN)
    Indicate(second_row[0], color=YELLOW, run_time=0.65)
    Scene.get_camera().rotate(-12, UP, about_point=ORIGIN)
    Scene.wait(0.3)

Scene.save_frame(
    "algan_outputs/checkpoints/materials_and_lighting.png"
)
