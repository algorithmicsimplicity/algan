from algan import *

Scene.instance().set_background_color(DARKER_GRAY, True)

with Off():
    title = Text(
        "TIMELINE LAB",
        font_size=54,
        weight="BOLD",
        color=WHITE,
    ).move(UP * 1.55)
    context_label = Tex(
        r"\mathrm{Seq}\ \longrightarrow\ \mathrm{Lag}\ \longrightarrow\ \mathrm{Sync}",
        font_size=31,
        color=TEAL_A,
    ).move(UP * 0.9)
    shapes = (
        Group(
            Circle(radius=0.43, color=BLUE),
            Square(side_length=0.86, color=GREEN),
            RegularPolygon(5, radius=0.5, color=ORANGE),
            Star(outer_radius=0.52, inner_radius=0.23, color=YELLOW),
        )
        .arrange_in_line(RIGHT, buffer=0.5)
        .move(DOWN * 0.05)
    )
    shape_labels = Group(
        Text("circle", font_size=23, color=GRAY_A),
        Text("square", font_size=23, color=GRAY_A),
        Text("polygon", font_size=23, color=GRAY_A),
        Text("star", font_size=23, color=GRAY_A),
    )
    for shape, label in zip(shapes, shape_labels):
        label.move_to(shape.location + DOWN * 0.72)
    counter = (
        NumericDisplay(
            0,
            num_decimal_places=0,
            num_integer_places=2,
            color=WHITE,
        )
        .scale(0.7)
        .move(DOWN * 1.42)
    )

with Seq():
    title.spawn()
    context_label.spawn()
    with Lag(0.25, run_time=1.5):
        for shape in shapes:
            shape.spawn()
    with Lag(0.2, run_time=1.0):
        for label in shape_labels:
            label.spawn()
    counter.spawn()
    with Sync(run_time=1.4):
        shapes.rotate(25, OUT)
        counter.change_value(42)
        context_label.color = BLUE_A
    with Lag(0.22, run_time=2.0):
        Indicate(shapes[0], color=YELLOW, run_time=0.55)
        Wiggle(shapes[1], scale_value=1.16, n_wiggles=4, run_time=0.7)
        Circumscribe(shapes[2], color=TEAL_A, run_time=0.75)
        Flash(shapes[3], color=ORANGE, num_lines=8, run_time=0.75)
    ApplyWave(shape_labels, direction=UP, amplitude=0.16, run_time=1.1)

with Sync():
    shapes.despawn()
    shape_labels.despawn()
    counter.despawn()
    context_label.despawn()

with Off():
    morph = Square(
        side_length=1.15,
        color=BLUE,
        border_color=WHITE,
        border_width=4,
    ).move(LEFT * 1.6)
    parent = (
        Group(
            Circle(radius=0.32, color=GREEN),
            RegularPolygon(3, radius=0.38, color=ORANGE),
            Star(outer_radius=0.4, inner_radius=0.18, color=YELLOW),
        )
        .arrange_in_line(RIGHT, buffer=0.35)
        .move(RIGHT * 1.2)
    )
    hierarchy_label = Text(
        "become + hierarchy + updater",
        font_size=31,
        color=GRAY_A,
    ).move(DOWN * 1.28)

with Seq():
    with Sync():
        morph.spawn()
        parent.spawn()
        hierarchy_label.spawn()
    morph = morph.become(Circle(radius=0.62, color=TEAL))
    morph = morph.become(RegularPolygon(6, radius=0.65, color=ORANGE))
    updater_id = parent.add_updater(lambda mob, time: mob.move(UP * time * 0.08))
    with Sync(run_time=1.5):
        parent.rotate(35, OUT)
        parent.scale(1.1)
        morph.rotate(180, OUT)
    parent.remove_updater(updater_id)
    Blink(morph, time_on=0.18, time_off=0.12, blinks=2)
    Scene.wait(0.35)

Scene.save_frame("algan_outputs/checkpoints/timeline_and_text.png")
