"""Minimal authoring mechanics, not a creative template.

Requires Algan. Run with the project's Python interpreter.
The explicit PREVIEW export overrides an Algan CLI quality default.
"""

from algan import Off, OUT, PREVIEW, RIGHT, LEFT, Scene, Square, Sync


def build_scene():
    with Off():
        shape = Square().scale(0.5).move_to(LEFT).spawn()
    with Sync(runtime=1.5):
        shape.move(RIGHT * 2)
        shape.rotate(90, OUT)
    Scene.wait(0.5)


if __name__ == "__main__":
    build_scene()
    result = Scene.save_video("renders/basic_scene.mp4", PREVIEW)
    print(result.status, result.output_path)
