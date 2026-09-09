"""Multi-scene Algan video template.

Render individual scenes while iterating, then concatenate the final video.
Scene functions managed by Project must not call Scene.save_video().
"""

from algan import *


def intro():
    Text("Video title", font_size=90).spawn()
    Scene.wait(1.5)


def explanation():
    with Off():
        left = Circle(color=BLUE).move(LEFT * 2).spawn()
        right = Circle(color=YELLOW).move(RIGHT * 2).spawn()

    with Sync(runtime=1.5):
        left.move(RIGHT * 1.2)
        right.move(LEFT * 1.2)

    Scene.wait(0.5)


def outro():
    Text("Takeaway", font_size=72).spawn()
    Scene.wait(1.5)


project = Project(
    [intro, explanation, outro],
    file_path="algan_video.mp4",
)


if __name__ == "__main__":
    project.run_cli()
