"""Two independently rendered scenes; Project owns their exports.

Run with explicit actions, for example:
  python multi_scene_project.py --render-video --video-settings PREVIEW
  python multi_scene_project.py --render-video --video-settings HD
  python multi_scene_project.py --concatenate-videos

Scene functions must NOT call Scene.save_video(). The subject/motion choices
below only demonstrate mechanics, not a proposed sequence for the user's video.
"""

from algan import Off, OUT, Project, RIGHT, Scene, Seq, Square, Triangle


def scene_a():
    with Off():
        subject = Square().scale(0.5).spawn()
    with Seq(runtime=1.0):
        subject.move(RIGHT)
    Scene.wait(0.25)


def scene_b():
    with Off():
        subject = Triangle().scale(0.5).spawn()
    with Seq(runtime=1.0):
        subject.rotate(90, OUT)
    Scene.wait(0.25)


if __name__ == "__main__":
    project = Project([scene_a, scene_b], file_path="renders/complete.mp4")
    dispatched = project.run_cli()
    if not dispatched:
        print("No project action selected. Use --help or --render-video.")
