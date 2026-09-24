"""Driver for a multi-scene narrated project (template: mechanics only).

Layout:

    project.py        this driver
    common.py         settings, post-processing, narration helper, the user's shared props
    scenes/*.py       one zero-argument authoring function per scene
    script.txt        the user's exact narration (optional, for --script checking)

Every action is Algan's own ``Project.run_cli()``; this file only chooses the draft
or final settings and hands the post-processing passes to the Project, so videos,
stills and profiles all use them. Usage (python = the interpreter that has Algan):

    python project.py --validate [--script script.txt]     author every scene, report
                                                            timing, check the narration
    python project.py --render-screenshots [SCENE ...] [--frames NAME ...] [--contact-sheet]
    python project.py --render-video [SCENE ...]           scene videos at DRAFT
    python project.py --render-video [SCENE ...] --final   scene videos at FINAL
    python project.py --concatenate-videos [--final]       join the scene videos rendered
                                                            at those settings

SCENE is a scene function name (for example ``first``), its prefixed name
(``0_first``) or its zero-based index. ``python project.py --help`` lists the rest.
"""

from algan import *

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import DRAFT, FINAL, POST  # noqa: E402
from scenes.first import first  # noqa: E402
from scenes.second import second  # noqa: E402

SCENES = [first, second]
OUT = os.path.join(HERE, 'renders')


def make_project(settings):
    return Project(SCENES, video_settings=settings,
                   file_path=os.path.join(OUT, 'video.mp4'),
                   video_directory=os.path.join(OUT, 'scenes'),
                   screenshot_directory=os.path.join(OUT, 'stills'),
                   transcript_directory=os.path.join(OUT, 'transcripts'),
                   post_processes=POST)


if __name__ == '__main__':
    final = '--final' in sys.argv[1:]
    argv = [arg for arg in sys.argv[1:] if arg != '--final']
    if not make_project(FINAL if final else DRAFT).run_cli(argv):
        print(__doc__)
        sys.exit(2)
