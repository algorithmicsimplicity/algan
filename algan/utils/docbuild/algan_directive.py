r"""Sphinx directive for rendering inline Algan examples.

The directive executes its indented Python body in an isolated namespace while
building HTML documentation. The example is rendered with a documentation
quality preset and its output is embedded beside the source code::

    .. algan:: MovingSquare
        :quality: low

        from algan import *

        Square().spawn().move(RIGHT)
        Scene.save_video()

Use the ``skip-manim`` Sphinx tag to replace rendered examples with source-code
placeholders. The historical tag name is retained for compatibility with the
documentation toolchain.
"""

from __future__ import annotations

import csv
import itertools as it
import re
import shutil
import subprocess
import textwrap
import traceback
from pathlib import Path
from timeit import timeit
from typing import TYPE_CHECKING, Any, TypedDict

import jinja2
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList

from algan import SceneManager, __version__ as algan_version
from algan.settings import SETTINGS
from algan.settings.video_settings import QUALITIES

if TYPE_CHECKING:
    from sphinx.application import Sphinx

__all__ = ["AlganDirective"]


class SetupMetadata(TypedDict):
    parallel_read_safe: bool
    parallel_write_safe: bool


class SkipAlganNode(nodes.Admonition, nodes.Element):
    """Placeholder used when embedded example rendering is disabled."""


def visit(self: SkipAlganNode, node: nodes.Element, name: str = "") -> None:
    self.visit_admonition(node, name)  # type: ignore[attr-defined]
    if not isinstance(node[0], nodes.title):
        node.insert(0, nodes.title("skip-algan", "Example Placeholder"))


def depart(self: SkipAlganNode, node: nodes.Element) -> None:
    self.depart_admonition(node)  # type: ignore[attr-defined]


def process_name_list(option_input: str, reference_type: str) -> list[str]:
    """Convert space-separated names to compact Sphinx references."""
    return [f":{reference_type}:`~.{name}`" for name in option_input.split()]


def _find_video(video_dir: Path, output_file: str) -> Path:
    for suffix in (".mp4", ".mov", ".webm"):
        candidate = video_dir / f"{output_file}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"The Algan example did not create {output_file}.mp4/.mov/.webm in "
        f"{video_dir}. End the directive body with Scene.save_video()."
    )


def _run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *arguments],
        check=True,
    )


class AlganDirective(Directive):
    """Render an inline Algan script and embed its output."""

    has_content = True
    required_arguments = 1
    optional_arguments = 0
    option_spec = {
        "hide_source": bool,
        "no_autoplay": bool,
        "quality": lambda arg: directives.choice(
            arg,
            ("low", "medium", "high", "fourk"),
        ),
        "save_as_gif": bool,
        "save_last_frame": bool,
        "ref_modules": lambda arg: process_name_list(arg, "mod"),
        "ref_classes": lambda arg: process_name_list(arg, "class"),
        "ref_functions": lambda arg: process_name_list(arg, "func"),
        "ref_methods": lambda arg: process_name_list(arg, "meth"),
    }
    final_argument_whitespace = True

    def run(self) -> list[nodes.Element]:
        environment = self.state.document.settings.env
        should_skip = (
            "skip-manim" in environment.app.builder.tags
            or environment.app.builder.name == "gettext"
        )
        if should_skip:
            node = SkipAlganNode()
            name = self.arguments[0]
            self.state.nested_parse(
                StringList(
                    [
                        f"Placeholder block for ``{name}``.",
                        "",
                        ".. code-block:: python",
                        "",
                        *["    " + line for line in self.content],
                    ]
                ),
                self.content_offset,
                node,
            )
            return [node]

        if not self.content:
            raise self.error("The algan directive requires a Python body.")

        clsname = self.arguments[0]
        occurrence = _next_occurrence(clsname)
        output_file = f"{clsname}-{occurrence}"

        hide_source = "hide_source" in self.options
        no_autoplay = "no_autoplay" in self.options
        save_as_gif = "save_as_gif" in self.options
        save_last_frame = "save_last_frame" in self.options
        if save_as_gif and save_last_frame:
            raise self.error("save_as_gif and save_last_frame are mutually exclusive")

        quality = f"{self.options.get('quality', 'example')}_quality"
        video_settings = QUALITIES[quality]

        document = self.state_machine.document
        source_path = Path(document.attributes["source"])
        source_rel_dir = source_path.relative_to(setup.confdir).parent  # type: ignore[attr-defined]
        destination_dir = Path(
            setup.app.builder.outdir, source_rel_dir  # type: ignore[attr-defined]
        ).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)

        media_dir = Path(setup.confdir, "media").resolve()  # type: ignore[attr-defined]
        video_dir = media_dir / "videos" / quality
        image_dir = media_dir / "images"
        video_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        user_code = list(self.content)
        if user_code[0].startswith(">>> "):
            user_code = [
                line[4:] for line in user_code if line.startswith((">>> ", "... "))
            ]
        source = "\n".join(user_code)
        namespace = {"__name__": f"_algan_docs_{output_file}"}

        try:
            with SETTINGS.override(
                video=video_settings.to_dict(),
                paths={
                    "output_path": str(video_dir),
                    "output_directory": "",
                    "output_filename": output_file,
                },
            ):
                SceneManager.reset()
                try:
                    run_time = timeit(
                        lambda: exec(compile(source, str(source_path), "exec"), namespace),
                        number=1,
                    )
                finally:
                    SceneManager.reset()
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(f"Error while rendering example {clsname}") from exc

        _write_rendering_stats(
            clsname,
            run_time,
            environment.docname,
        )

        rendered_video = _find_video(video_dir, output_file)
        filesrc = rendered_video
        embedded_suffix = rendered_video.suffix

        if save_as_gif:
            filesrc = video_dir / f"{output_file}.gif"
            _run_ffmpeg("-y", "-i", str(rendered_video), str(filesrc))
            embedded_suffix = ".gif"
        elif save_last_frame:
            filesrc = image_dir / f"{output_file}.png"
            _run_ffmpeg(
                "-y",
                "-sseof",
                "-0.1",
                "-i",
                str(rendered_video),
                "-frames:v",
                "1",
                str(filesrc),
            )
            embedded_suffix = ".png"
        else:
            destination = destination_dir / f"{output_file}{embedded_suffix}"
            shutil.copyfile(rendered_video, destination)

        ref_content = (
            self.options.get("ref_modules", [])
            + self.options.get("ref_classes", [])
            + self.options.get("ref_functions", [])
            + self.options.get("ref_methods", [])
        )
        ref_block = "References: " + " ".join(ref_content) if ref_content else ""

        source_block = "\n".join(
            [
                ".. code-block:: python",
                "",
                *["    " + line for line in self.content],
            ]
        )
        filesrc_rel = filesrc.relative_to(setup.confdir).as_posix()  # type: ignore[attr-defined]
        rendered = jinja2.Template(TEMPLATE).render(
            clsname=clsname,
            clsname_lowercase=clsname.lower(),
            hide_source=hide_source,
            no_autoplay=no_autoplay,
            output_file=output_file,
            embedded_suffix=embedded_suffix,
            save_last_frame=save_last_frame,
            save_as_gif=save_as_gif,
            filesrc_rel=filesrc_rel,
            source_block=source_block,
            ref_block=ref_block,
        )
        self.state_machine.insert_input(
            rendered.split("\n"),
            source=document.attributes["source"],
        )
        return []


_class_occurrences: dict[str, int] = {}


def _next_occurrence(name: str) -> int:
    occurrence = _class_occurrences.get(name, 0) + 1
    _class_occurrences[name] = occurrence
    return occurrence


rendering_times_file_path = Path(__file__).resolve().parents[3] / "docs" / "rendering_times.csv"


def _write_rendering_stats(scene_name: str, run_time: float, file_name: str) -> None:
    with rendering_times_file_path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                re.sub(r"^(reference/)|(algan\.)", "", file_name),
                scene_name,
                f"{run_time:.3f}",
            ]
        )


def _log_rendering_times(*args: Any) -> None:
    if not rendering_times_file_path.exists():
        return
    with rendering_times_file_path.open(encoding="utf-8") as file:
        data = [row for row in csv.reader(file) if row]
    if not data:
        return

    print("\nRendering Summary\n-----------------\n")
    max_file_length = max(len(row[0]) for row in data)
    for key, group_iter in it.groupby(data, key=lambda row: row[0]):
        key = key.ljust(max_file_length + 1, ".")
        group = list(group_iter)
        if len(group) == 1:
            row = group[0]
            print(f"{key}{row[2].rjust(7, '.')}s {row[1]}")
            continue
        time_sum = sum(float(row[2]) for row in group)
        print(f"{key}{f'{time_sum:.3f}'.rjust(7, '.')}s  => {len(group)} EXAMPLES")
        for row in group:
            print(f"{' ' * max_file_length} {row[2].rjust(7)}s {row[1]}")
    print("")


def _delete_rendering_times(*args: Any) -> None:
    rendering_times_file_path.unlink(missing_ok=True)
    _class_occurrences.clear()


def setup(app: Sphinx) -> SetupMetadata:
    app.add_node(SkipAlganNode, html=(visit, depart))

    setup.app = app  # type: ignore[attr-defined]
    setup.confdir = app.confdir  # type: ignore[attr-defined]

    app.add_directive("algan", AlganDirective)
    app.connect("builder-inited", _delete_rendering_times)
    app.connect("build-finished", _log_rendering_times)

    app.add_js_file("manim-binder.min.js")
    app.add_js_file(
        None,
        body=textwrap.dedent(
            f"""\
                window.initManimBinder({{branch: "v{algan_version}"}})
            """
        ).strip(),
    )

    return {
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }


TEMPLATE = r"""
{% if not hide_source %}
.. raw:: html

    <div id="{{ clsname_lowercase }}" class="admonition admonition-manim-example">
    <p class="admonition-title">Example: {{ clsname }} <a class="headerlink" href="#{{ clsname_lowercase }}">¶</a></p>

{% endif %}

{% if not (save_as_gif or save_last_frame) %}
.. raw:: html

    <video
        class="manim-video"
        controls
        loop
        {{ '' if no_autoplay else 'autoplay' }}
        src="./{{ output_file }}{{ embedded_suffix }}">
    </video>

{% else %}
.. image:: /{{ filesrc_rel }}
    :align: center
{% endif %}

{% if not hide_source %}
{{ source_block }}

{{ ref_block }}

.. raw:: html

    </div>
{% endif %}
"""
