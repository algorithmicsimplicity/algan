"""Multi-scene project authoring and output management."""

from __future__ import annotations

import argparse
import inspect
import os
import re
import textwrap
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from algan.errors import AlganConfigurationError
from algan.logging.logger import get_logger
from algan.settings import SETTINGS
from algan.settings.video_settings import VideoSettings

logger = get_logger()
_ACTIVE_PROJECT_RUN = ContextVar("algan_active_project_run", default=None)


def _get_active_project_run():
    return _ACTIVE_PROJECT_RUN.get()


@dataclass(frozen=True)
class _ProjectScene:
    id: int
    name: str
    function: object

    @property
    def stem(self) -> str:
        return f"{self.id}_{self.name}"


@dataclass
class _ProjectSceneRun:
    project: Project
    scene: _ProjectScene
    mode: Literal["screenshots", "video"]
    next_frame_index: int = 0
    frame_results: list = field(default_factory=list)
    allow_video_render: bool = False

    @property
    def render_screenshots(self) -> bool:
        return self.mode == "screenshots"

    def prepare_frame_path(self, file_path) -> Path:
        frame_id = self.project.frame_id(self.scene.id, self.next_frame_index)
        self.next_frame_index += 1

        if file_path is None:
            file_path = SETTINGS.paths.output_filename
        raw_path = os.fspath(file_path)
        requested = Path(raw_path)
        if requested.suffix == "":
            requested = requested.with_suffix(".png")
        requested = requested.with_name(f"{frame_id}_{requested.name}")

        # Match Scene's path contract: a bare name uses the configured project
        # screenshot directory, while an explicit parent remains explicit.
        if not requested.is_absolute() and os.path.dirname(raw_path) == "":
            requested = self.project.screenshot_directory / requested
        return requested

    def record_frame_results(self, result) -> None:
        if isinstance(result, list):
            self.frame_results.extend(result)
        else:
            self.frame_results.append(result)


class Project:
    """Coordinate a fixed collection of independently-authored Scene functions.

    Scene IDs are their zero-based positions in ``scene_functions``. Output
    stems are ``<id>_<name>``. Frame IDs are derived from the scene ID and the
    save-frame call's local index, so rendering any subset produces the same
    names as rendering the whole project.

    Parameters
    ----------
    scene_functions
        A nonempty iterable of callables. Each callable must be invokable with
        no arguments. Optional parameters are allowed.
    file_path
        Destination for :meth:`concatenate_videos`. It follows
        :meth:`Scene.save_video <algan.scene.Scene.save_video>` path rules and
        defaults to the main script's configured output filename.
    video_directory, screenshot_directory, transcript_directory
        Output directories. A bare directory name is placed under Algan's
        standard output directory; a path with an explicit parent is used as
        supplied.
    transcript_line_length
        Maximum number of characters per transcript line.
    video_settings
        Optional settings used to author and render every project Scene.
    speech_source
        Optional speech generator installed on each Scene's audio manager.
    """

    def __init__(
        self,
        scene_functions,
        video_settings: VideoSettings | None = None,
        file_path=None,
        *,
        video_directory="videos",
        screenshot_directory="screenshots",
        transcript_directory="transcripts",
        transcript_line_length: int = 88,
        speech_source=None,
    ):
        try:
            functions = tuple(scene_functions)
        except TypeError as exc:
            raise AlganConfigurationError(
                "Project scene_functions must be a nonempty iterable of callables"
            ) from exc
        if not functions:
            raise AlganConfigurationError("Project scene_functions cannot be empty")
        if (
            isinstance(transcript_line_length, bool)
            or not isinstance(transcript_line_length, int)
            or transcript_line_length <= 0
        ):
            raise AlganConfigurationError(
                "transcript_line_length must be a positive integer"
            )

        scenes = []
        seen_names = set()
        for scene_id, function in enumerate(functions):
            if not callable(function):
                raise AlganConfigurationError(
                    f"Project scene entry {scene_id} is not callable"
                )
            try:
                signature = inspect.signature(function)
            except (TypeError, ValueError) as exc:
                raise AlganConfigurationError(
                    f"Could not inspect Project scene entry {scene_id}"
                ) from exc
            required = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            if required:
                names = ", ".join(parameter.name for parameter in required)
                raise AlganConfigurationError(
                    f"Project scene entry {scene_id} requires arguments: {names}"
                )

            if getattr(function, "__algan_scene__", False):
                name = getattr(
                    function,
                    "name",
                    getattr(function, "__algan_scene_name__", None),
                )
            else:
                name = getattr(function, "__name__", None)
            if not isinstance(name, str) or not name:
                raise AlganConfigurationError(
                    f"Project scene entry {scene_id} does not have a valid name"
                )
            if name in seen_names:
                raise AlganConfigurationError(
                    f"Project scene names must be unique; found {name!r} more than once"
                )
            seen_names.add(name)
            scenes.append(_ProjectScene(scene_id, name, function))

        self.scene_functions = functions
        self._scenes = tuple(scenes)
        self.scene_names = tuple(scene.stem for scene in self._scenes)
        self.video_directory = self._resolve_directory(video_directory)
        self.screenshot_directory = self._resolve_directory(screenshot_directory)
        self.transcript_directory = self._resolve_directory(transcript_directory)
        self.transcript_line_length = transcript_line_length
        self.video_settings = video_settings
        self.speech_source = speech_source

        from algan.utils.algan_utils import _resolve_output_destination

        self.file_path = _resolve_output_destination(file_path, ".mp4")
        self.global_transcript_path = self.transcript_directory / "transcript.txt"
        self._scene_video_paths = tuple(
            self.video_directory / f"{scene.stem}.mp4" for scene in self._scenes
        )
        normalized_file_path = self._normalized_path(self.file_path)
        for scene, scene_path in zip(self._scenes, self._scene_video_paths):
            if normalized_file_path == self._normalized_path(scene_path):
                raise AlganConfigurationError(
                    f"Project file_path collides with scene path {scene.stem!r}"
                )

    @staticmethod
    def _resolve_directory(directory) -> Path:
        raw_path = os.fspath(directory)
        requested = Path(raw_path)
        if not requested.is_absolute() and os.path.dirname(raw_path) == "":
            requested = (
                Path(SETTINGS.paths.output_root)
                / SETTINGS.paths.output_directory
                / requested
            )
        return requested

    @staticmethod
    def _normalized_path(path: Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def _scene_for_selector(self, selector) -> _ProjectScene:
        if isinstance(selector, bool):
            raise AlganConfigurationError(
                f"Invalid Project scene selector: {selector!r}"
            )
        if isinstance(selector, int):
            if 0 <= selector < len(self._scenes):
                return self._scenes[selector]
            raise AlganConfigurationError(f"Unknown Project scene ID: {selector}")
        if isinstance(selector, str):
            for scene in self._scenes:
                if selector == scene.name or selector == scene.stem:
                    return scene
            match = re.fullmatch(r"(\d+)_(.*)", selector)
            if match:
                scene_id = int(match.group(1))
                if 0 <= scene_id < len(self._scenes):
                    expected = self._scenes[scene_id]
                    raise AlganConfigurationError(
                        f"Project scene selector {selector!r} does not match "
                        f"{expected.stem!r}"
                    )
            raise AlganConfigurationError(f"Unknown Project scene name: {selector!r}")
        raise AlganConfigurationError(f"Invalid Project scene selector: {selector!r}")

    def _selected_scenes(self, selectors) -> tuple[_ProjectScene, ...]:
        if selectors is None:
            return self._scenes
        if isinstance(selectors, (str, int)) and not isinstance(selectors, bool):
            selectors = (selectors,)
        else:
            try:
                selectors = tuple(selectors)
            except TypeError as exc:
                raise AlganConfigurationError(
                    f"Invalid Project scene selection: {selectors!r}"
                ) from exc
        selected_ids = {self._scene_for_selector(selector).id for selector in selectors}
        return tuple(scene for scene in self._scenes if scene.id in selected_ids)

    def frame_id(self, scene, local_frame_index: int) -> int:
        """Return the stable project-global ID for one Scene save-frame call."""
        project_scene = self._scene_for_selector(scene)
        if (
            isinstance(local_frame_index, bool)
            or not isinstance(local_frame_index, int)
            or local_frame_index < 0
        ):
            raise AlganConfigurationError(
                "local_frame_index must be a non-negative integer"
            )
        return local_frame_index * len(self._scenes) + project_scene.id

    def _format_transcript(self, raw_transcript: str) -> str:
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n", raw_transcript.strip()):
            normalized = " ".join(paragraph.split())
            if normalized:
                paragraphs.append(
                    textwrap.fill(normalized, width=self.transcript_line_length)
                )
        return "\n\n".join(paragraphs) + ("\n" if paragraphs else "")

    def _sync_scene_transcript(self, scene: _ProjectScene, raw_transcript: str) -> None:
        transcript_path = self.transcript_directory / f"{scene.stem}.txt"
        transcript = self._format_transcript(raw_transcript)
        adjusted = False
        if transcript:
            previous = (
                transcript_path.read_text(encoding="utf-8")
                if transcript_path.exists()
                else None
            )
            if previous != transcript:
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_path.write_text(transcript, encoding="utf-8")
                adjusted = True
        elif transcript_path.exists():
            transcript_path.unlink()
            adjusted = True

        if adjusted:
            self._update_global_transcript()

    def _update_global_transcript(self) -> None:
        transcripts = []
        for scene in self._scenes:
            transcript_path = self.transcript_directory / f"{scene.stem}.txt"
            if transcript_path.exists():
                transcript = transcript_path.read_text(encoding="utf-8").strip()
                if transcript:
                    transcripts.append(transcript)
        combined = "\n\n".join(transcripts) + ("\n" if transcripts else "")
        if combined or self.global_transcript_path.exists():
            self.global_transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.global_transcript_path.write_text(combined, encoding="utf-8")

    def render_screenshots(
        self,
        scenes=None,
        *,
        video_settings: VideoSettings | None = None,
    ):
        """Run save-frame calls for one, many, or all project scenes.

        No scene videos are rendered. ``scenes`` accepts an ID, an unprefixed
        name, a full prefixed name, an iterable mixing those forms, or ``None``
        for all scenes.
        """
        return self._render(
            scenes,
            mode="screenshots",
            video_settings=video_settings,
        )

    def run_cli(self, argv=None) -> bool:
        """Run the project action requested by command-line arguments.

        ``argv`` defaults to the current process arguments (excluding the
        executable/script name). The recognized arguments are::

            --render-screenshots [SCENE ...]
            --render-video [SCENE ...]
            --concatenate-videos

        Scene values accept the same IDs and names as :meth:`render_video` and
        :meth:`render_screenshots`. Omitting them renders every scene. Unknown
        arguments are ignored so this can be called from scripts launched by
        tools that add their own command-line options.

        Returns ``True`` after dispatching a recognized action and ``False``
        when no project action was present.
        """
        parser = argparse.ArgumentParser(add_help=False)
        actions = parser.add_mutually_exclusive_group()
        actions.add_argument(
            "--render-screenshots",
            nargs="*",
            metavar="SCENE",
        )
        actions.add_argument(
            "--render-video",
            nargs="*",
            metavar="SCENE",
        )
        actions.add_argument(
            "--concatenate-videos",
            action="store_true",
        )
        arguments, _unknown = parser.parse_known_args(argv)

        if arguments.render_screenshots is not None:
            scenes = self._parse_cli_scene_selectors(arguments.render_screenshots)
            self.render_screenshots(scenes)
            return True
        if arguments.render_video is not None:
            scenes = self._parse_cli_scene_selectors(arguments.render_video)
            self.render_video(scenes)
            return True
        if arguments.concatenate_videos:
            self.concatenate_videos()
            return True
        return False

    @staticmethod
    def _parse_cli_scene_selectors(selectors):
        if not selectors:
            return None
        return tuple(
            int(selector) if re.fullmatch(r"\d+", selector) else selector
            for selector in selectors
        )

    def render_video(
        self,
        scenes=None,
        *,
        video_settings: VideoSettings | None = None,
        overwrite: bool = True,
        **save_video_kwargs,
    ):
        """Render full videos for one, many, or all project scenes.

        Scene save-frame calls are skipped. ``scenes`` accepts an ID, an
        unprefixed name, a full prefixed name, an iterable mixing those forms,
        or ``None`` for all scenes.
        """
        return self._render(
            scenes,
            mode="video",
            video_settings=video_settings,
            overwrite=overwrite,
            **save_video_kwargs,
        )

    def _render(
        self,
        scenes=None,
        *,
        mode: Literal["screenshots", "video"],
        video_settings: VideoSettings | None = None,
        overwrite: bool = True,
        **save_video_kwargs,
    ):
        """Shared implementation for screenshot and video project renders."""
        selected = self._selected_scenes(scenes)
        effective_settings = video_settings or self.video_settings or SETTINGS.video
        results = []

        from algan.scene import Scene

        for project_scene in selected:
            with Scene(video_settings=effective_settings) as active_scene:
                run = _ProjectSceneRun(self, project_scene, mode)
                active_scene._project_run = run
                active_scene._suppress_automatic_transcript = True
                active_scene.audio_manager.set_speech_source(self.speech_source)
                run_token = _ACTIVE_PROJECT_RUN.set(run)
                try:
                    project_scene.function()
                    self._sync_scene_transcript(
                        project_scene,
                        active_scene.audio_manager.video_transcript,
                    )
                    if mode == "video":
                        run.allow_video_render = True
                        try:
                            result = active_scene.save_video(
                                self._scene_video_paths[project_scene.id],
                                video_settings=effective_settings,
                                overwrite=overwrite,
                                **save_video_kwargs,
                            )
                        finally:
                            run.allow_video_render = False
                        results.append(result)
                    else:
                        results.extend(run.frame_results)
                finally:
                    _ACTIVE_PROJECT_RUN.reset(run_token)
                    active_scene._project_run = None
                    active_scene._suppress_automatic_transcript = False
            logger.info(
                f"Finished rendering project scene {project_scene.stem} in {mode} mode"
            )
        return results

    def concatenate_videos(self, *, threads: int | None = None, reencode=False):
        """Concatenate the project's scene videos into ``file_path``."""
        from algan.utils.algan_utils import concatenate_videos

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        return concatenate_videos(
            os.fspath(self.video_directory),
            threads=threads,
            reencode=reencode,
            output_file=os.fspath(self.file_path.resolve()),
            input_files=tuple(
                os.fspath(scene_path.resolve())
                for scene_path in self._scene_video_paths
            ),
        )


__all__ = ["Project"]
