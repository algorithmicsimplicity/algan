"""Multi-scene project authoring and output management."""

from __future__ import annotations

import argparse
import fnmatch
import inspect
import math
import os
import re
import textwrap
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from algan.errors import AlganConfigurationError
from algan.logging.logger import get_logger
from algan.settings import SETTINGS
from algan.settings.video_settings import _PRESETS_BY_NAME, VideoSettings

if TYPE_CHECKING:
    from algan.viewer.viewer import ViewerHandle

logger = get_logger()
_ACTIVE_PROJECT_RUN = ContextVar("algan_active_project_run", default=None)
_SceneSelection = int | str | Iterable[int | str] | None


@dataclass(frozen=True)
class SceneValidation:
    """Authored scene report: ID, name, seconds, checkpoint count and exact speech text."""

    id: int
    name: str
    duration_seconds: float
    checkpoints: int
    transcript: str


def _describe_script_mismatch(scenes, expected_words, context=6):
    """Explain where narration first departs from a script, or return None.

    Names the scene holding the differing narration word and shows a few words
    either side in both texts, since a bare word number is hard to find in a
    long script. Past the end of the narration, the last scene is named.
    """
    actual_words, owners = [], []
    for scene in scenes:
        words = scene.transcript.split()
        actual_words.extend(words)
        owners.extend((scene.name, local) for local in range(len(words)))
    if actual_words == expected_words:
        return None
    index = next(
        (
            i
            for i, (spoken, wanted) in enumerate(zip(actual_words, expected_words))
            if spoken != wanted
        ),
        min(len(actual_words), len(expected_words)),
    )

    def show(words, end_label):
        before = words[max(0, index - context) : index]
        word = words[index] if index < len(words) else end_label
        after = words[index + 1 : index + 1 + context]
        prefix = "... " if index > context else ""
        suffix = " ..." if index + 1 + context < len(words) else ""
        return f"{prefix}{' '.join([*before, f'[{word}]', *after])}{suffix}"

    wanted = expected_words[index] if index < len(expected_words) else "<end of script>"
    spoken = actual_words[index] if index < len(actual_words) else "<end of narration>"
    if index < len(owners):
        name, local = owners[index]
        where = f"in scene {name!r} (word {local + 1} of that scene)"
    elif scenes:
        where = f"after the end of scene {scenes[-1].name!r}"
    else:
        where = "with no narration in the selected scenes"
    return (
        f"Narration differs from script at word {index + 1}: expected "
        f"{wanted!r}, got {spoken!r}, {where}.\n"
        f"  script:    {show(expected_words, '<end of script>')}\n"
        f"  narration: {show(actual_words, '<end of narration>')}\n"
        f"({len(expected_words)} script words, {len(actual_words)} narration words)"
    )


@dataclass(frozen=True)
class ProjectValidation:
    """Results from authoring selected scenes without producing images or videos."""

    scenes: tuple[SceneValidation, ...]

    @property
    def duration_seconds(self) -> float:
        """Total authored duration in seconds, including Speech holds."""
        return sum(scene.duration_seconds for scene in self.scenes)


@dataclass(frozen=True)
class RenderEstimate:
    """Sample-based export estimate, in seconds, excluding authoring and concatenation.

    ``warm_seconds`` scales the samples' weighted warm render rate to the
    project duration. ``cold_overhead_seconds`` sums observed first-pass excess
    over warm passes; it includes cache, compilation and clock effects.
    ``range_seconds`` applies the fastest/slowest sampled scene rates plus that
    overhead, and is a workload range, not a statistical confidence interval.
    Unseen shaders, scene density, settings and hardware can invalidate it.
    ``profiles`` contains the underlying per-scene ``profile_scene`` results.
    """

    duration_seconds: float
    warm_seconds: float
    cold_overhead_seconds: float
    range_seconds: tuple[float, float]
    profiles: dict[str, list[dict]]

    @property
    def estimated_seconds(self) -> float:
        """Warm export estimate plus observed startup overhead, in seconds."""
        return self.warm_seconds + self.cold_overhead_seconds


def _get_active_project_run():
    return _ACTIVE_PROJECT_RUN.get()


class _StopSceneEarly(BaseException):
    """Abandon a scene function once every requested frame has been queued.

    Derived from BaseException rather than Exception on purpose. It unwinds
    through arbitrary user scene code, and a scene that wraps its own work in
    ``except Exception`` would otherwise swallow the abort and carry on
    authoring the rest of itself -- which is the one thing this exists to avoid.
    """


def _frame_pattern_matches(
    pattern, frame_id: int, local_name: str, full_name: str
) -> bool:
    """Does one frame-selection pattern pick out this save-frame call?

    An all-digit pattern is the frame's index within its scene -- the ``3`` in
    ``s1_f3_thing.png``. A pattern holding a glob metacharacter is matched with
    fnmatch. Anything else is a plain substring, so ``perturbed`` finds
    ``s1_f4_s05_perturbed_output_distance`` without anyone having to spell out
    the prefix. Names are matched with and without that prefix, and case is
    ignored throughout.
    """
    if pattern.isdigit():
        return int(pattern) == frame_id
    pattern = pattern.lower()
    names = (local_name.lower(), full_name.lower())
    if any(character in pattern for character in "*?["):
        return any(fnmatch.fnmatchcase(name, pattern) for name in names)
    return any(pattern in name for name in names)


def _video_settings_for_name(name: str) -> VideoSettings:
    """Resolve a ``--video-settings`` preset name, case-insensitively."""
    try:
        return _PRESETS_BY_NAME[str(name).strip().upper()]
    except KeyError:
        raise AlganConfigurationError(
            f"Unknown video settings preset: {name!r}. Choose one of: "
            f"{', '.join(_PRESETS_BY_NAME)}"
        ) from None


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
    mode: Literal["screenshots", "video", "validate", "profile", "view", "subtitles"]
    next_frame_index: int = 0
    frame_results: list = field(default_factory=list)
    allow_video_render: bool = False
    frame_patterns: tuple[str, ...] = ()
    stop_early: bool = False
    matched_patterns: set = field(default_factory=set)
    stopped_early: bool = False
    last_frame_name: str = ""
    _render_current_frame: bool = False
    frame_options: dict = field(default_factory=dict)
    frame_batches: list = field(default_factory=list)

    @property
    def render_screenshots(self) -> bool:
        return self.mode == "screenshots"

    def prepare_frame_path(self, file_path) -> Path:
        scene_id = self.scene.id
        frame_id = (
            self.next_frame_index
        )  # self.project.frame_id(self.scene.id, self.next_frame_index)
        self.next_frame_index += 1

        if file_path is None:
            file_path = SETTINGS.paths.output_filename
        raw_path = os.fspath(file_path)
        # ``expanduser`` before the directory probe, as
        # ``_resolve_output_destination`` does: ``Path("~/stills").is_dir()`` is
        # False however real the directory is, so an unexpanded ``~`` would be
        # read as a file name here and written to a literal ``~`` directory.
        requested = Path(raw_path).expanduser()
        is_directory = requested.is_dir() or raw_path.endswith(
            tuple(separator for separator in (os.sep, os.altsep) if separator)
        )
        if is_directory:
            requested = requested / SETTINGS.paths.output_filename
        if requested.suffix == "":
            requested = requested.with_suffix(".png")
        local_name = requested.stem
        requested = requested.with_name(f"s{scene_id}_f{frame_id}_{requested.name}")

        # Whether this frame survives the selection is settled here, where its
        # index and both spellings of its name are all in hand. save_frame only
        # has to ask the answer.
        self.last_frame_name = requested.stem
        self._render_current_frame = self.render_screenshots and self._frame_selected(
            frame_id, local_name, requested.stem
        )

        # Match Scene's path contract: a bare name uses the configured project
        # screenshot directory, while an explicit parent remains explicit.
        if (
            not is_directory
            and not requested.is_absolute()
            and os.path.dirname(raw_path) == ""
        ):
            requested = self.project.screenshot_directory / requested
        return requested

    def _frame_selected(self, frame_id, local_name, full_name) -> bool:
        if not self.frame_patterns:
            return True
        selected = False
        for pattern in self.frame_patterns:
            if _frame_pattern_matches(pattern, frame_id, local_name, full_name):
                self.matched_patterns.add(pattern)
                selected = True
        return selected

    def should_render_frame(self) -> bool:
        """Whether the save-frame call being served should actually render."""
        return self._render_current_frame

    def queue_frames(self, batch) -> None:
        if self.frame_batches and self.frame_batches[-1].compatible_with(batch):
            self.frame_batches[-1].targets.extend(batch.targets)
        else:
            self.frame_batches.append(batch)

    def render_frames(self) -> None:
        # Run only after successful authoring (or the requested early stop).
        # Replacing placeholders keeps the public results in checkpoint order.
        self.frame_results = []
        try:
            for batch in self.frame_batches:
                self.frame_results.extend(batch.render())
        finally:
            self.frame_batches.clear()

    def record_frame_results(self, result) -> None:
        if isinstance(result, list):
            self.frame_results.extend(result)
        else:
            self.frame_results.append(result)
        if (
            self.stop_early
            and self.frame_patterns
            and len(self.matched_patterns) == len(self.frame_patterns)
        ):
            # Every pattern has been queued; render them after unwinding.
            # Unwinding here rather than at the next save_frame
            # is what skips authoring the whole tail of a long scene.
            self.stopped_early = True
            raise _StopSceneEarly


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
    video_settings
        Settings used to author and render every project Scene. Defaults to
        None, using SETTINGS.video.
    file_path
        Destination for :meth:`concatenate_videos`. It follows
        :meth:`Scene.save_video <algan.scene.Scene.save_video>` path rules and
        defaults to the main script's configured output filename.
    video_directory, screenshot_directory, transcript_directory
        Output directories. A bare directory name is placed under Algan's
        standard output directory; a path with an explicit parent is used as
        supplied. Default to ``"videos"``, ``"screenshots"`` and
        ``"transcripts"``, respectively.
    transcript_line_length
        Maximum number of characters per transcript line. Defaults to 88.
    speech_source
        Speech generator installed on each Scene's audio manager. Defaults to
        None, using the default speech source.
    post_processes
        Sequence of frame-processing callables, as for :meth:`Scene.save_video`.
        Used for videos, screenshots and profiling, including :meth:`run_cli`.
        Defaults to None, using the renderer's defaults. An empty sequence
        disables post-processing; explicit export options override this default.

    Animation
    ---------
    Construction records nothing. Each requested operation authors its selected
    scene functions in isolated Scenes before rendering or validating them.

    Examples
    --------
    .. code-block:: python

        from functools import partial
        from algan import Project, Scene, Square
        from algan.rendering.post_processing.bloom import bloom_filter


        def intro():
            Square().spawn()
            Scene.save_frame("opening")


        project = Project(
            [intro], post_processes=[partial(bloom_filter, glow_spread=0.015)]
        )
        project.run_cli()
    """

    def __init__(
        self,
        scene_functions: Iterable[Callable[[], None]],
        video_settings: VideoSettings | None = None,
        file_path: str | Path | None = None,
        *,
        video_directory: str | Path = "videos",
        screenshot_directory: str | Path = "screenshots",
        transcript_directory: str | Path = "transcripts",
        transcript_line_length: int = 88,
        speech_source: Callable | None = None,
        post_processes: Sequence[Callable] | None = None,
    ) -> None:
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
        from algan.render_loop import _check_post_processes

        if callable(post_processes):
            _check_post_processes(post_processes)
        self.post_processes = None if post_processes is None else tuple(post_processes)
        _check_post_processes(self.post_processes)
        self.last_contact_sheet_path: Path | None = None

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

    def view(
        self,
        scenes: _SceneSelection = None,
        *,
        video_settings: VideoSettings | None = None,
        port: int = 0,
        open_browser: bool = True,
        block: bool = True,
    ) -> ViewerHandle:
        """Open the project in an interactive viewer with one tab per scene.

        Open with no scene selected or loaded. Tabs use the project's prefixed
        scene names and project order. Clicking a tab authors only that scene
        on its first visit, then displays it. Subsequent visits reuse its
        recording without running its authoring code again. Selecting a tab
        stops playback and starts that scene at time zero, with its own frame
        rate, hierarchy, attributes, pixel inspector and synchronized transcript.
        Only the selected scene renders; switching releases the previous
        scene's frame cache, so returning to it renders its frames again.

        Animation
        ---------
        Does not run any scene function until its tab is selected. That scene's
        animations and Speech blocks are recorded normally; mobs must still be
        spawned to appear. Embedded save-frame, save-video and Scene.view calls
        are skipped. No images, videos or transcript files are exported, though
        speech generation may populate its normal cache. Viewing preserves the
        authored scenes and restores the caller's active Scene.

        Parameters
        ----------
        scenes
            Scene ID, name, prefixed name, or iterable mixing these forms.
            Defaults to None, offering a tab for every scene, in project order.
            This chooses which tabs appear, not which scenes are loaded.
        video_settings
            Settings used for authoring and viewing. Defaults to None, meaning
            author with the project's settings or SETTINGS.video, then preview
            each scene at PREVIEW resolution and its own frame rate. An explicit
            preset such as HD overrides both authoring and viewing settings.
        port
            Local server port. Defaults to 0, meaning any free port.
        open_browser
            Whether to open the viewer in the default browser. Defaults to True.
        block
            Whether to serve until Ctrl-C. Defaults to True. False returns a
            running handle for a REPL or test; call its stop() method to close
            the viewer. A blocking viewer occupies the warm render daemon
            until it stops.

        Returns
        -------
        ViewerHandle
            The viewer's URL and stop() method; also usable as a context manager.

        Raises
        ------
        AlganConfigurationError
            If the tab selection is empty or a selector is invalid. Authoring
            failures and invalid scene durations are reported in the viewer
            when that scene's tab is selected; other tabs remain usable.

        See Also
        --------
        Scene.view : Inspect a single already-authored scene.
        render_video : Export the selected scenes instead of opening a viewer.

        Examples
        --------
        .. code-block:: python

            from algan import Project, Scene, Square, Circle, RIGHT


            def intro():
                Square().spawn().move(RIGHT)


            def outro():
                Circle().spawn()
                Scene.wait(1)


            project = Project([intro, outro])
            project.view()  # tabs: 0_intro and 1_outro
        """
        from algan.scene import _note_render_requested
        from algan.viewer.viewer import _view_project

        selected = self._selected_scenes(scenes)
        if not selected:
            raise AlganConfigurationError("Project.view needs at least one scene")
        _note_render_requested()
        # Capture defaults without constructing a Scene. Tab order must not
        # make one scene inherit another's render settings during authoring.
        author_settings = (
            video_settings or self.video_settings or SETTINGS.video
        ).as_preset()
        raytracing = SETTINGS.raytracing.to_dict()

        def load_scene(scene_id):
            previous = SETTINGS.raytracing.to_dict()
            try:
                SETTINGS.raytracing._restore(raytracing)
                authored = self._render(
                    scene_id, mode="view", video_settings=author_settings
                )
                _, _, scene, settings = authored[0]
                return scene, settings
            finally:
                SETTINGS.raytracing._restore(previous)

        return _view_project(
            [(scene.id, scene.stem) for scene in selected],
            load_scene,
            video_settings,
            port=port,
            open_browser=open_browser,
            block=block,
        )

    def render_screenshots(
        self,
        scenes=None,
        *,
        frames=None,
        stop_early: bool = False,
        video_settings: VideoSettings | None = None,
        contact_sheet: bool | str | Path = False,
        contact_sheet_columns: int = 4,
        **save_frame_kwargs,
    ):
        """Author scenes, then render their collected stills in batches.

        Each scene is authored once before its selected save-frame requests are
        rendered. Compatible requests share memory-bounded batches, including
        non-consecutive timestamps; intervening video frames are not rendered.
        The result list retains checkpoint and timestamp order and stable names.
        Calls with different render options form separate batches.

        No scene videos are rendered. ``scenes`` accepts an ID, an unprefixed
        name, a full prefixed name, an iterable mixing those forms, or ``None``
        for all scenes.

        Animation
        ---------
        Authors each selected scene before rendering it. A failed authoring
        pass writes no screenshots for that scene. ``stop_early`` explicitly
        limits authoring to the requested checkpoints.

        Parameters
        ----------
        scenes
            Scene ID, name, prefixed name, or an iterable of these. Defaults to
            None, meaning every scene.
        frames
            Which save-frame calls to render, as a pattern or an iterable of
            them. A pattern is a frame index within its scene (``3``), a glob
            (``"s05_*"``), or a plain substring (``"perturbed"``); names match
            with or without the ``s<scene>_f<index>_`` prefix, case-insensitively.
            Frames that match nothing are skipped without being rendered, which
            saves the render but not the authoring. Defaults to ``None``, meaning
            every frame.
        stop_early
            Whether to abandon each scene as soon as every ``frames`` pattern has
            matched at least once, rather than authoring the rest of it. This is
            what makes iterating on an early frame of a long scene quick, but it
            leaves the scene half-run: its transcript is not synced and its later
            frames on disk stay as an earlier run left them. Requires ``frames``.
            Defaults to False. Collected requests render after unwinding.
        video_settings
            Settings used to author the scene and as checkpoint defaults.
            Defaults to None, meaning the project's settings or SETTINGS.video.
        contact_sheet
            Also combine the selected stills into a labelled image. True writes
            ``contact_sheet.png`` in the screenshot directory; a path chooses
            the destination (bare names use that directory). Labels are the
            stable checkpoint filenames. Defaults to False, writing no sheet.
        contact_sheet_columns
            Maximum thumbnails per row, a positive integer. Defaults to 4.
        **save_frame_kwargs
            Default ``background`` and ``post_processes`` for checkpoints that
            do not supply them. ``overwrite=False`` also preserves existing
            files even when an individual checkpoint allows overwriting.

        Returns
        -------
        list of RenderResult
            Completed results, in scene/checkpoint/timestamp order.

        Raises
        ------
        AlganConfigurationError
            If a scene or frame selection, render option, or timestamp is invalid.

        Examples
        --------
        .. code-block:: python

            from algan import Project, Scene


            def intro():
                Scene.wait(2)
                Scene.save_frame("intro", at=[0.5, 1.5])
                Scene.wait(1)
                Scene.save_frame("end")


            results = Project([intro]).render_screenshots()
        """
        if (
            isinstance(contact_sheet_columns, bool)
            or not isinstance(contact_sheet_columns, int)
            or contact_sheet_columns < 1
        ):
            raise AlganConfigurationError(
                "contact_sheet_columns must be a positive integer"
            )
        self.last_contact_sheet_path = None
        results = self._render(
            scenes,
            mode="screenshots",
            frames=frames,
            stop_early=stop_early,
            video_settings=video_settings,
            **save_frame_kwargs,
        )
        if contact_sheet and results:
            self.last_contact_sheet_path = self._write_contact_sheet(
                results,
                contact_sheet,
                contact_sheet_columns,
                overwrite=save_frame_kwargs.get("overwrite", True),
            )
        return results

    def _write_contact_sheet(self, results, destination, columns, *, overwrite):
        from PIL import Image, ImageDraw, ImageFont, ImageOps

        path = Path(
            "contact_sheet.png" if destination is True else destination
        ).expanduser()
        if not path.parent.parts and not path.is_absolute():
            path = self.screenshot_directory / path
        if not path.suffix:
            path = path.with_suffix(".png")
        path = path.resolve()
        sources = [Path(result.output_path).resolve() for result in results]
        if path in sources:
            raise AlganConfigurationError(
                "The contact sheet cannot overwrite a selected still"
            )
        if path.exists() and not overwrite:
            return path
        # Bound each thumbnail independently, preserving portrait and landscape
        # framing. Open one source at a time, so full-resolution stills do not
        # accumulate in memory alongside the sheet.
        font = ImageFont.load_default()
        labels = [textwrap.wrap(source.stem, 42) or [source.stem] for source in sources]
        label_height = 16 * max(map(len, labels)) + 12
        width, height, gutter = 320, 240, 12
        columns = min(columns, len(sources))
        rows = math.ceil(len(sources) / columns)
        sheet = Image.new(
            "RGB",
            (
                columns * (width + gutter) + gutter,
                rows * (height + label_height + gutter) + gutter,
            ),
            "#181c24",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (source, label) in enumerate(zip(sources, labels)):
            x = gutter + (index % columns) * (width + gutter)
            y = gutter + (index // columns) * (height + label_height + gutter)
            with Image.open(source) as original:
                thumbnail = ImageOps.contain(original.convert("RGBA"), (width, height))
                sheet.paste(
                    thumbnail,
                    (
                        x + (width - thumbnail.width) // 2,
                        y + (height - thumbnail.height) // 2,
                    ),
                    thumbnail,
                )
            draw.multiline_text(
                (x, y + height + 6),
                "\n".join(label),
                font=font,
                fill="white",
                spacing=4,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        logger.info(f"Saved contact sheet: {path}")
        return path

    def run_cli(self, argv=None) -> bool:
        """Run the project action requested by command-line arguments.

        ``argv`` defaults to the current process arguments (excluding the
        executable/script name). The recognized arguments are::

            --render-screenshots [SCENE ...] [--frames PATTERN ...] [--stop-early] [--contact-sheet [PATH]]
            --render-video [SCENE ...]
            --validate [SCENE ...] [--script PATH]
            --profile [SCENE ...] [--profile-runs N]
            --estimate-render-time SCENE ... [--profile-runs N]
            --concatenate-videos
            --video-settings PRESET
            --help

        Scene values accept the same IDs and names as :meth:`render_video` and
        :meth:`render_screenshots`. Omitting them renders every scene. Unknown
        arguments are ignored so this can be called from scripts launched by
        tools that add their own command-line options -- but ``-h``/``--help``
        is now handled here, so a script that wants its own help must parse it
        before calling this.

        Returns ``True`` after dispatching a recognized action and ``False``
        when no project action was present.
        """
        parser = argparse.ArgumentParser(
            description="Render this Algan project.",
            epilog=textwrap.dedent(
                """\
                Scenes are named by ID (0), name (intro), or stem (0_intro).
                Omit them to render every scene.

                examples:
                  --render-screenshots
                  --render-screenshots 1 --frames perturbed,initial_output
                  --render-screenshots 1 --frames "s05_*" --stop-early
                  --render-video intro --video-settings HD
                """
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        actions = parser.add_mutually_exclusive_group()
        actions.add_argument(
            "--validate",
            nargs="*",
            metavar="SCENE",
            help="Author scenes and report narration timing without rendering.",
        )
        actions.add_argument(
            "--profile",
            nargs="*",
            metavar="SCENE",
            help="Render selected scenes with stage timing reports.",
        )
        actions.add_argument(
            "--estimate-render-time",
            nargs="+",
            metavar="SCENE",
            help="Profile these reference scenes and estimate full export time.",
        )
        parser.add_argument(
            "--profile-runs",
            type=int,
            default=2,
            help="Complete profiling passes per scene (default: 2).",
        )
        actions.add_argument(
            "--render-screenshots",
            nargs="*",
            metavar="SCENE",
            help="Render the save-frame stills of these scenes.",
        )
        actions.add_argument(
            "--render-video",
            nargs="*",
            metavar="SCENE",
            help="Render videos of these scenes.",
        )
        actions.add_argument(
            "--concatenate-videos",
            action="store_true",
            help="Join the already-rendered scene videos into one file.",
        )
        parser.add_argument(
            "--frames",
            nargs="+",
            metavar="PATTERN",
            help="Screenshots only: render only the save-frame calls matching these "
            "patterns. A pattern is a frame index within its scene (3), a glob "
            "(s05_*), or a substring (perturbed). Comma-separated lists are "
            "accepted. Skipping a frame saves its render, not its authoring.",
        )
        parser.add_argument(
            "--stop-early",
            action="store_true",
            help="Screenshots only: abandon each scene once every --frames pattern "
            "has matched, instead of authoring the rest of it. Much faster when "
            "iterating on an early frame, but it leaves the scene's transcript "
            "and later frames stale, and says so loudly when it happens.",
        )
        parser.add_argument(
            "--video-settings",
            metavar="PRESET",
            help="Render at this preset instead of the project's own. One of: "
            f"{', '.join(_PRESETS_BY_NAME)} (case-insensitive).",
        )
        parser.add_argument(
            "--contact-sheet",
            nargs="?",
            const=True,
            default=False,
            metavar="PATH",
            help="Screenshots only: also write a labelled contact sheet.",
        )
        parser.add_argument(
            "--script",
            type=Path,
            metavar="PATH",
            help="Validation only: compare narration with this UTF-8 script word for word.",
        )
        arguments, _unknown = parser.parse_known_args(argv)
        if arguments.contact_sheet and arguments.render_screenshots is None:
            raise AlganConfigurationError(
                "--contact-sheet only applies to --render-screenshots"
            )
        if arguments.script is not None and arguments.validate is None:
            raise AlganConfigurationError("--script only applies to --validate")

        # Only forward what was actually asked for, so a project that overrides
        # render_screenshots/render_video with a narrower signature still works.
        shared = {}
        if arguments.video_settings is not None:
            shared["video_settings"] = _video_settings_for_name(
                arguments.video_settings
            )

        if any(
            value is not None
            for value in (
                arguments.validate,
                arguments.profile,
                arguments.estimate_render_time,
            )
        ):
            if arguments.frames or arguments.stop_early:
                raise AlganConfigurationError(
                    "--frames and --stop-early only apply to --render-screenshots"
                )
            if arguments.validate is not None:
                if arguments.script is not None:
                    shared["script"] = arguments.script
                report = self.validate(
                    self._parse_cli_scene_selectors(arguments.validate), **shared
                )
                for scene in report.scenes:
                    print(
                        f"{scene.name}: {scene.duration_seconds:.2f}s, "
                        f"{scene.checkpoints} checkpoints"
                    )
                print(
                    f"Validated {len(report.scenes)} scenes; {report.duration_seconds:.2f}s total. "
                    "No frames rendered."
                )
            elif arguments.profile is not None:
                self.profile(
                    self._parse_cli_scene_selectors(arguments.profile),
                    runs=arguments.profile_runs,
                    **shared,
                )
            else:
                estimate = self.estimate_render_time(
                    self._parse_cli_scene_selectors(arguments.estimate_render_time),
                    runs=arguments.profile_runs,
                    **shared,
                )
                print(
                    f"Estimated export: {estimate.estimated_seconds:.1f}s; "
                    f"sampled range {estimate.range_seconds[0]:.1f}-{estimate.range_seconds[1]:.1f}s. "
                    f"Observed startup overhead: {estimate.cold_overhead_seconds:.1f}s. "
                    "Excludes authoring and concatenation; unsampled effects may change the estimate."
                )
            return True
        if arguments.render_screenshots is not None:
            options = dict(shared)
            if arguments.frames:
                options["frames"] = tuple(arguments.frames)
            if arguments.stop_early:
                options["stop_early"] = True
            if arguments.contact_sheet:
                options["contact_sheet"] = arguments.contact_sheet
            scenes = self._parse_cli_scene_selectors(arguments.render_screenshots)
            self.render_screenshots(scenes, **options)
            return True
        if arguments.render_video is not None:
            if arguments.frames or arguments.stop_early:
                raise AlganConfigurationError(
                    "--frames and --stop-early only apply to --render-screenshots"
                )
            scenes = self._parse_cli_scene_selectors(arguments.render_video)
            self.render_video(scenes, **shared)
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

    def validate(
        self,
        scenes: _SceneSelection = None,
        *,
        video_settings: VideoSettings | None = None,
        script: str | Path | None = None,
    ) -> ProjectValidation:
        """Author scenes and resolve narration timing without rendering frames.

        Scene save-frame and save-video calls are skipped. Speech still obtains
        audio from the configured source, which can populate caches, and project
        transcripts are updated. This checks authoring and timing; it does not
        run frame updaters, compile shaders, or check visual output.

        Animation
        ---------
        Authors the selected scene functions immediately in isolated Scenes;
        it records their animations without rendering frames.

        Parameters
        ----------
        scenes
            Scene ID, name, prefixed name, or iterable mixing these forms.
            Defaults to None, meaning every scene.
        video_settings
            Settings used while authoring, including screen layout. Defaults
            to None, meaning the project's settings or SETTINGS.video.
        script
            Expected narration text, or a Path to a UTF-8 text file. Compared
            against the selected scenes in project order. Whitespace is ignored;
            words, punctuation and capitalization must match exactly. Defaults
            to None, skipping the comparison. A string always means literal text.

        Returns
        -------
        ProjectValidation
            Per-scene ID, name, duration_seconds, checkpoints and unwrapped
            transcript, plus total duration_seconds. Durations include holds.

        Raises
        ------
        AlganConfigurationError
            If a scene selector or authored duration is invalid, or narration
            differs from ``script``. The message names the first differing
            word, the scene it falls in, and the words around it in both
            texts. Authoring and speech-source errors propagate with the scene
            name attached.

        Examples
        --------
        .. code-block:: python

            from algan import Project, Scene


            def intro():
                Scene.wait(2)


            report = Project([intro]).validate()
            print(report.duration_seconds)  # 2.0
        """
        from algan.scene import _note_render_requested

        # An intentional author-only pass must not trigger the daemon's
        # "script forgot to export" warning. Skipped exports count there too.
        _note_render_requested()
        if script is not None and not isinstance(script, (str, Path)):
            raise AlganConfigurationError("script must be narration text or a Path")
        expected = (
            script.read_text(encoding="utf-8-sig")
            if isinstance(script, Path)
            else script
        )
        report = ProjectValidation(
            tuple(
                self._render(
                    scenes,
                    mode="validate",
                    video_settings=video_settings,
                )
            )
        )
        if expected is not None:
            mismatch = _describe_script_mismatch(report.scenes, expected.split())
            if mismatch is not None:
                raise AlganConfigurationError(mismatch)
        return report

    def profile(
        self,
        scenes: _SceneSelection = None,
        *,
        video_settings: VideoSettings | None = None,
        runs: int = 2,
        **profile_kwargs,
    ) -> dict[str, list[dict]]:
        """Render selected scenes through the stage-by-stage scene profiler.

        Each pass re-authors its scene with this project's speech source. Scene
        checkpoints and manual exports are skipped. Profile videos and reports
        go under the project's video directory in ``profiling`` by default.

        Parameters
        ----------
        scenes
            Scene ID, name, prefixed name, or iterable mixing these forms.
            Defaults to None, meaning every scene; select short representative
            scenes to limit profiling cost.
        video_settings
            Render and authoring settings. Defaults to None, meaning this
            project's settings or SETTINGS.video.
        runs
            Number of complete render passes per scene. Defaults to 2, to
            compare first-pass and warm timings. Must be a positive integer.
        **profile_kwargs
            Passed to ``algan.utils.profiling_utils.profile_scene``, including
            ``kernel_profiler``, ``telemetry``, ``output_directory`` and
            ``save_video_kwargs``. Kernel GPU profiling defaults to False here
            to preserve the production runtime; stage wall timers stay enabled.
            ``tag`` defaults to the stable scene stem.

        Returns
        -------
        dict
            Scene stems mapped to the profiler's per-pass result dictionaries.
            These include render wall time (``total``), authoring_seconds,
            scene_seconds, output_path and inclusive/exclusive stage times.

        Examples
        --------
        .. code-block:: python

            from algan import Project, Scene


            def intro():
                Scene.wait(1)


            profiles = Project([intro]).profile("intro", telemetry=False)
        """
        from algan.scene import Scene, SceneManager
        from algan.utils.profiling_utils import (
            _temporary_instrumentation,
            profile_scene,
        )

        selected = self._selected_scenes(scenes)
        effective = video_settings or self.video_settings or SETTINGS.video
        profiles = {}
        for project_scene in selected:

            def author(project_scene=project_scene):
                scene = Scene.current()
                run = _ProjectSceneRun(self, project_scene, "profile")
                scene._project_run = run
                scene._suppress_automatic_transcript = True
                scene.audio_manager.set_speech_source(self.speech_source)
                token = _ACTIVE_PROJECT_RUN.set(run)
                try:
                    project_scene.function()
                    self._sync_scene_transcript(
                        project_scene, scene.audio_manager.video_transcript
                    )
                finally:
                    _ACTIVE_PROJECT_RUN.reset(token)
                run.allow_video_render = True

            options = dict(profile_kwargs)
            if self.post_processes is not None:
                export_options = dict(options.get("save_video_kwargs") or {})
                export_options.setdefault("post_processes", self.post_processes)
                options["save_video_kwargs"] = export_options
            options.setdefault("kernel_profiler", False)
            options.setdefault("output_directory", self.video_directory / "profiling")
            options.setdefault("tag", project_scene.stem)
            try:
                with _temporary_instrumentation():
                    profiles[project_scene.stem] = profile_scene(
                        author,
                        effective,
                        runs=runs,
                        **options,
                    )
            finally:
                SceneManager.reset()
        return profiles

    def estimate_render_time(
        self,
        reference_scenes: _SceneSelection,
        *,
        video_settings: VideoSettings | None = None,
        runs: int = 2,
        **profile_kwargs,
    ) -> RenderEstimate:
        """Estimate full export time by profiling chosen representative scenes.

        Author all scenes to measure duration, then render only the selected
        reference scenes through :meth:`profile`. This is an explicit sampling
        render, so choose short scenes representative of the project's effects.
        Estimates assume the same quality, hardware, renderer and encoding.
        The sampled range cannot bound costs of effects absent from the samples.

        Parameters
        ----------
        reference_scenes
            Scene ID, name, prefixed name, or iterable mixing these forms.
            Required; None explicitly selects every scene as a reference.
        video_settings
            Settings used for validation and profiling. Defaults to None,
            meaning the project's settings or SETTINGS.video.
        runs
            Complete passes per reference scene. Defaults to 2. At least two
            are required to separate first-pass from warm timings.
        **profile_kwargs
            Passed to :meth:`profile`, including ``save_video_kwargs`` for
            matching the intended encoder, and ``telemetry``.

        Returns
        -------
        RenderEstimate
            Project duration, warm_seconds, cold_overhead_seconds,
            estimated_seconds, range_seconds and underlying profiles.
            Times exclude authoring, validation, sampling and concatenation.

        Raises
        ------
        AlganConfigurationError
            If runs is less than two or a reference has zero authored duration.

        Examples
        --------
        .. code-block:: python

            from algan import Project, Scene


            def intro():
                Scene.wait(1)


            estimate = Project([intro]).estimate_render_time("intro")
            print(estimate.range_seconds)
        """
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 2:
            raise AlganConfigurationError(
                "Estimation needs at least two profiling runs"
            )
        selected = self._selected_scenes(reference_scenes)
        if not selected:
            raise AlganConfigurationError("Select at least one reference scene")
        validation = self.validate(video_settings=video_settings)
        by_id = {scene.id: scene for scene in validation.scenes}
        if any(by_id[scene.id].duration_seconds <= 0 for scene in selected):
            raise AlganConfigurationError(
                "Reference scenes must have positive duration"
            )
        profiles = self.profile(
            [s.id for s in selected],
            video_settings=video_settings,
            runs=runs,
            **profile_kwargs,
        )
        rates, durations, warm_times = [], [], []
        overhead = 0.0
        for passes in profiles.values():
            duration = passes[-1]["scene_seconds"]
            warm = sum(p["total"] for p in passes[1:]) / (len(passes) - 1)
            if duration <= 0 or any(
                not math.isclose(p["scene_seconds"], duration) for p in passes
            ):
                raise AlganConfigurationError(
                    "Reference duration changed between profiling passes"
                )
            rates.append(warm / duration)
            durations.append(duration)
            warm_times.append(warm)
            overhead += max(0.0, passes[0]["total"] - warm)
        total = validation.duration_seconds
        return RenderEstimate(
            total,
            total * sum(warm_times) / sum(durations),
            overhead,
            (total * min(rates) + overhead, total * max(rates) + overhead),
            profiles,
        )

    def save_subtitles(
        self,
        file_path: str | Path | None = None,
        scenes: _SceneSelection = None,
        *,
        subtitle_format: str | None = None,
        video_settings: VideoSettings | None = None,
        animate_fade_out: bool | None = None,
        include_speech: bool = True,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        max_duration: float = 6.0,
        overwrite: bool = True,
    ) -> Path:
        """Export one subtitle file for selected scenes joined in project order.

        Scene times are offset by the preceding selected scenes' video durations,
        including silent scenes, speech holds and requested final fade-outs,
        rounded to video frames. Use the same settings as the video export.
        See :meth:`Scene.save_subtitles <algan.scene.Scene.save_subtitles>` for
        speech alignment and manual-caption semantics.

        Animation
        ---------
        Authors each selected scene in isolation, obtaining its speech clips,
        then writes subtitles without rendering frames. Save-frame, save-video
        and save-subtitles calls inside scene functions are suppressed. Existing
        Scenes are unchanged. No spawned mobs are required.

        Parameters
        ----------
        file_path
            Destination using Algan's usual output path rules. Defaults to None,
            meaning the project's video path with the subtitle extension.
        scenes
            Scene ID, name or iterable mixing these forms. Defaults to None,
            meaning all scenes. Selection always retains project order.
        subtitle_format
            ``"srt"`` or ``"vtt"``. Defaults to None, inferring the filename's
            extension or using SRT if none was supplied.
        video_settings
            Authoring settings and frame rate used for scene offsets. Defaults
            to None, meaning the project's settings or SETTINGS.video.
        animate_fade_out
            Include the final fade used by :meth:`render_video`. Defaults to None,
            meaning ``SETTINGS.style.fade_out_on_scene_end``. Pass the same value
            as the video export to keep subsequent scenes' captions aligned.
        include_speech
            Include Speech narration as well as manual captions. Defaults to True.
        max_chars_per_line
            Positive speech line length in characters. Defaults to 42; single
            long words remain intact. Manual captions are not rewrapped.
        max_lines
            Positive maximum lines in each speech cue. Defaults to 2.
        max_duration
            Positive maximum speech cue duration in seconds. Defaults to 6;
            a single word may exceed it.
        overwrite
            Defaults to True: replace an existing file. False returns its path
            without re-authoring any scenes or changing the file.

        Returns
        -------
        pathlib.Path
            Absolute path to the combined subtitle file.

        Raises
        ------
        AlganConfigurationError
            If selection, timing, format or grouping options are invalid.

        Examples
        --------
        .. code-block:: python

            from algan import Project, Scene


            def intro():
                Scene.add_subcaption("Welcome!", duration=2)
                Scene.wait(2)


            Project([intro], file_path="lesson.mp4").save_subtitles()
        """
        from algan.scene import _note_render_requested
        from algan.sound.subtitles import (
            _subtitle_destination,
            _SubtitleOptions,
            _write_subtitles,
        )

        options = _SubtitleOptions(
            include_speech, max_chars_per_line, max_lines, max_duration
        )
        if animate_fade_out is not None and not isinstance(animate_fade_out, bool):
            raise AlganConfigurationError("animate_fade_out must be a boolean or None")
        destination, subtitle_format = _subtitle_destination(
            file_path, subtitle_format, default=self.file_path
        )
        # Resolve a generator selection only once; validate even when skipping
        # an existing output below.
        selected = tuple(scene.id for scene in self._selected_scenes(scenes))
        _note_render_requested()
        if not overwrite and destination.exists():
            return destination
        results = self._render(
            selected,
            mode="subtitles",
            video_settings=video_settings,
            subtitle_options=options,
            animate_fade_out=animate_fade_out,
        )
        cues = []
        offset = 0.0
        for duration, scene_cues in results:
            cues.extend(
                (start + offset, end + offset, text) for start, end, text in scene_cues
            )
            offset += duration
        return _write_subtitles(destination, cues, subtitle_format, overwrite)

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

    @staticmethod
    def _normalized_frame_patterns(frames) -> tuple[str, ...]:
        """Flatten a frame selection into de-duplicated pattern strings."""
        if frames is None:
            return ()
        if isinstance(frames, (str, int)) and not isinstance(frames, bool):
            frames = (frames,)
        try:
            raw = tuple(frames)
        except TypeError as exc:
            raise AlganConfigurationError(
                f"Invalid Project frame selection: {frames!r}"
            ) from exc
        patterns = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise AlganConfigurationError(
                    f"Invalid Project frame selector: {item!r}"
                )
            for part in str(item).split(","):
                part = part.strip()
                if part and part not in patterns:
                    patterns.append(part)
        if not patterns:
            raise AlganConfigurationError("Project frame selection cannot be empty")
        return tuple(patterns)

    def _render(
        self,
        scenes=None,
        *,
        mode: Literal["screenshots", "video", "validate", "view", "subtitles"],
        video_settings: VideoSettings | None = None,
        overwrite: bool = True,
        frames=None,
        stop_early: bool = False,
        subtitle_options=None,
        **save_video_kwargs,
    ):
        """Author isolated scenes for exports, validation or interactive viewing."""
        selected = self._selected_scenes(scenes)
        frame_patterns = self._normalized_frame_patterns(frames)
        if mode != "screenshots" and (frame_patterns or stop_early):
            raise AlganConfigurationError(
                "frames and stop_early only apply to screenshot renders"
            )
        if stop_early and not frame_patterns:
            raise AlganConfigurationError(
                "stop_early needs a frames selection to know what to stop after"
            )
        if mode == "screenshots":
            unknown = set(save_video_kwargs) - {"background", "post_processes"}
            if unknown:
                raise AlganConfigurationError(
                    f"Unsupported screenshot options: {', '.join(sorted(unknown))}"
                )
        if mode in ("screenshots", "video") and self.post_processes is not None:
            save_video_kwargs.setdefault("post_processes", self.post_processes)
        effective_settings = video_settings or self.video_settings or SETTINGS.video
        results = []
        matched_anywhere = set()

        from algan.scene import Scene

        for project_scene in selected:
            with Scene(video_settings=effective_settings) as active_scene:
                run = _ProjectSceneRun(
                    self,
                    project_scene,
                    mode,
                    frame_patterns=frame_patterns,
                    stop_early=stop_early,
                    frame_options={"overwrite": overwrite, **save_video_kwargs},
                )
                active_scene._project_run = run
                active_scene._suppress_automatic_transcript = True
                active_scene.audio_manager.set_speech_source(self.speech_source)
                run_token = _ACTIVE_PROJECT_RUN.set(run)
                try:
                    try:
                        with suppress(_StopSceneEarly):
                            project_scene.function()
                    except Exception as exc:
                        message = f"While authoring project scene {project_scene.stem}"
                        if hasattr(exc, "add_note"):
                            exc.add_note(message)
                        else:  # Python 3.10 has no exception notes.
                            logger.error(message)
                        raise
                    matched_anywhere |= run.matched_patterns
                    if run.stopped_early:
                        # The scene never finished, so its transcript would be a
                        # truncated copy of the real one. Leave the file alone.
                        self._warn_partial_scene(project_scene, run)
                    elif mode != "view":
                        self._sync_scene_transcript(
                            project_scene,
                            active_scene.audio_manager.video_transcript,
                        )
                    if mode == "view":
                        duration = float(active_scene._recorded_end_time_for_render())
                        if not math.isfinite(duration) or duration < 0:
                            raise AlganConfigurationError(
                                f"Invalid duration for {project_scene.stem}: {duration}"
                            )
                        results.append(
                            (
                                project_scene.id,
                                project_scene.stem,
                                active_scene,
                                SETTINGS.raytracing.to_dict(),
                            )
                        )
                    elif mode == "validate":
                        duration = float(active_scene._recorded_end_time_for_render())
                        if not math.isfinite(duration) or duration < 0:
                            raise AlganConfigurationError(
                                f"Invalid duration for {project_scene.stem}: {duration}"
                            )
                        results.append(
                            SceneValidation(
                                project_scene.id,
                                project_scene.stem,
                                duration,
                                run.next_frame_index,
                                active_scene.audio_manager.video_transcript,
                            )
                        )
                    elif mode == "subtitles":
                        from algan.sound.subtitles import _scene_cues, _seconds

                        # Apply the same authored finalization as a video export,
                        # without rendering or mutating any caller-owned Scene.
                        fps = effective_settings.frames_per_second
                        if active_scene._recorded_end_time_for_render() == 0 and any(
                            actor.is_spawned() for actor in active_scene.actors
                        ):
                            active_scene.wait(1 / fps)
                        fade_out = save_video_kwargs.get("animate_fade_out")
                        if fade_out is None:
                            fade_out = SETTINGS.style.fade_out_on_scene_end
                        if fade_out:
                            active_scene.despawn_mobs(retain_history=True, runtime=0.5)
                        duration = _seconds(
                            active_scene._recorded_end_time_for_render(),
                            "Scene duration",
                        )
                        if duration < 0:
                            raise AlganConfigurationError(
                                "Scene duration must be non-negative"
                            )
                        duration = round(duration * fps) / fps
                        results.append(
                            (
                                duration,
                                _scene_cues(
                                    active_scene, subtitle_options, duration=duration
                                ),
                            )
                        )
                    elif mode == "video":
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
                        run.render_frames()
                        results.extend(run.frame_results)
                finally:
                    run.frame_batches.clear()
                    _ACTIVE_PROJECT_RUN.reset(run_token)
                    active_scene._project_run = None
                    active_scene._suppress_automatic_transcript = False
            verb = (
                "Authored"
                if mode in ("view", "subtitles")
                else "Validated"
                if mode == "validate"
                else "Stopped early in"
                if run.stopped_early
                else "Finished rendering"
            )
            logger.info(f"{verb} project scene {project_scene.stem} in {mode} mode")

        unmatched = tuple(
            pattern for pattern in frame_patterns if pattern not in matched_anywhere
        )
        if unmatched:
            logger.warning(
                f"No frame matched {', '.join(repr(_) for _ in unmatched)} in any "
                f"selected scene, so nothing was rendered for "
                f"{'them' if len(unmatched) > 1 else 'it'}. Frame patterns are "
                f"matched against the save-frame name, not the file on disk."
            )
        return results

    @staticmethod
    def _warn_partial_scene(
        project_scene: _ProjectScene, run: _ProjectSceneRun
    ) -> None:
        """Say plainly that a stop_early scene is not a render of that scene."""
        logger.warning(
            f"PARTIAL RENDER: scene {project_scene.stem} was abandoned after frame "
            f"{run.last_frame_name} because every frame pattern had matched. The "
            f"rest of the scene was never authored, so its transcript has been "
            f"left untouched and every later frame of it on disk is whatever an "
            f"earlier run wrote. This is a preview, not a render of the scene -- "
            f"re-run it without stop_early before trusting the output."
        )

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
