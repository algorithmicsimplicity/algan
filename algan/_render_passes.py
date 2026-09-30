"""Auxiliary render passes -- depth, normal, object ID -- written beside a render.

``Scene.save_video(passes=...)`` and ``Scene.save_frame(passes=...)`` ask the
render loop for per-frame pass data (``get_frames(aux_passes=True,
aux_sink=...)``, computed by :mod:`algan.rendering.raytracing.aux_passes`: one
pinhole ray through each pixel centre, first surface with alpha >= 0.5). This
module turns that data into files a compositor or video editor imports:

* ``depth``: 32-bit float OpenEXR, one channel (``Y``): the distance from the
  camera along its forward axis to the surface, in world units. Pixels that hit
  nothing hold :data:`DEPTH_BACKGROUND`.
* ``normal``: 16-bit RGB PNG: the surface's unit shading normal in camera space
  (x right, y up, z toward the camera), encoded ``rgb = n * 0.5 + 0.5``. Pixels
  that hit nothing are black, which no unit normal encodes to.
* ``object_id``: 8-bit RGB PNG: every object a flat colour, a bijection of its
  id (:func:`algan.rendering.pass_identity.pass_id_color`), black for nothing.

A video writes each pass as a numbered image sequence in its own directory
(``intro.depth/intro.depth.00000.exr`` ...), which every editor imports and
no container can garble; a still writes one file per pass (``shot.depth.exr``).
Each render also writes ``<output file name>.passes.json`` (``intro.mp4.passes.json``),
describing the encodings and
listing every object id with the Mob it names.

Encoding runs through FFmpeg (the binary Algan already uses for video), one
process per pass, each behind its own bounded queue and writer thread.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from algan.errors import AlganConfigurationError
from algan.logging.logger import get_logger

logger = get_logger()

#: The passes that can be requested, in the order they are written.
PASS_NAMES = ("depth", "normal", "object_id")

#: Depth written where a pixel hits nothing: "infinitely far", the value
#: Blender writes to its Z pass, so depth-driven fog and defocus treat the
#: background as distant. A float, not ``inf``, which several tools reject.
DEPTH_BACKGROUND = 1e10

#: Object id written for a surface whose source Mob could not be named.
#: Not expected in practice; listed in the sidecar if it ever appears.
UNIDENTIFIED_PASS_ID = (1 << 24) - 1

#: Surfaces less opaque than this are looked through by every pass.
ALPHA_THRESHOLD = 0.5


@dataclass(frozen=True)
class _PassFormat:
    extension: str
    input_format: str
    output_format: str
    codec: str
    codec_args: tuple
    description: str


_FORMATS = {
    "depth": _PassFormat(
        "exr",
        "grayf32le",
        "grayf32le",
        "exr",
        ("-compression", "zip1"),
        "OpenEXR, 32-bit float, one channel (Y)",
    ),
    "normal": _PassFormat(
        "png",
        "rgb48le",
        "rgb48be",
        "png",
        ("-pred", "up"),
        "PNG, 16-bit RGB",
    ),
    "object_id": _PassFormat(
        "png",
        "rgb24",
        "rgb24",
        "png",
        (),
        "PNG, 8-bit RGB",
    ),
}


def normalize_passes(passes) -> tuple[str, ...]:
    """Validate a ``passes`` argument into canonical order, without repeats.

    Accepts one name or an iterable of names; ``None`` and an empty iterable
    mean no passes.

    Raises
    ------
    AlganConfigurationError
        If a name is not one of :data:`PASS_NAMES`.
    """
    if passes is None:
        return ()
    if isinstance(passes, str):
        passes = (passes,)
    try:
        requested = list(passes)
    except TypeError as exc:
        raise AlganConfigurationError(
            f"passes must be a pass name or a sequence of them, one or more of "
            f"{', '.join(repr(n) for n in PASS_NAMES)}; got {passes!r}"
        ) from exc
    unknown = [name for name in requested if name not in PASS_NAMES]
    if unknown:
        raise AlganConfigurationError(
            f"Unknown render pass {unknown[0]!r}. Passes are "
            f"{', '.join(repr(n) for n in PASS_NAMES)}."
        )
    return tuple(name for name in PASS_NAMES if name in requested)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


class _ArrayFrame:
    """A host array standing in for a frame tensor in the writer queue.

    ``render_loop.write_frames_from_queue`` asks each queued frame for
    ``.numpy()``; 16-bit and float pass frames are plain numpy arrays.
    """

    __slots__ = ("array",)

    def __init__(self, array):
        self.array = np.ascontiguousarray(array)

    def numpy(self):
        return self.array


def encode_depth(depth: torch.Tensor) -> np.ndarray:
    """``[F, H, W]`` world-unit depths (``inf`` = no hit) -> little-endian f32."""
    d = depth.detach().cpu().to(torch.float32).numpy()
    return np.where(np.isfinite(d), d, np.float32(DEPTH_BACKGROUND)).astype("<f4")


def encode_normal(normal: torch.Tensor, hit: torch.Tensor) -> np.ndarray:
    """``[F, H, W, 3]`` camera-space unit normals -> little-endian u16 RGB.

    ``hit`` is ``[F, H, W]``; pixels without a hit are written black.
    """
    n = normal.detach().cpu().to(torch.float32).numpy()
    encoded = np.clip(np.rint((n * 0.5 + 0.5) * 65535.0), 0.0, 65535.0)
    encoded = encoded.astype("<u2")
    encoded[~hit.detach().cpu().numpy()] = 0
    return encoded


class ObjectIdResolver:
    """Maps the renderer's source Mob ids to object ids, and remembers them.

    ``registry`` is the render-scoped ``{Mob.id: Mob}`` table the primitive
    build fills (``scene._aux_id_registry``). Each id is resolved once, through
    :func:`algan.rendering.pass_identity.resolve_pass_owner`; ``objects``
    accumulates the sidecar's description of every object id actually written.
    """

    def __init__(self, registry):
        self.registry = registry
        self._pass_id_of = {}
        self.objects = {}

    def _resolve(self, mob_id):
        known = self._pass_id_of.get(mob_id)
        if known is not None:
            return known
        from algan.rendering.pass_identity import describe_owner, resolve_pass_owner

        mob = self.registry.get(mob_id) if mob_id >= 0 else None
        if mob is None:
            pass_id = UNIDENTIFIED_PASS_ID
            entry = self.objects.setdefault(
                pass_id, {"id": pass_id, "label": "unidentified geometry", "mobs": []}
            )
        else:
            pass_id, owner = resolve_pass_owner(mob)
            entry = self.objects.setdefault(pass_id, {"id": pass_id, "mobs": []})
            description = describe_owner(owner)
            if description not in entry["mobs"]:
                entry["mobs"].append(description)
            explicit = getattr(owner, "pass_index", None)
            if explicit is not None:
                entry["pass_index"] = int(explicit)
        self._pass_id_of[mob_id] = pass_id
        return pass_id

    def pass_ids(self, mob_ids: torch.Tensor) -> torch.Tensor:
        """``[F, H, W]`` source Mob ids (``-2`` miss, ``-1`` unknown) -> ids."""
        mob_ids = mob_ids.detach().cpu().to(torch.int64)
        values, inverse = torch.unique(mob_ids, return_inverse=True)
        mapped = torch.tensor(
            [0 if v == -2 else self._resolve(int(v)) for v in values.tolist()],
            dtype=torch.int64,
        )
        return mapped[inverse]


def encode_object_id(pass_ids: torch.Tensor) -> np.ndarray:
    """``[F, H, W]`` object ids -> ``[F, H, W, 3]`` u8 colours (0 -> black)."""
    from algan.rendering.pass_identity import pass_id_colors

    return pass_id_colors(pass_ids).cpu().numpy().astype(np.uint8)


def encode_passes(names, aux, resolver=None) -> dict[str, np.ndarray]:
    """Encode one ``aux_sink`` batch into per-pass ``[F, H, W(, C)]`` arrays."""
    hit = torch.isfinite(aux["depth"])
    out = {}
    for name in names:
        if name == "depth":
            out[name] = encode_depth(aux["depth"])
        elif name == "normal":
            out[name] = encode_normal(aux["normal"], hit)
        elif name == "object_id":
            out[name] = encode_object_id(resolver.pass_ids(aux["mob_id"]))
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _ffmpeg_binary():
    from algan.utils.video_encoding import _moviepy_ffmpeg_binary, resolve_encode_binary

    binary = resolve_encode_binary(None)
    if binary:
        return binary
    try:
        return _moviepy_ffmpeg_binary() or "ffmpeg"
    except Exception:  # noqa: BLE001 -- an unimportable moviepy is not ours
        return "ffmpeg"


_ENCODERS_OF = {}


def check_pass_encoders(names) -> str:
    """The FFmpeg binary to encode ``names`` with; fail early if it cannot.

    The binary's encoder list is asked for once per process (a subprocess
    each time would tax every still of a contact sheet).
    """
    from algan.utils.video_encoding import _listed_encoders

    binary = _ffmpeg_binary()
    needed = sorted({_FORMATS[name].codec for name in names})
    available = _ENCODERS_OF.get(binary)
    if available is None:
        available = _listed_encoders(binary)
        if available is not None:
            # Only a real answer is remembered: an FFmpeg that could not be
            # asked this time may be installed before the next render.
            _ENCODERS_OF[binary] = available
        elif shutil.which(binary) is None and not Path(binary).is_file():
            raise AlganConfigurationError(
                f"Cannot run FFmpeg ({binary!r}), which writes the requested "
                "render passes. Install FFmpeg, or point "
                "SETTINGS.paths.ffmpeg_binary at an FFmpeg executable."
            )
    if available is not None:
        missing = [codec for codec in needed if codec not in available]
        if missing:
            raise AlganConfigurationError(
                f"{binary} cannot encode {', '.join(missing)}, which the "
                f"requested render passes are written with. Install an FFmpeg "
                f"build that lists it under `{binary} -encoders` (any FFmpeg 5 "
                f"or newer does) and point SETTINGS.paths.ffmpeg_binary at it."
            )
    return binary


class PassEncoder:
    """One FFmpeg process encoding one pass: an image sequence, or one still.

    Shaped like the video writer moviepy provides -- ``proc.stdin``,
    ``write_frame`` and ``close`` -- so ``render_loop._VideoWriter`` and its
    zero-copy pipe write drive it unchanged.
    """

    def __init__(
        self, binary, name, width, height, fps, target, *, still, display=None
    ):
        fmt = _FORMATS[name]
        self.name = name
        # ``target`` is what FFmpeg is given (a sequence pattern has its
        # literal ``%`` doubled); ``display`` is the path messages name.
        self.target = Path(display if display is not None else target)
        command = [
            binary, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", fmt.input_format,
            "-s", f"{int(width)}x{int(height)}",
            # moviepy's own formatting, so pass frames time like the video's.
            "-r", f"{float(fps):.2f}",
            "-i", "-", "-an",
            "-c:v", fmt.codec, *fmt.codec_args, "-pix_fmt", fmt.output_format,
        ]  # fmt: skip
        if still:
            command += ["-frames:v", "1", "-update", "1"]
        else:
            command += ["-f", "image2", "-start_number", "0"]
        command.append(str(target))
        # stderr to a file, not a pipe: several encoders run at once, and a
        # pipe nobody drains can deadlock one of them.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 -- closed in close/abort
        try:
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except BaseException:
            self._stderr.close()
            raise

    def write_frame(self, array):
        try:
            self.proc.stdin.write(memoryview(np.ascontiguousarray(array)))
        except OSError as exc:
            # FFmpeg exited (a full disk, an unwritable directory): report
            # what it said rather than a bare broken pipe.
            raise self._failure(self._finish(timeout=10)) from exc

    def _finish(self, timeout=None):
        proc, self.proc = self.proc, None
        if proc is None:
            return 0
        with contextlib.suppress(OSError):
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()

    def _failure(self, code):
        self._stderr.seek(0)
        message = self._stderr.read().decode(errors="replace").strip()
        return RuntimeError(
            f"FFmpeg failed writing the {self.name} pass to {self.target} "
            f"(exit code {code}): {message or 'no message'}"
        )

    def close(self):
        if self.proc is None:
            return
        code = self._finish()
        try:
            if code != 0:
                raise self._failure(code)
        finally:
            self._stderr.close()

    def abort(self):
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        self._stderr.close()


def _stem(path: Path) -> Path:
    return Path(path).with_suffix("")


def still_pass_paths(image_path: Path, names) -> dict[str, Path]:
    """``shot.png`` -> ``{"depth": shot.depth.exr, ...}``."""
    stem = _stem(image_path)
    return {
        name: stem.with_name(f"{stem.name}.{name}.{_FORMATS[name].extension}")
        for name in names
    }


def video_pass_dirs(video_path: Path, names) -> dict[str, Path]:
    """``intro.mp4`` -> ``{"depth": intro.depth/, ...}``."""
    stem = _stem(video_path)
    return {name: stem.with_name(f"{stem.name}.{name}") for name in names}


def sidecar_path(output_path: Path) -> Path:
    """``intro.mp4`` -> ``intro.mp4.passes.json``.

    Named after the whole file name, extension included: a still and a video
    of the same stem (``save_frame()`` and ``save_video()`` with the default
    name, say) each keep their own description.
    """
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.name}.passes.json")


def _sequence_pattern(directory: Path, name: str) -> str:
    """The FFmpeg image2 target for ``directory``'s frames.

    image2 reads every ``%`` in the whole path as a printf directive, so the
    literal parts have theirs doubled; only the frame counter is one.
    """
    stem = directory.name.removesuffix("_temp")
    literal = str(directory / stem).replace("%", "%%")
    return f"{literal}.%05d.{_FORMATS[name].extension}"


def _sequence_file_regex(directory_stem: str, name: str):
    return re.compile(
        rf"^{re.escape(directory_stem)}\.\d{{5,}}\.{_FORMATS[name].extension}$"
    )


def _sidecar(
    names, paths, resolver, width, height, fps, *, output, extra, present=None
):
    base = Path(output).parent
    passes = {}
    for name in names:
        fmt = _FORMATS[name]
        entry = {
            "path": os.path.relpath(paths[name], base),
            "format": fmt.description,
        }
        if name == "depth":
            entry.update(
                unit="world units",
                definition=(
                    "distance from the camera along its forward axis (planar "
                    "depth, not ray length)"
                ),
                background=DEPTH_BACKGROUND,
            )
        elif name == "normal":
            entry.update(
                space="camera: x = screen right, y = screen up, z = toward the camera",
                encoding="rgb = normal * 0.5 + 0.5, over 0..65535",
                background=[0, 0, 0],
            )
        elif name == "object_id":
            from algan.rendering.pass_identity import pass_id_color

            objects = []
            for pass_id in sorted(resolver.objects):
                if present is not None and pass_id not in present:
                    continue
                info = dict(resolver.objects[pass_id])
                rgb = pass_id_color(pass_id)
                info["rgb"] = list(rgb)
                info["color"] = "#{:02x}{:02x}{:02x}".format(*rgb)
                objects.append(info)
            entry.update(
                encoding=(
                    "each object one flat colour; rgb -> id is a bijection (see "
                    "algan.rendering.pass_identity.pass_id_from_color)"
                ),
                background=[0, 0, 0],
                objects=objects,
            )
        passes[name] = entry
    return {
        "generator": "algan",
        "version": 1,
        "output": os.path.relpath(output, base),
        "resolution": [int(width), int(height)],
        "frames_per_second": float(fps),
        **extra,
        "sampling": (
            "one pinhole ray through each pixel centre, first surface at least "
            f"{ALPHA_THRESHOLD} opaque; not anti-aliased, not defocused"
        ),
        "passes": passes,
    }


def _write_json(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class VideoPassJob:
    """The passes of one ``save_video`` call, from the first frame to publish.

    Frames are encoded into ``<stem>.<pass>_temp/`` beside the destination and
    moved into ``<stem>.<pass>/`` only after the video itself is published, so
    a failed render leaves the previous passes (and the video) alone.
    """

    names: tuple
    video_path: Path
    binary: str
    directories: dict = field(default_factory=dict)
    temporaries: dict = field(default_factory=dict)
    writers: dict = field(default_factory=dict)
    resolver: ObjectIdResolver | None = None
    pending: deque = field(default_factory=deque)
    frames_written: int = 0
    keep_temporaries: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0

    @classmethod
    def create(cls, names, video_path):
        names = normalize_passes(names)
        if not names:
            return None
        job = cls(names, Path(video_path), check_pass_encoders(names))
        job.directories = video_pass_dirs(job.video_path, names)
        for name, directory in job.directories.items():
            # Before rendering, not at publish: a file where a pass directory
            # must go would otherwise fail after the video had been replaced.
            if directory.exists() and not directory.is_dir():
                raise AlganConfigurationError(
                    f"Cannot write the {name} pass to {directory}: a file of "
                    "that name is in the way. Move it, or render to another "
                    "name."
                )
        job.temporaries = {
            name: path.with_name(path.name + "_temp")
            for name, path in job.directories.items()
        }
        return job

    @property
    def wants_identity(self):
        return "object_id" in self.names

    def open(self, scene, video_writer_cls):
        """Start one encoder (and writer thread) per pass for ``scene``'s frames."""
        self.width = int(scene.num_pixels_screen_width)
        self.height = int(scene.num_pixels_screen_height)
        self.fps = float(scene.frames_per_second)
        if self.wants_identity:
            scene._aux_id_registry = {}
            self.resolver = ObjectIdResolver(scene._aux_id_registry)
        for name in self.names:
            temporary = self.temporaries[name]
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            encoder = PassEncoder(
                self.binary,
                name,
                self.width,
                self.height,
                self.fps,
                _sequence_pattern(temporary, name),
                still=False,
                display=temporary,
            )
            writer = video_writer_cls(encoder)
            writer.start()
            self.writers[name] = writer

    def sink(self, aux):
        self.pending.append(aux)

    def take(self, frame_count):
        """Encode the aux batch delivered with the next ``frame_count`` frames."""
        if not self.pending:
            raise RuntimeError("the renderer yielded frames without their passes")
        aux = self.pending.popleft()
        if int(aux["depth"].shape[0]) != int(frame_count):
            raise RuntimeError(
                f"the renderer delivered passes for {int(aux['depth'].shape[0])} "
                f"frames alongside {int(frame_count)}"
            )
        return encode_passes(self.names, aux, self.resolver)

    def put(self, encoded, index, copies=1):
        for name in self.names:
            self.writers[name].put(_ArrayFrame(encoded[name][index]), copies)
        self.frames_written += int(copies)

    def finish(self):
        """Flush every encoder; raises the first failure after stopping the rest."""
        error = None
        for writer in self.writers.values():
            try:
                writer.finish()
                writer.file_writer.close()
            except BaseException as exc:  # noqa: BLE001 -- re-raised below
                if error is None:
                    error = exc
                self._abort_writer(writer)
        if error is not None:
            raise error

    @staticmethod
    def _abort_writer(writer):
        try:
            writer.abort()
        except Exception:  # noqa: BLE001 -- cleanup must not mask the cause
            logger.debug("pass writer abort failed", exc_info=True)
        try:
            writer.file_writer.abort()
        except Exception:  # noqa: BLE001
            logger.debug("pass encoder abort failed", exc_info=True)

    def abort(self):
        for writer in self.writers.values():
            self._abort_writer(writer)
        self.discard()

    def discard(self):
        if self.keep_temporaries:
            return
        for temporary in self.temporaries.values():
            shutil.rmtree(temporary, ignore_errors=True)

    def release(self, scene):
        if self.wants_identity and hasattr(scene, "_aux_id_registry"):
            del scene._aux_id_registry

    def publish(self):
        """Move the finished sequences into place and write the sidecar.

        Runs after the video is published. If a move fails (a file locked by an
        editor, say), the sequences not yet in place are kept in their
        ``_temp`` directories -- they are fully encoded -- and the error names
        them rather than deleting them.
        """
        published = []
        try:
            for name in self.names:
                temporary = self.temporaries[name]
                directory = self.directories[name]
                directory.mkdir(parents=True, exist_ok=True)
                stale = _sequence_file_regex(directory.name, name)
                # Replace only what an earlier render of these passes wrote: a
                # longer earlier video must not leave frames past this one's end.
                for existing in directory.iterdir():
                    if existing.is_file() and stale.match(existing.name):
                        existing.unlink()
                for produced in sorted(temporary.iterdir()):
                    os.replace(produced, directory / produced.name)
                shutil.rmtree(temporary, ignore_errors=True)
                published.append(name)
                logger.info("Wrote the %s pass to %s", name, directory)
        except OSError as exc:
            self.keep_temporaries = True
            left = [str(self.temporaries[n]) for n in self.names if n not in published]
            raise RuntimeError(
                f"The video was written, but moving its render passes into "
                f"place failed ({exc}). The encoded frames not yet in place "
                f"are kept in: {', '.join(left)}."
            ) from exc
        _write_json(
            sidecar_path(self.video_path),
            _sidecar(
                self.names,
                self.directories,
                self.resolver,
                self.width,
                self.height,
                self.fps,
                output=self.video_path,
                extra={"frame_count": int(self.frames_written)},
            ),
        )
        return dict(self.directories)


def write_still_passes(scene, names, image_path, aux, index, resolver, time_stamp):
    """Write frame ``index`` of an ``aux_sink`` batch as one still's pass files.

    The sidecar lists only the objects visible in this still.
    """
    binary = check_pass_encoders(names)
    paths = still_pass_paths(image_path, names)
    frame = {key: value[index : index + 1] for key, value in aux.items()}
    encoded = encode_passes(names, frame, resolver)
    present = None
    if "object_id" in names:
        present = set(resolver.pass_ids(frame["mob_id"]).unique().tolist()) - {0}
    index = 0
    width = int(scene.num_pixels_screen_width)
    height = int(scene.num_pixels_screen_height)
    fps = float(scene.frames_per_second)
    for name in names:
        final = paths[name]
        temporary = final.with_name(f"{final.stem}_temp{final.suffix}")
        encoder = PassEncoder(binary, name, width, height, fps, temporary, still=True)
        try:
            encoder.write_frame(encoded[name][index])
            encoder.close()
        except BaseException:
            encoder.abort()
            temporary.unlink(missing_ok=True)
            raise
        os.replace(temporary, final)
    _write_json(
        sidecar_path(image_path),
        _sidecar(
            names,
            paths,
            resolver,
            width,
            height,
            fps,
            output=image_path,
            extra={"timestamp": float(time_stamp)},
            present=present,
        ),
    )
    return paths
