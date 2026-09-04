"""Command-line interface for Algan animation engine.

Provides commands for rendering scenes, managing the warm daemon, checking
environment health, and scaffolding new scenes.

``algan render`` runs a scene script; the script is what calls
``Scene.save_video()``, so the flags that change *what* it renders are applied
to ``SETTINGS`` in the process the script runs in, and only fill in what the
script does not say for itself. Without such a flag the script is launched as
its own process, exactly as a shell would, so the render daemon serves it warm;
``-q`` and ``-o`` run it here instead, because a warm process shared with other
runs cannot be handed one run's settings.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _version() -> str:
    """The installed Algan version, without importing the package."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("algan")
    except PackageNotFoundError:
        return "0+unknown"


def _ffmpeg_binary() -> tuple[str | None, str]:
    """The ffmpeg Algan will actually encode with, and where it came from.

    Not ``shutil.which("ffmpeg")``: a pinned ``SETTINGS.paths.ffmpeg_binary``
    outranks everything, and with nothing pinned Algan encodes through
    imageio-ffmpeg's bundled build -- which is why a render on a machine with
    an empty ``PATH`` produces a perfectly good mp4 while ``PATH`` alone says
    there is no ffmpeg at all.
    """
    from algan.settings import SETTINGS

    pinned = SETTINGS.paths.ffmpeg_binary
    if pinned:
        return str(pinned), "SETTINGS.paths.ffmpeg_binary"
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            return bundled, "bundled with imageio-ffmpeg"
    except Exception:  # noqa: BLE001 -- fall through to PATH
        pass
    found = shutil.which("ffmpeg")
    return (found, "PATH") if found else (None, "")


def _cmd_check(_args: argparse.Namespace) -> int:
    """Check system environment and dependencies."""
    print("=== Algan Environment Health Check ===")
    print(f"Algan: {_version()}")

    # 1. Python version
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"Python: {py_ver} ({sys.executable})")

    # 2. PyTorch & Acceleration
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(
                f"  [OK] CUDA acceleration: {device_name} (compute capability {torch.cuda.get_device_capability(0)})"
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  [OK] Apple Silicon MPS acceleration available")
        else:
            print("  [INFO] Running on CPU (no CUDA/MPS GPU detected).")

        # Which device Algan will actually use is a different question from
        # which ones exist -- ALGAN_RENDER_DEVICE and
        # SETTINGS.computing.render_device both sit between them -- and it is
        # the one a "why is this slow" or "why did this fail" starts from.
        from algan.rendering.mps_compat import mps_friendly
        from algan.settings import SETTINGS

        print(f"  Render device: {SETTINGS.computing.render_device}")
        if mps_friendly():
            print(
                "  [INFO] MPS-friendly mode is ON: float32 accumulators and "
                "int32 reductions, so renders are not bit-reproducible."
            )
        # Whether the pipeline's torch arithmetic runs fused. Off is a real
        # answer on Windows, and the reason is what a "why is this slower
        # than the docs say" starts from.
        from algan.utils.torch_compile import (
            torch_compile_enabled,
            torch_compile_support,
        )

        supported, reason = torch_compile_support()
        print(
            f"  torch.compile: {'ON' if torch_compile_enabled() else 'OFF'} "
            f"(SETTINGS.computing.torch_compile={SETTINGS.computing.torch_compile!r}"
            + ("" if supported else f"; unsupported here: {reason}")
            + ")"
        )
    except ImportError:
        print("  [ERROR] PyTorch is not installed.")

    # 3. The kernel compiler
    try:
        from algan.taichi_compat import BACKEND, describe_backend

        # Names the implementation, not just the version: the two report
        # unrelated version numbers, so "1.3.0" alone would read as a downgrade.
        print(f"Kernel compiler: {describe_backend()} (ALGAN_TAICHI_BACKEND={BACKEND})")
    except ImportError:
        print("  [ERROR] No kernel compiler (taichi / quadrants) is installed.")

    # The warm-start memoization is worth tens of seconds per process, and it
    # is version-gated to the compiler internals it patches -- so a compiler
    # release it does not recognise turns it off. That has already happened
    # once and went unnoticed (`taichi_patches/PLAN.md` §6.1), because a silent
    # no-op reads exactly like a slow machine. Reported separately from the
    # compiler line above so an import failure here cannot be mistaken for the
    # compiler itself being missing.
    from algan.utils.taichi_warmstart import skipped_reason

    warmstart_off = skipped_reason()
    if warmstart_off is not None:
        print(
            f"  [WARNING] Kernel warm-start memoization is off: {warmstart_off}. "
            "Every render pays the compiler's full Python frontend cost."
        )

    # The launch-plan cache is gated the same way and fails the same way: a
    # compiler release it does not recognise leaves every kernel launch paying
    # the compiler's full Python argument re-validation (~0.2-0.4 ms a launch,
    # hundreds of launches a render), with nothing to show for it but a
    # slower clock.
    from algan.utils.taichi_fast_launch import skipped_reason as fast_launch_skipped

    fast_launch_off = fast_launch_skipped()
    if fast_launch_off is not None:
        print(
            f"  [WARNING] Kernel fast-launch dispatcher is off: {fast_launch_off}. "
            "Every kernel launch pays the compiler's full Python argument re-validation."
        )
    # The source-keyed index is the other half of the warm frontend: with it
    # a kernel the process has compiled before skips the AST transform. It is
    # opt-in, so its state is reported as information rather than a warning
    # -- but reported, because "on and silently degraded to a no-op" is the
    # failure mode the line above exists for too.
    from algan.utils.taichi_source_key import skipped_reason as source_key_skipped

    source_key_off = source_key_skipped()
    if source_key_off is None:
        print("  Kernel source-keyed cache index: ON (ALGAN_TAICHI_SOURCE_KEY=1)")
    else:
        print(f"  [INFO] Kernel source-keyed cache index is off: {source_key_off}.")
    # Same shape of hazard for the early-return rewrite: version-gated to the
    # compiler it wraps, and when it is off the only symptom is a shader
    # stage that used to compile now failing with the compiler's own message.
    from algan.utils.taichi_early_return import skipped_reason as _early_return_off

    early_return_off = _early_return_off()
    if early_return_off is not None:
        print(
            f"  [WARNING] Early `return` in inlined @ti.func bodies is off: "
            f"{early_return_off}. A `return` under a runtime if/for/while in "
            "a shader stage will be rejected by the compiler."
        )

    # 4. FFmpeg
    ffmpeg_path, source = _ffmpeg_binary()
    if ffmpeg_path:
        print(f"FFmpeg: [OK] {ffmpeg_path} ({source})")
    else:
        print(
            "  [WARNING] No ffmpeg found: none pinned in "
            "SETTINGS.paths.ffmpeg_binary, none bundled with imageio-ffmpeg, "
            "and none on PATH. Video export will fail."
        )

    # 5. LaTeX (optional)
    latex_path = shutil.which("latex")
    dvisvgm_path = shutil.which("dvisvgm")
    if latex_path and dvisvgm_path:
        print(f"LaTeX (Tex/MathTex): [OK] latex={latex_path}, dvisvgm={dvisvgm_path}")
    else:
        print(
            "  [INFO] LaTeX not found (optional - standard Text mobs work via Pango without LaTeX)."
        )

    # 6. Paths -- this command's help says it reports them, and "where did my
    # video go" is the question it is most often run to answer.
    try:
        from algan.settings import SETTINGS

        paths = SETTINGS.paths
        print("Paths:")
        print(f"  output:      <script directory>/{paths.output_directory}")
        print("               (output_root defaults to the running script's own")
        print("                directory, so a render lands beside your scene)")
        print(f"  cache:       {paths.cache_directory}")
        print(f"  daemon home: {_daemon_home()}")
    except Exception as exc:  # noqa: BLE001 -- a health check must still finish
        print(f"  [WARNING] could not resolve Algan's paths: {exc}")

    print("======================================")
    return 0


def _daemon_home() -> str:
    """``$ALGAN_HOME``: the daemon state file and its log live here."""
    from algan import daemon_client

    return daemon_client.algan_home()


def _cmd_new(args: argparse.Namespace) -> int:
    """Scaffold a new Algan scene script."""
    target = Path(args.name)
    if not target.name.endswith(".py"):
        target = target.with_suffix(".py")

    if target.exists() and not args.force:
        print(
            f"Error: File '{target}' already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    template = '''"""Algan animation scene."""

from algan import *


def construct():
    # 1. Create mobs
    square = Square(color=BLUE)
    circle = Circle(color=RED, add_to_scene=False)

    # 2. Animate
    square.spawn()
    square.rotate(90)
    square.become(circle)

    # 3. Save output video
    Scene.save_video()


if __name__ == "__main__":
    construct()
'''
    target.write_text(template, encoding="utf-8")
    print(f"Created new scene file: {target.resolve()}")
    print(f"To render: algan render {target}")
    return 0


#: ``algan daemon <verb>``: the trigger-socket line commands, which are what an
#: editor keybinding pokes. Each needs the daemon's token, which is why they are
#: subcommands rather than the raw socket one-liner they replace.
DAEMON_TRIGGERS = ("render", "ping", "quit")


def _cmd_daemon(args: argparse.Namespace) -> int:
    """Manage or run the Algan warm render daemon."""
    if args.stop:
        return _trigger_daemon("quit")
    if args.trigger is not None:
        return _trigger_daemon(args.trigger)
    from algan import daemon

    print("Launching Algan render daemon...")
    print(
        "Scripts run inside it, so everything above `import algan` runs twice "
        "(once in the launching process, once here), atexit handlers do not "
        "run, and a run's stdin is /dev/null."
    )
    # An explicit empty argv. Given None, the daemon parses ``sys.argv[1:]``,
    # which here is this CLI's own arguments -- it would read "daemon" as the
    # name of a script to render and exit with "script not found".
    return daemon.main([])


def _trigger_daemon(verb: str) -> int:
    """Send one trigger verb, with the state file's token, to the daemon.

    The token is required for every verb (a bare ``quit`` from any local
    process used to be enough to stop someone else's daemon), and the state
    file is where both the token and the actual port live -- the daemon
    publishes an ephemeral port there when the preferred one is taken.
    """
    import socket

    from algan import daemon_client

    state = daemon_client.read_state()
    if state is None:
        print("No Algan daemon is running.")
        return 0
    port = int(state["port"])
    try:
        with socket.create_connection(("127.0.0.1", port), 5) as sock:
            sock.sendall(f"{verb} {state['token']}\n".encode())
            reply = sock.recv(128).decode("utf-8", "replace").strip()
    except OSError as exc:
        # A daemon killed outright leaves its registration behind, and every
        # later run then tries the dead port and falls back cold. Clear it:
        # whichever verb was asked for, nothing is answering.
        daemon_client._clear_stale_state(state)
        print(f"No daemon answered on port {port} ({exc}); cleared its registration.")
        return 0
    pid = state.get("pid", "?")
    if reply.startswith("err:"):
        print(f"Daemon (pid {pid}) refused `{verb}`: {reply[4:].strip()}")
        return 1
    if verb == "quit":
        print(f"Daemon (pid {pid}) is stopping [{reply or 'no reply'}].")
    else:
        print(f"Daemon (pid {pid}) on port {port}: {reply or 'no reply'}")
    return 0


#: ``-q`` values, each naming the Algan video preset of the same name. Algan's
#: own vocabulary rather than a second one for the same things.
QUALITY_PRESETS = ("preview", "ld", "md", "hd", "production", "uhd")


def _output_settings(output: str) -> dict[str, str]:
    """Path settings that send a script's output to ``output``.

    Algan resolves a bare name as ``output_root / output_directory / name``, so
    pointing ``output_root`` at the destination and emptying the subdirectory
    is what makes ``-o`` hold for a script that names its own video
    (``save_video("intro")`` -> ``<output>/intro.mp4``) as well as for one that
    names nothing. A path *with* a directory in it is honoured as given by
    ``save_video`` itself, which is the one case ``-o`` cannot reach.

    A value with no suffix, or a trailing separator, or an existing directory,
    is taken as a directory; anything else names the file.
    """
    path = Path(output).expanduser()
    if output.endswith(("/", os.sep)) or not path.suffix or path.is_dir():
        return {"output_root": str(path), "output_directory": ""}
    return {
        "output_root": str(path.parent),
        "output_directory": "",
        # Kept whole, suffix included: Algan honours the extension when the
        # name carries one and picks the codec's own when it does not.
        "output_filename": path.name,
    }


def _cmd_render(args: argparse.Namespace) -> int:
    """Run a scene script, with any settings the flags override applied."""
    target = Path(args.script)
    if not target.exists():
        print(f"Error: Script '{target}' not found.", file=sys.stderr)
        return 1
    target = target.resolve()

    quality = getattr(args, "quality", None)
    if args.command == "preview" and quality is None:
        quality = "preview"
    if quality is None and args.output is None:
        return _run_as_subprocess(target, args)
    return _run_here(target, args, quality)


def _run_as_subprocess(target: Path, args: argparse.Namespace) -> int:
    """Run the script the way a shell would, so the render daemon serves it."""
    from algan.environment import env_overrides

    env = os.environ.copy()
    if args.no_daemon:
        env.update(env_overrides(ALGAN_USE_DAEMON="0"))
    return subprocess.call(
        [sys.executable, str(target), *(args.extra_args or [])], env=env
    )


def _run_here(target: Path, args: argparse.Namespace, quality: str | None) -> int:
    """Run the script in this process, with the flags applied to ``SETTINGS``.

    The flags are settings, and settings do not survive a process boundary: the
    script itself is what calls ``Scene.save_video()``, so the only way to
    choose its quality or its destination from out here is to be the process it
    runs in. That costs this run the render daemon, which is why a run with no
    flags to apply is still handed to a subprocess and served warm.
    """
    import runpy

    import algan
    from algan import SETTINGS
    from algan.settings.path_settings import output_filename_for, output_root_for

    # This process's __main__ is the `algan` console script, so the output
    # defaults Algan resolved when it was imported point at *that*. Repoint
    # them at the scene, exactly as the daemon does for a client's script.
    SETTINGS.paths.set(
        output_root=output_root_for(str(target)),
        output_filename=output_filename_for(str(target)),
    )
    if quality is not None:
        SETTINGS.video.set(getattr(algan, quality.upper()))
    if args.output is not None:
        SETTINGS.paths.set(**_output_settings(args.output))

    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = [str(target), *(args.extra_args or [])]
    sys.path.insert(0, str(target.parent))  # as `python scene.py` would
    from algan.scene import renders_requested, warn_if_nothing_rendered

    before = renders_requested()
    try:
        runpy.run_path(str(target), run_name="__main__")
        warn_if_nothing_rendered(target, before)
    except SystemExit as exiting:
        code = exiting.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    finally:
        sys.argv, sys.path = old_argv, old_path
    return 0


def _split_forwarded(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` at the first ``--``: ours, then the script's.

    Done here rather than with an ``argparse.REMAINDER`` positional, which
    swallows every option after the script name -- ``algan render scene.py -q
    ld`` parsed as a script called ``scene.py`` with two arguments to forward,
    so ``-q``, ``-o`` and ``--no-daemon`` all silently did nothing.

    Arguments this CLI does not recognise are forwarded as well, so a script
    with a command line of its own (``Project.run_cli()`` reads one) still
    works without a separator. ``--`` is what settles the collisions that
    leaves: a script flag spelled like one of ours, and ``--help``, which
    otherwise prints this CLI's.
    """
    if "--" not in argv:
        return argv, []
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1 :]


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    if argv is None:
        argv = sys.argv[1:]
    argv, forwarded = _split_forwarded(argv)

    parser = argparse.ArgumentParser(
        prog="algan",
        description="Algan: High-performance 2D/3D programmatic animation engine.",
    )
    # Read from the installed metadata rather than by importing the package:
    # `algan --version` and `algan --help` used to pay the full ~3 s import
    # (torch, taichi, the vendored manim) to print one line.
    parser.add_argument(
        "-v", "--version", action="version", version=f"Algan {_version()}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # render
    render_parser = subparsers.add_parser(
        "render",
        help="Render a scene script to video",
        epilog="Arguments this CLI does not recognise, and anything after --, "
        "are forwarded to the script (which may have a command line of its "
        "own: see Project.run_cli). Use -- for a script flag spelled like one "
        "of ours, and for the script's own --help.",
        # Off so that a script's own --out is forwarded rather than being read
        # here as an abbreviation of --output.
        allow_abbrev=False,
    )
    render_parser.add_argument("script", help="Path to Python script containing Scene")
    render_parser.add_argument(
        "-q",
        "--quality",
        choices=QUALITY_PRESETS,
        default=None,
        help="Video preset to render at, unless the script names one itself",
    )
    render_parser.add_argument(
        "-o",
        "--output",
        help="Directory or file to write output to, unless the script gives a "
        "path of its own",
    )
    render_parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Bypass the warm render daemon and execute in a fresh process "
        "(-q and -o bypass it anyway: they are settings, and a warm process "
        "cannot be told them per run)",
    )
    # preview
    preview_parser = subparsers.add_parser(
        "preview",
        help="Render at the low-resolution preview preset",
        epilog="Arguments this CLI does not recognise, and anything after --, "
        "are forwarded to the script.",
        allow_abbrev=False,
    )
    preview_parser.add_argument("script", help="Path to Python script containing Scene")
    preview_parser.add_argument(
        "-o",
        "--output",
        help="Directory or file to write output to, unless the script gives a "
        "path of its own",
    )
    preview_parser.add_argument(
        "--no-daemon", action="store_true", help="Bypass the warm render daemon"
    )
    # daemon
    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Run or poke the warm render daemon",
        description="With no argument, run a daemon in this terminal. With a "
        "verb, send that trigger to the running one: `render` re-runs its last "
        "script (bind an editor key to it), `ping` checks it is alive, `quit` "
        "stops it. Each carries the token from the daemon's state file.",
    )
    daemon_parser.add_argument(
        "trigger",
        nargs="?",
        choices=DAEMON_TRIGGERS,
        default=None,
        help="Trigger to send to the running daemon",
    )
    daemon_parser.add_argument(
        "--stop", action="store_true", help="Stop running daemon (same as `quit`)"
    )

    # check
    subparsers.add_parser(
        "check", help="Check system dependencies, acceleration, and paths"
    )

    # new
    new_parser = subparsers.add_parser("new", help="Scaffold a new scene script")
    new_parser.add_argument(
        "name", help="Name of the new scene file (e.g. my_scene.py)"
    )
    new_parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite if file already exists"
    )

    # Convenience: if first arg is a .py file, treat as 'render <file.py>'
    if argv and argv[0].endswith(".py") and not argv[0].startswith("-"):
        argv = ["render"] + argv

    args, unrecognized = parser.parse_known_args(argv)
    if getattr(args, "command", None) in ("render", "preview"):
        # A scene script has a command line of its own -- ``Project.run_cli()``
        # reads one -- so an argument this CLI does not recognise is the
        # script's, not a mistake. Anything after ``--`` is the script's too,
        # and is how a script flag spelled like one of ours gets through.
        args.extra_args = [*unrecognized, *forwarded]
    elif unrecognized:
        # Nothing else here runs a script, so an unknown argument is a typo.
        parser.error(f"unrecognized arguments: {' '.join(unrecognized)}")

    if args.command == "check":
        return _cmd_check(args)
    elif args.command == "new":
        return _cmd_new(args)
    elif args.command == "daemon":
        return _cmd_daemon(args)
    elif args.command in ("render", "preview"):
        return _cmd_render(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
