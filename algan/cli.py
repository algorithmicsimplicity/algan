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


def _cmd_check(_args: argparse.Namespace) -> int:
    """Check system environment and dependencies."""
    print("=== Algan Environment Health Check ===")

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
    except ImportError:
        print("  [ERROR] PyTorch is not installed.")

    # 3. Taichi runtime
    try:
        import taichi as ti

        print(f"Taichi: {ti.__version__}")
    except ImportError:
        print("  [ERROR] Taichi is not installed.")

    # 4. FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"FFmpeg: [OK] {ffmpeg_path}")
    else:
        print("  [WARNING] FFmpeg not found on PATH. Video export may fail.")

    # 5. LaTeX (optional)
    latex_path = shutil.which("latex")
    dvisvgm_path = shutil.which("dvisvgm")
    if latex_path and dvisvgm_path:
        print(f"LaTeX (Tex/MathTex): [OK] latex={latex_path}, dvisvgm={dvisvgm_path}")
    else:
        print(
            "  [INFO] LaTeX not found (optional - standard Text mobs work via Pango without LaTeX)."
        )

    print("======================================")
    return 0


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


def _cmd_daemon(args: argparse.Namespace) -> int:
    """Manage or run the Algan warm render daemon."""
    if args.stop:
        return _stop_daemon()
    from algan import daemon

    print("Launching Algan render daemon...")
    # An explicit empty argv. Given None, the daemon parses ``sys.argv[1:]``,
    # which here is this CLI's own arguments -- it would read "daemon" as the
    # name of a script to render and exit with "script not found".
    return daemon.main([])


def _stop_daemon() -> int:
    """Ask a running daemon to quit, through its trigger socket."""
    import socket

    from algan import daemon_client

    state = daemon_client.read_state()
    if state is None:
        print("No Algan daemon is running.")
        return 0
    port = int(state["port"])
    try:
        with socket.create_connection(("127.0.0.1", port), 5) as sock:
            sock.sendall(b"quit\n")
            reply = sock.recv(16).decode("utf-8", "replace").strip()
    except OSError as exc:
        # A daemon killed outright leaves its registration behind, and every
        # later run then tries the dead port and falls back cold. Since we came
        # here to make sure no daemon is running, clear it.
        daemon_client._clear_stale_state(state)
        print(f"No daemon answered on port {port} ({exc}); cleared its registration.")
        return 0
    print(f"Daemon (pid {state.get('pid', '?')}) is stopping [{reply or 'no reply'}].")
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
    try:
        runpy.run_path(str(target), run_name="__main__")
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
    so ``-q``, ``-o`` and ``--no-daemon`` all silently did nothing. Splitting
    first means an unknown flag of ours is still an error rather than being
    quietly handed to the script.
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
    import algan

    parser.add_argument(
        "-v", "--version", action="version", version=f"Algan {algan.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # render
    render_parser = subparsers.add_parser(
        "render",
        help="Render a scene script to video",
        epilog="Arguments after -- are forwarded to the script.",
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
        epilog="Arguments after -- are forwarded to the script.",
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
    daemon_parser = subparsers.add_parser("daemon", help="Manage warm render daemon")
    daemon_parser.add_argument(
        "--stop", action="store_true", help="Stop running daemon"
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

    args = parser.parse_args(argv)
    args.extra_args = forwarded

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
