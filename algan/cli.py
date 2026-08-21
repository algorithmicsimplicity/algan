"""Command-line interface for Algan animation engine.

Provides commands for rendering scenes, previewing frames, managing the warm daemon,
checking environment health, and scaffolding new scenes.
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
    from algan import daemon

    if args.stop:
        print("Stopping any running Algan daemon...")
        # algan.daemon stops when asked or on code change
        return 0
    print("Launching Algan render daemon...")
    daemon.main()
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    """Execute a Python scene script to render video."""
    target = Path(args.script)
    if not target.exists():
        print(f"Error: Script '{target}' not found.", file=sys.stderr)
        return 1

    env = os.environ.copy()
    if args.quality:
        env["ALGAN_QUALITY"] = args.quality
    if args.overwrite:
        env["ALGAN_OVERWRITE"] = "1"
    if args.no_daemon:
        env["ALGAN_NO_DAEMON"] = "1"

    cmd = [sys.executable, str(target.resolve())]
    if args.extra_args:
        cmd.extend(args.extra_args)

    return subprocess.call(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="algan",
        description="Algan: High-performance 2D/3D programmatic animation engine.",
    )
    parser.add_argument("-v", "--version", action="version", version="Algan 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # render
    render_parser = subparsers.add_parser(
        "render", help="Render a scene script to video"
    )
    render_parser.add_argument("script", help="Path to Python script containing Scene")
    render_parser.add_argument(
        "-q",
        "--quality",
        choices=["high", "medium", "low"],
        default=None,
        help="Render quality preset",
    )
    render_parser.add_argument("-o", "--output", help="Output file destination")
    render_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    render_parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Bypass warm daemon and execute in fresh process",
    )
    render_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments to forward to the script",
    )

    # preview
    preview_parser = subparsers.add_parser(
        "preview", help="Render still frames / preview"
    )
    preview_parser.add_argument("script", help="Path to Python script containing Scene")
    preview_parser.add_argument("-o", "--output", help="Output file destination")
    preview_parser.add_argument(
        "--no-daemon", action="store_true", help="Bypass warm daemon"
    )
    preview_parser.add_argument(
        "extra_args", nargs=argparse.REMAINDER, help="Extra arguments"
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
