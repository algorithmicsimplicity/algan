"""Build Algan's Sphinx documentation and optionally open it in a browser."""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-examples",
        action="store_true",
        help="Build without rendering embedded Algan examples.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated HTML in a browser.",
    )
    parser.add_argument(
        "--browser",
        help="Browser name understood by Python's webbrowser module.",
    )
    args = parser.parse_args()

    docs_dir = Path(__file__).resolve().parent
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    command = [
        sys.executable,
        "-m",
        "sphinx",
        "-M",
        "html",
        str(source_dir),
        str(build_dir),
    ]
    if args.skip_examples:
        command.extend(["-t", "skip-manim"])

    subprocess.run(command, cwd=docs_dir.parent, check=True)

    if not args.no_open:
        website = (build_dir / "html" / "index.html").resolve().as_uri()
        if args.browser:
            webbrowser.get(args.browser).open_new_tab(website)
        else:
            webbrowser.open_new_tab(website)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
