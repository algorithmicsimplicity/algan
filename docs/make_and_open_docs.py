"""Build Algan's Sphinx documentation and optionally open it in a browser."""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


def _sphinx_command(docs_dir: Path, *, skip_examples: bool) -> list[str]:
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"
    # The Algan directive resolves skip-manim while Sphinx reads sources, so a
    # shared doctree cache would preserve placeholders when the tag is removed.
    doctree_dir = build_dir / (
        "doctrees-without-examples" if skip_examples else "doctrees-with-examples"
    )

    command = [
        sys.executable,
        "-m",
        "sphinx",
        "-M",
        "html",
        str(source_dir),
        str(build_dir),
        "-d",
        str(doctree_dir),
    ]
    if skip_examples:
        command.extend(["-t", "skip-manim"])
    return command


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
    build_dir = docs_dir / "build"

    command = _sphinx_command(docs_dir, skip_examples=args.skip_examples)
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
