This directory contains copy-pasted code from some PyPI packages (sect, ground and manim).
Each subdirectory carries its own LICENSE file with the upstream copyright notice, as MIT
requires, plus a short provenance note on the version it was taken from.

Why did we copy-paste code from these packages instead of adding them as dependencies?
Because these packages have unreasonably large and/or cumbersome transitive dependencies
for the small slice of each that Algan actually uses:
  - sect (a polygon-triangulation library) and ground (its geometry-primitives dependency)
    pull in sympy transitively, which (at least on Windows) requires the Microsoft Visual
    Studio build tools to be installed (all 7GB of it!) -- to use a handful of triangulation
    and geometry-primitive functions.
  - manim pulls pycairo and manimpango, and neither publishes a Linux wheel, so depending
    on it put a from-source build of Cairo and Pango in front of every `pip install algan`
    on Linux. Algan needs none of that: it uses Manim to *build Bezier geometry* and renders
    that with its own ray tracer, so what is vendored here is Manim Community's geometry
    subset -- the Mobject graph, the Bezier and SVG/LaTeX machinery, and the shape, graphing,
    text and 3-D classes on top of them -- with the animations, scenes, cameras, renderers,
    CLI and plugin system left out. `manim` is *not* a dependency in pyproject.toml; this
    directory is the only Manim in an Algan process, registered under the top-level name
    `manim` by algan/external_libraries/manim_alias.py so `import manim` in a user script
    reaches exactly the classes the compatibility layer checks against.

    manim/ is not hand-maintained: `scripts/vendor_manim.py` regenerates it from an upstream
    sdist, and manim/VENDORING.md lists the version, what is kept, what is dropped, and every
    edit made to upstream source.

To make the Algan install experience easier and consistent across operating systems,
we copy pasted the relevant code from these modules to Algan so that
we can use only the parts of these libraries that we require without incurring their dependencies.
