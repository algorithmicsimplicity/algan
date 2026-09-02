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
  - manim/ here is the SVG/Tex rendering subset (LaTeX-to-path parsing, SVG path data,
    the small piece of Manim's Mobject machinery those need), not a full copy of Manim.
    Note that `manim` is, as of this writing, *also* still a direct dependency in
    pyproject.toml for the rest of Algan's Manim-compatibility layer (see RELEASE_AUDIT.md
    #2 for the resulting two-Manims situation and the pending decision on it); this
    directory's copy exists so the SVG/Tex subset doesn't require anything beyond what is
    vendored here, independent of whichever way that decision goes.

To make the Algan install experience easier and consistent across operating systems,
we copy pasted the relevant code from these modules to Algan so that
we can use only the parts of these libraries that we require without incurring their dependencies.