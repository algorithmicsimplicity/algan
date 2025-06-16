This directory contains copy-pasted code from some PyPI packages (e.g. sect, ground and manim).

Why did we copy-paste code from these packages instead of adding them as dependencies?
Because these packages have unreasonably large and/or cumbersome dependencies.
For example, sect depends on sympy, which (at least on Windows) requires the microsoft
visual studio build tools to be installed (all 7GBs of it!).
Manim on MacOS and Linux requires installing build tools to compile Pango.

To make the Algan install experience easier and consistent across operating systems,
we copy pasted the relevant code from these modules to Algan so that
we can use only the parts of these libraries that we require without incurring their dependencies.