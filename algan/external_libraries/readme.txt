This directory contains copy-pasted code from some PyPI packages (e.g. sect and ground).

Why did we copy-paste code from these packages instead of adding them as dependencies?
Because these packages have unreasonably large and cumbersome dependencies.
For example, sect depends on sympy, which (at least on Windows) requires the microsoft
visual studio build tools to be installed (all 7GBs of it!).
We copy pasted the relevant code from these modules to Algan so that
we can use only the parts we require without incurring these large dependencies.