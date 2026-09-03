# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Import the checkout being documented, regardless of the caller's cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from algan import __version__ as algan_version
from algan.utils.docbuild.module_parsing import parse_module_attributes

# -- Project information -----------------------------------------------------

project = "Algan"
copyright = f"2025-{datetime.now().year}, Algorithmic Simplicity"  # noqa: A001
author = "Algorithmic Simplicity"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",
    "sphinxext.opengraph",
    "algan.utils.docbuild.algan_directive",
    "algan.utils.docbuild.autocolor_directive",
    "algan.utils.docbuild.autoaliasattr_directive",
    "algan.utils.docbuild.manim_example_directive",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.programoutput",
    "myst_parser",
    "sphinx_design",
    "sphinx_reredirects",
]

# Automatically generate stub pages when using the .. autosummary directive
autosummary_generate = True
"""autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}"""

myst_enable_extensions = ["colon_fence", "amsmath"]

# redirects (for moved / deleted pages)
redirects = {
    "installation/linux": "uv.html",
    "installation/macos": "uv.html",
    "installation/windows": "uv.html",
    # Duplicates of installation/uv, removed so one page is canonical.
    "installation/installation": "uv.html",
    "new_user_tutorials/installation": "../installation/uv.html",
    # The Manim quickstart collection became one page.
    "manim_user_quickstart/index": "../manim_migration_guide.html",
    "manim_user_quickstart/migrating_from_manim": "../manim_migration_guide.html",
    # The two catalogue pages moved out of the advanced tutorials.
    "advanced_user_tutorials/mob_gallery": "../galleries/mob_gallery.html",
    "advanced_user_tutorials/built_in_animations": (
        "../galleries/built_in_animations.html"
    ),
}

# generate documentation from type hints
ALIAS_DOCS_DICT = parse_module_attributes()[0]
autodoc_typehints = "description"
autodoc_type_aliases = {
    alias_name: f"~algan.{module}.{alias_name}"
    for module, module_dict in ALIAS_DOCS_DICT.items()
    for category_dict in module_dict.values()
    for alias_name in category_dict
}
autoclass_content = "both"

# controls whether functions documented by the autofunction directive
# appear with their full module names
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Custom section headings in our documentation. "Animation" is the section every
# user-facing scene-mutating method carries (see DOCSTRINGS.md); napoleon only
# recognizes a NumPy-style heading that is registered here, and an unregistered
# one passes through verbatim and breaks the build.
napoleon_custom_sections = ["Animation", "Tests", ("Test", "Tests")]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
html_extra_path = ["robots.txt"]

exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "furo"
html_favicon = str(Path("_static/favicon.ico"))

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/algorithmicsimplicity/algan/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    # Placeholder marks -- swap the two SVGs in _static for real Algan artwork.
    "light_logo": "algan-logo-sidebar.svg",
    "dark_logo": "algan-logo-sidebar-dark.svg",
    "light_css_variables": {
        "color-content-foreground": "#000000",
        "color-background-primary": "#ffffff",
        "color-background-border": "#ffffff",
        "color-sidebar-background": "#f8f9fb",
        "color-brand-content": "#1c00e3",
        "color-brand-primary": "#192bd0",
        "color-link": "#c93434",
        "color-link--hover": "#5b0000",
        "color-inline-code-background": "#f6f6f6;",
        "color-foreground-secondary": "#000",
    },
    "dark_css_variables": {
        "color-content-foreground": "#ffffffd9",
        "color-background-primary": "#131416",
        "color-background-border": "#303335",
        "color-sidebar-background": "#1a1c1e",
        "color-brand-content": "#2196f3",
        "color-brand-primary": "#007fff",
        "color-link": "#51ba86",
        "color-link--hover": "#9cefc6",
        "color-inline-code-background": "#262626",
        "color-foreground-secondary": "#ffffffd9",
    },
}
html_title = f"Algan v{algan_version}"

# This specifies any additional css files that will override the theme's
html_css_files = ["custom.css"]


# external links
extlinks = {
    "issue": ("https://github.com/algorithmicsimplicity/algan/issues/%s", "#%s"),
    "pr": ("https://github.com/algorithmicsimplicity/algan/pull/%s", "#%s"),
}

# opengraph settings
ogp_site_name = "Algan | Documentation"
ogp_site_url = "https://algorithmicsimplicity.github.io/algan/"
ogp_social_cards = {
    "image": "_static/logo.png",
}


# inheritance_graph settings
inheritance_graph_attrs = {
    "concentrate": True,
    "size": '""',
    "splines": "ortho",
    "nodesep": 0.1,
    "ranksep": 0.2,
}

inheritance_node_attrs = {
    "penwidth": 0,
    "shape": "box",
    "width": 0.05,
    "height": 0.05,
    "margin": 0.05,
}

inheritance_edge_attrs = {
    "penwidth": 1,
}

html_js_files = ["responsiveSvg.js"]

graphviz_output_format = "svg"


# `Color` subclasses `torch.Tensor`, so its autosummary attribute table lists
# ~40 tensor internals (`is_mkldnn`, `output_nr`, `nbytes`, ...) alongside its
# three real ones. The class template already drops inherited *methods*; this
# drops inherited attributes too, by identity against `torch.Tensor`, so a
# subclass documents only what Algan defines on it.
def _skip_inherited_tensor_members(app, what, name, obj, skip, options):
    if skip:
        return skip
    import torch

    if obj is not None and obj is getattr(torch.Tensor, name, None):
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_inherited_tensor_members)
