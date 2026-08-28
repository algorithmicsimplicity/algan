"""Persistent (cross-process) cache for Manim SVG/Tex geometry.

Manim renders :class:`~manim.Text`/:class:`~manim.Tex`/:class:`~manim.MathTex`
by compiling LaTeX to an SVG and then parsing that SVG into a tree of
:class:`~manim.VMobject` bezier glyphs.  The LaTeX -> SVG step is already
cached to disk by Manim, but the SVG -> VMobject *parse* is not: Manim only
memoises it in a process-local dict (``SVG_HASH_TO_MOB_MAP``), and it stores /
retrieves entries by deep-copying the whole glyph tree.

For a large label (e.g. a 2500-glyph block of text) that parse plus the two
deep copies costs ~60s, and because the memo is in-memory only, *every* fresh
render process pays it again.  This module makes the parse result persist to
disk so the control points are generated exactly once and cheaply reloaded on
every subsequent run.

The design mirrors the on-disk geometry cache used by
:mod:`algan.mobs.triangulated_bezier_circuit`: a stable ``sha256`` content key
selects a file under ``SETTINGS.paths.cache_directory``; the value is the
per-glyph point arrays + style, saved with :func:`torch.save` and rebuilt into
lightweight VMobjects (no LaTeX, no ``svgelements`` parse, no deep copy).

Manim 0.21 added ``SVGMobject.id_to_vgroup_dict``, which maps each SVG element
id to a ``VGroup`` of the glyphs beneath it. ``MathTex._break_up_by_substrings``
indexes it to split a formula into its ``tex_strings``, so a cache hit that
rebuilt only the glyph tree left the dict empty and every ``Tex`` after the very
first (i.e. every run once this cache is warm) died on ``KeyError: 'root'``.
The recipe therefore also stores, per group id, the indices of its members in
the top-level glyph list, and :func:`_rebuild` reconstructs the dict from them.
Manim 0.19 has no such attribute; the group map is simply absent there, so both
versions round-trip through the same cache. Recipes are tagged with
:data:`_RECIPE_TAG` so entries written by an older Algan (or under a manim that
did not need the map) are detected and transparently re-parsed rather than
replayed into a broken mobject.

The directory is bounded: each save evicts least-recently-used entries once the
total exceeds a cap (default 512 MB, override via ``ALGAN_MANIM_SVG_CACHE_MB``).

It is wired in by :func:`install`, which monkeypatches
``SVGMobject.init_svg_mobject`` on the *installed* ``manim`` package (the one
Algan actually imports).  Importing this module installs the patch once.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
from functools import wraps
from pathlib import Path

import numpy as np
import torch

from algan.environment import env_float
from algan.logging.logger import get_logger
from algan.settings import SETTINGS

# Keys we never persist per glyph: ``path_obj`` is a large ``svgelements`` Path
# that is only needed to *produce* the points (already done); ``submobjects``
# and ``updaters`` are rebuilt structurally / reset fresh.
_SKIP_KEYS = ("path_obj", "submobjects", "updaters")

# Marks a recipe as ``(_RECIPE_TAG, nodes, groups)``. Recipes without it predate
# the group map and are re-parsed when the running manim needs one. Bump the
# suffix whenever the payload layout changes so old entries are re-parsed
# instead of misread -- the cache key is content-addressed, not version-keyed,
# so a manim upgrade otherwise reads a recipe written by the previous version.
_RECIPE_TAG = "algan-manim-svg-recipe-v2"

# Process-local memo of already-loaded recipes, keyed by the stable hash. This
# replaces Manim's SVG_HASH_TO_MOB_MAP for the patched path: a cache hit rebuilds
# fresh mobjects from the recipe instead of deep-copying a stored mobject.
_MEM_CACHE: dict[str, tuple] = {}

_installed = False

# Total on-disk cap for the cache directory. Once a save pushes the directory
# past this, least-recently-used files are evicted until it fits. Override with
# ALGAN_MANIM_SVG_CACHE_MB (0 or negative disables the cap).
_DEFAULT_CACHE_MB = 1024


def _max_cache_bytes() -> int:
    return int(env_float("ALGAN_MANIM_SVG_CACHE_MB", _DEFAULT_CACHE_MB) * 1024 * 1024)


def _cache_dir() -> Path:
    return Path(SETTINGS.paths.cache_directory) / "manim_svg"


def _manim_generated_svg_basename_is_content_addressed(path: Path) -> bool:
    """True when ``path`` is one of manim's own generated SVGs.

    Manim names both kinds after a hash of what produced them --
    ``tex_hash(source) + .svg`` in ``tex_dir``, ``_text2hash(settings) + .svg``
    in ``text_dir`` -- so for these the basename already encodes the LaTeX
    source, template, environment and preamble, or the string, font and color.
    Everything else -- a user's own ``logo.svg`` -- has a basename that says
    nothing about its contents.
    """
    try:
        from manim import config

        parent = path.parent.resolve()
        return any(
            parent == Path(config.get_dir(name)).resolve()
            for name in ("tex_dir", "text_dir")
        )
    except Exception:  # noqa: BLE001 - a key that falls back to hashing is safe
        return False


def _svg_content_id(file_name) -> str:
    """The cache identity of an SVG source file.

    For manim's own generated SVGs -- Tex and Pango text -- the basename is
    already a content hash, and hashing the SVG bytes instead would key on
    dvisvgm's exact version and output ordering, breaking the cross-machine
    sharing this cache is built for. So those keep the basename.

    Every other SVG -- one the user drew -- is keyed on its **contents**.
    Keying a user file on its basename meant that editing ``logo.svg`` and
    re-running silently replayed the previous drawing, forever and across
    processes, and that two unrelated ``logo.svg`` files collided. It also meant
    a *deleted* file kept resolving: this cache is consulted before manim ever
    looks the path up, so the stale entry answered instead of manim raising.
    """
    if file_name is None:
        return "None"
    path = Path(file_name)
    if _manim_generated_svg_basename_is_content_addressed(path):
        return path.name

    # ``file_name`` is whatever the caller passed; manim only resolves it later,
    # in ``generate_mobject``. Resolve it the same way manim will, so the key
    # describes the bytes that will actually be parsed.
    try:
        from manim.utils.images import get_full_vector_image_path

        path = Path(get_full_vector_image_path(path))
    except Exception:  # noqa: BLE001 - fall through to the miss below
        pass

    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        # No readable file. Miss deliberately rather than replaying a stale
        # entry, and let manim's own lookup raise the real error.
        return f"{path.name}:<unresolved>"
    return f"{path.name}:{digest}"


def _stable_key(svg_mob) -> str:
    """A cross-process content hash for an SVG mobject's parse result.

    Manim's own ``hash_seed`` is fine for an in-process dict but relies on
    ``hash()`` of strings, which is salted per process. We hash a canonical
    string instead, identifying the source file by
    :func:`_svg_content_id` so the key is stable and content-addressed across
    machines and working directories.
    """
    file_id = _svg_content_id(svg_mob.file_name)

    def canon(d):
        return "{" + ",".join(f"{k}={d[k]}" for k in sorted(d)) + "}"

    parts = [
        type(svg_mob).__name__,
        canon(svg_mob.svg_default),
        canon(svg_mob.path_string_config),
        file_id,
        str(getattr(svg_mob, "_renderer_type", "")),
    ]
    hasher = hashlib.sha256()
    hasher.update("|".join(parts).encode())
    return hasher.hexdigest()


def _extract_node(mob) -> tuple:
    """Snapshot ``mob`` (recursively) into a picklable recipe.

    Returns ``(module, qualname, state, children)`` where ``state`` is the
    instance ``__dict__`` minus :data:`_SKIP_KEYS`.
    """
    state = {k: v for k, v in mob.__dict__.items() if k not in _SKIP_KEYS}
    children = [_extract_node(sm) for sm in mob.submobjects]
    cls = type(mob)
    return (cls.__module__, cls.__qualname__, state, children)


def _extract_groups(svg_mob) -> dict[str, list[int]] | None:
    """Snapshot ``id_to_vgroup_dict`` as indices into the top-level glyph list.

    The VGroups hold *references* to the very glyphs in ``svg_mob.submobjects``
    (``get_mobjects_from`` adds each leaf shape to both), which is what lets
    ``MathTex`` re-parent them. Positions survive pickling where identity does
    not, so store those and re-resolve them against the rebuilt glyphs.

    Returns ``None`` on manim versions without the attribute (< 0.21).
    """
    id_to_vgroup = getattr(svg_mob, "id_to_vgroup_dict", None)
    if id_to_vgroup is None:
        return None
    index_of = {id(sm): i for i, sm in enumerate(svg_mob.submobjects)}
    return {
        name: [index_of[id(m)] for m in vgroup.submobjects if id(m) in index_of]
        for name, vgroup in id_to_vgroup.items()
    }


def _extract(svg_mob) -> tuple:
    """Recipe for an SVGMobject's parsed children (its top-level glyphs)."""
    nodes = tuple(_extract_node(sm) for sm in svg_mob.submobjects)
    return (_RECIPE_TAG, nodes, _extract_groups(svg_mob))


def _parse_recipe(recipe) -> tuple[tuple, dict[str, list[int]] | None]:
    """Split a stored recipe into ``(nodes, groups)``.

    Untagged recipes are pre-group entries left by an older Algan; their first
    element is a node 4-tuple rather than the string tag, so the two layouts are
    never confusable.
    """
    if len(recipe) == 3 and recipe[0] == _RECIPE_TAG:
        return recipe[1], recipe[2]
    return recipe, None


def _restore_value(v):
    # Copy mutable arrays so sibling rebuilds (and repeated cache hits) never
    # alias the same buffer -- downstream code mutates points/rgbas in place.
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, list):
        return list(v)
    if isinstance(v, dict):
        return dict(v)
    return v


_class_cache: dict[tuple, type] = {}


def _resolve_class(module: str, qualname: str) -> type:
    key = (module, qualname)
    cls = _class_cache.get(key)
    if cls is None:
        obj = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        cls = obj
        _class_cache[key] = cls
    return cls


def _rebuild_node(node: tuple):
    module, qualname, state, children = node
    cls = _resolve_class(module, qualname)
    mob = cls.__new__(cls)  # bypass __init__: we restore state directly
    mob.__dict__.update({k: _restore_value(v) for k, v in state.items()})
    mob.submobjects = [_rebuild_node(c) for c in children]
    mob.updaters = []
    return mob


def _rebuild(svg_mob, nodes: tuple, groups: dict[str, list[int]] | None) -> None:
    """Replay a recipe onto ``svg_mob``: glyph tree first, then the group map.

    Everything is constructed before anything is attached, so a rebuild that
    raises part-way leaves ``svg_mob`` untouched for the caller to re-parse.
    """
    children = [_rebuild_node(node) for node in nodes]
    rebuilt_groups = None
    if groups is not None:
        from manim.mobject.types.vectorized_mobject import VGroup

        rebuilt_groups = {}
        for name, indices in groups.items():
            members = [children[i] for i in indices if 0 <= i < len(children)]
            vgroup = VGroup()
            if members:
                vgroup.add(*members)
            rebuilt_groups[name] = vgroup

    svg_mob.add(*children)
    if rebuilt_groups is not None:
        svg_mob.id_to_vgroup_dict = rebuilt_groups


def _load_disk(key: str):
    path = _cache_dir() / f"{key}.pt"
    if not path.exists():
        return None
    try:
        recipe = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - a corrupt cache must never be fatal
        get_logger().warning(f"manim_svg_cache: failed to load {path.name}: {e}")
        return None
    # Touch so the LRU cap treats this as recently used.
    with contextlib.suppress(OSError):
        os.utime(path, None)
    return recipe


def _enforce_cap() -> None:
    """Evict least-recently-used cache files until the dir fits the cap.

    LRU is approximated by file mtime, which ``_load_disk`` bumps on every hit.
    """
    cap = _max_cache_bytes()
    if cap <= 0:
        return
    d = _cache_dir()
    try:
        entries = []
        total = 0
        for f in d.glob("*.pt"):
            try:
                st = f.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, f))
            total += st.st_size
        if total <= cap:
            return
        entries.sort()  # oldest first
        for _mtime, size, f in entries:
            if total <= cap:
                break
            try:
                f.unlink()
                total -= size
            except OSError:
                pass
    except Exception as e:  # noqa: BLE001 - eviction is best-effort, never fatal
        get_logger().warning(f"manim_svg_cache: cap enforcement failed: {e}")


def _save_disk(key: str, recipe: tuple) -> None:
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.pt"
    tmp = d / f"{key}.pt.{os.getpid()}.tmp"
    try:
        torch.save(recipe, tmp)
        os.replace(tmp, path)  # atomic: never leave a half-written cache file
    except Exception as e:  # noqa: BLE001
        get_logger().warning(f"manim_svg_cache: failed to save {path.name}: {e}")
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        return
    _enforce_cap()


def _patched_init_svg_mobject(self, use_svg_cache: bool):
    """Drop-in replacement for ``SVGMobject.init_svg_mobject``.

    Adds a persistent disk layer in front of the (still process-local) parse:
    memo -> disk -> generate. On a fresh parse the original glyph tree is kept
    untouched (byte-identical to the un-cached path) and its recipe is written
    to disk for later runs.

    A recipe is only replayed when it can reproduce everything this manim reads
    back off the mobject; otherwise it is discarded and re-parsed, which
    re-saves it in the current format. That covers upgrading into a manim that
    needs the group map, and glyph classes moving between manim versions (the
    cache key is content-addressed, so an upgrade keeps hitting the same entry).
    """
    if not use_svg_cache:
        self.generate_mobject()
        return self

    key = _stable_key(self)
    # Manim >= 0.21 initialises this in __init__, before calling us.
    needs_groups = hasattr(self, "id_to_vgroup_dict")

    recipe = _MEM_CACHE.get(key)
    if recipe is None:
        recipe = _load_disk(key)
        if recipe is not None:
            _MEM_CACHE[key] = recipe

    if recipe is not None:
        nodes, groups = _parse_recipe(recipe)
        if groups is None and needs_groups:
            # Pre-group entry from an older Algan: replaying it would leave
            # id_to_vgroup_dict empty and break MathTex. Re-parse instead.
            _MEM_CACHE.pop(key, None)
        else:
            try:
                _rebuild(self, nodes, groups)
                return self
            except Exception as e:  # noqa: BLE001 - a stale recipe is recoverable
                get_logger().warning(
                    f"manim_svg_cache: could not replay {key[:12]} ({e}); reparsing"
                )
                _MEM_CACHE.pop(key, None)
                self.submobjects = []

    # Cold (or unusable recipe): parse for real, then persist the result.
    self.generate_mobject()
    _MEM_CACHE[key] = _extract(self)
    _save_disk(key, _MEM_CACHE[key])
    return self


def _redirect_manim_dirs() -> None:
    """Point manim's tex/text output into Algan's cache directory.

    Manim compiles LaTeX (and renders Pango text) into ``{media_dir}/Tex`` /
    ``{media_dir}/texts`` -- relative to the *current working directory* by
    default, so every project re-pays every LaTeX compile. Redirect both into
    ``SETTINGS.paths.cache_directory`` (content-hashed filenames make the
    cache safely shareable across projects).

    Done here (this module rides along on Algan's first manim import) rather
    than in the ``Tex``/``Text`` mobs so that raw manim mobjects wrapped in
    :class:`~algan.mobs.manim_mob.ManimMob` use the same cache.
    """
    from manim import config
    from manim.utils import tex_file_writing

    _configure_manim_dirs(config, create=False)

    # Manim's Text path creates ``text_dir`` with mkdir(parents=True), but its
    # Tex path (``generate_tex_file``) uses a *single-level* mkdir, which
    # crashes whenever the cache directory tree has been removed (e.g.
    # ``clear_cache()``, or the test suite's per-test wipe). Wrap it to
    # guarantee the directory exists, whole tree included, on every call.
    #
    # Algan exposes its vendored Manim package through the top-level ``manim``
    # alias. Python can consequently load ``tex_file_writing`` under both
    # module names; MathTex may hold the vendored instance while this redirect
    # imported the aliased instance. Patch both so a later cache wipe is safe
    # regardless of which module owns ``tex_to_svg_file``.
    tex_modules = [tex_file_writing]
    with contextlib.suppress(ImportError):
        from algan.external_libraries.manim.utils import (
            tex_file_writing as vendored_tex_file_writing,
        )

        tex_modules.append(vendored_tex_file_writing)

    for module in dict.fromkeys(tex_modules):
        original = module.generate_tex_file
        if getattr(original, "_algan_ensures_tex_dir", False):
            continue

        @wraps(original)
        def generate_tex_file_with_dir(*args, _original=original, **kwargs):
            Path(config.get_dir("tex_dir")).mkdir(parents=True, exist_ok=True)
            return _original(*args, **kwargs)

        generate_tex_file_with_dir._algan_ensures_tex_dir = True
        module.generate_tex_file = generate_tex_file_with_dir


def _configure_manim_dirs(config, *, create: bool = True) -> tuple[Path, Path]:
    """Keep Manim's Text/Tex scratch files inside Algan's cache tree.

    This is also called by :func:`algan.mobs.text.make_manim_dir`, because the
    runtime-adjustable content-cache setting may have moved since Manim was
    first imported.

    ``create`` is false on the import-time redirect: importing Algan must not
    write to disk. The directories are then made on first use instead -- Manim
    creates ``text_dir`` itself with ``parents=True``, and the
    ``generate_tex_file`` wrapper installed below does the same for ``tex_dir``.
    """
    manim_dir = Path(SETTINGS.paths.cache_directory) / "manim"
    config.tex_dir = os.fspath(manim_dir / "Tex")
    config.text_dir = os.fspath(manim_dir / "texts")

    tex_dir = Path(config.get_dir("tex_dir"))
    text_dir = Path(config.get_dir("text_dir"))
    if create:
        tex_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
    return tex_dir, text_dir


def install() -> None:
    """Monkeypatch the installed ``manim`` to use the persistent SVG cache.

    Idempotent. Called on import of this module. We patch the *installed*
    ``manim`` package (what Algan imports at runtime) rather than the vendored
    copy under ``algan.external_libraries`` -- the vendored copy is only used
    for a handful of helpers and is not on the Tex geometry path.
    """
    global _installed
    if _installed:
        return
    try:
        from manim.mobject.svg import svg_mobject as _svg
    except Exception as e:  # noqa: BLE001 - never break import if manim moves things
        get_logger().warning(
            f"manim_svg_cache: could not install ({e}); Tex geometry uncached"
        )
        return

    try:
        _redirect_manim_dirs()
    except Exception as e:  # noqa: BLE001 - a failed redirect must not break manim
        get_logger().warning(f"manim_svg_cache: cache-dir redirect failed ({e})")

    # Stash the renderer type on the class so _stable_key can read it without
    # importing manim.config here.
    try:
        from manim import config as _config

        _svg.SVGMobject._renderer_type = _config.renderer
    except Exception:  # noqa: BLE001
        pass

    _svg.SVGMobject.init_svg_mobject = _patched_init_svg_mobject
    _installed = True


install()
