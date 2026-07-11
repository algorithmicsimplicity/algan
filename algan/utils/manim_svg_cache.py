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
selects a file under ``DIRECTORY_DEFAULTS.cache_directory``; the value is the
per-glyph point arrays + style, saved with :func:`torch.save` and rebuilt into
lightweight VMobjects (no LaTeX, no ``svgelements`` parse, no deep copy).

The directory is bounded: each save evicts least-recently-used entries once the
total exceeds a cap (default 512 MB, override via ``ALGAN_MANIM_SVG_CACHE_MB``).

It is wired in by :func:`install`, which monkeypatches
``SVGMobject.init_svg_mobject`` on the *installed* ``manim`` package (the one
Algan actually imports).  Importing this module installs the patch once.
"""

import hashlib
import importlib
import os
from pathlib import Path

import numpy as np
import torch

from algan.settings.defaults import DIRECTORY_DEFAULTS
from algan.logging.logger import get_logger

# Keys we never persist per glyph: ``path_obj`` is a large ``svgelements`` Path
# that is only needed to *produce* the points (already done); ``submobjects``
# and ``updaters`` are rebuilt structurally / reset fresh.
_SKIP_KEYS = ("path_obj", "submobjects", "updaters")

# Process-local memo of already-loaded recipes, keyed by the stable hash. This
# replaces Manim's SVG_HASH_TO_MOB_MAP for the patched path: a cache hit rebuilds
# fresh mobjects from the recipe instead of deep-copying a stored mobject.
_MEM_CACHE: dict[str, tuple] = {}

_installed = False

# Total on-disk cap for the cache directory. Once a save pushes the directory
# past this, least-recently-used files are evicted until it fits. Override with
# ALGAN_MANIM_SVG_CACHE_MB (0 or negative disables the cap).
_DEFAULT_CACHE_MB = 512


def _max_cache_bytes() -> int:
    try:
        mb = float(os.environ.get("ALGAN_MANIM_SVG_CACHE_MB", _DEFAULT_CACHE_MB))
    except ValueError:
        mb = _DEFAULT_CACHE_MB
    return int(mb * 1024 * 1024)


def _cache_dir() -> Path:
    return Path(DIRECTORY_DEFAULTS.cache_directory) / "manim_svg"


def _stable_key(svg_mob) -> str:
    """A cross-process content hash for an SVG mobject's parse result.

    Manim's own ``hash_seed`` is fine for an in-process dict but relies on
    ``hash()`` of strings, which is salted per process. We hash a canonical
    string instead. The SVG file *basename* already encodes the full LaTeX
    source, template, environment and preamble (it is ``tex_hash(...) + .svg``),
    so keying on the basename (not the absolute path) is both stable and
    content-addressed across machines / working directories.
    """
    file_name = svg_mob.file_name
    file_id = Path(file_name).name if file_name is not None else "None"

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


def _extract(svg_mob) -> tuple:
    """Recipe for an SVGMobject's parsed children (its top-level glyphs)."""
    return tuple(_extract_node(sm) for sm in svg_mob.submobjects)


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


def _rebuild(svg_mob, recipe: tuple) -> None:
    svg_mob.add(*[_rebuild_node(node) for node in recipe])


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
    try:
        os.utime(path, None)
    except OSError:
        pass
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
            try:
                tmp.unlink()
            except OSError:
                pass
        return
    _enforce_cap()


def _patched_init_svg_mobject(self, use_svg_cache: bool) -> None:
    """Drop-in replacement for ``SVGMobject.init_svg_mobject``.

    Adds a persistent disk layer in front of the (still process-local) parse:
    memo -> disk -> generate. On a fresh parse the original glyph tree is kept
    untouched (byte-identical to the un-cached path) and its recipe is written
    to disk for later runs.
    """
    if not use_svg_cache:
        self.generate_mobject()
        return

    key = _stable_key(self)

    recipe = _MEM_CACHE.get(key)
    if recipe is None:
        recipe = _load_disk(key)
        if recipe is None:
            # Cold: parse for real, then persist the result.
            self.generate_mobject()
            _MEM_CACHE[key] = _extract(self)
            _save_disk(key, _MEM_CACHE[key])
            return
        _MEM_CACHE[key] = recipe

    _rebuild(self, recipe)


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
        get_logger().warning(f"manim_svg_cache: could not install ({e}); Tex geometry uncached")
        return

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
