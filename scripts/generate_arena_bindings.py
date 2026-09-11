"""Generate packed arena bindings without importing Algan, torch or a compiler."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

DTYPE_TAGS = ("f32", "i32", "i64", "f16", "u8")


def _validated_fields(fields):
    names, spec = [], []
    for field in fields:
        if not isinstance(field, str) and (
            not isinstance(field, (tuple, list)) or len(field) != 3
        ):
            raise ValueError(f"invalid arena field: {field!r}")
        name = field if isinstance(field, str) else field[0]
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError(f"invalid or repeated argument name: {name!r}")
        names.append(name)
        if not isinstance(field, str):
            if len(field) != 3 or field[1] not in DTYPE_TAGS:
                raise ValueError(f"invalid arena field: {field!r}")
            if type(field[2]) is not int or field[2] < 1:
                raise ValueError(f"invalid arena rank: {field!r}")
            spec.append(tuple(field))
    return names, spec


def _replace_region(source, kernel, kind, body, indent=""):
    start = f"{indent}# BEGIN GENERATED {kernel} {kind}"
    end = f"{indent}# END GENERATED {kernel} {kind}"
    pattern = re.escape(start) + r"\n.*?" + re.escape(end)
    if len(re.findall(pattern, source, flags=re.S)) != 1:
        raise ValueError(f"expected exactly one {kernel} {kind} region")
    return re.sub(pattern, lambda _: f"{start}\n{body}{end}", source, flags=re.S)


def generate_sources(root):
    """Return expected kernel source by path; validate all layouts before writing."""
    directory = Path(root) / "algan/rendering/raytracing"
    schema = ast.parse((directory / "arena_layouts.py").read_text(encoding="utf-8"))
    value = next(
        node.value
        for node in schema.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "LAYOUTS" for t in node.targets)
    )
    layouts = ast.literal_eval(value)
    sources = {}
    for kernel, (filename, fields) in layouts.items():
        if Path(filename).name != filename or not filename.endswith("_taichi.py"):
            raise ValueError(f"invalid kernel filename: {filename!r}")
        names, spec = _validated_fields(fields)
        path = directory / filename
        source = sources.get(path, path.read_text(encoding="utf-8"))
        tree = ast.parse(source)
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == f"{kernel}_arena"
        )
        tags = [tag for tag in DTYPE_TAGS if any(s[1] == tag for s in spec)]
        kept = [field for field in fields if isinstance(field, str)]
        expected = kept + [f"arena_{tag}" for tag in tags] + ["aoff", "ashp"]
        actual = [a.arg for a in fn.args.args]
        if actual != expected:
            raise ValueError(
                f"{kernel}: kernel signature disagrees with layout: {actual} != {expected}"
            )
        cursor = 0
        bindings = []
        for offset, (name, tag, rank) in enumerate(spec):
            dims = ", ".join(f"ashp[{j}]" for j in range(cursor, cursor + rank))
            if rank == 1:
                dims += ","
            bindings.append(
                f"    {name} = ti.static(ArenaView(arena_{tag}, aoff[{offset}], ({dims})))\n"
            )
            cursor += rank
        source = _replace_region(source, kernel, "bindings", "".join(bindings), "    ")
        prefix = f"_{kernel.upper()}"
        lines = [f"{prefix}_ARENA = (\n"]
        lines += [f'    ("{name}", "{tag}", {rank}),\n' for name, tag, rank in spec]
        lines += [")\n\n", f"{prefix}_PARAMS = (\n"]
        lines += [f'    "{name}",\n' for name in names]
        lines += [")\n"]
        source = _replace_region(source, kernel, "layout", "".join(lines))
        sources[path] = source
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail on stale generated regions without writing",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = generate_sources(root)
    stale = [p for p, s in sources.items() if p.read_text(encoding="utf-8") != s]
    if args.check and stale:
        parser.exit(
            1,
            "Stale arena bindings: "
            + ", ".join(str(p.relative_to(root)) for p in stale)
            + "\n",
        )
    for path in stale:
        path.write_text(sources[path], encoding="utf-8")
    print(
        f"Arena bindings {'checked' if args.check else 'generated'}: {len(sources)} modules"
    )


if __name__ == "__main__":
    main()
