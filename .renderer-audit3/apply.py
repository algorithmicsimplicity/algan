"""Apply the locally verified text deltas on an exact, independently checked tree."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

payload = Path(sys.argv[1])
root = Path.cwd()

def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def apply_delta(data, expected_digest):
    assert hashlib.sha256(data).hexdigest() == expected_digest, "transport digest"
    edits = json.loads(gzip.decompress(data))
    for name, change in edits.items():
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts, name
        assert name.startswith(("algan/", "tests/unit_tests/", "benchmarks/", "reports/")), name
        target = root / path
        if change["before"] is None:
            assert not target.exists(), name
            old = []
        else:
            original = target.read_bytes()
            assert blob(original) == change["before"], (name, "base blob")
            old = original.decode("utf-8").splitlines(keepends=True)
        result = []
        for edit in change["edits"]:
            if isinstance(edit, str):
                result.append(edit)
            else:
                start, stop, indent = edit
                assert 0 <= start <= stop <= len(old), name
                for line in old[start:stop]:
                    if indent < 0:
                        assert line.startswith(" " * -indent), name
                        result.append(line[-indent:])
                    else:
                        result.append(" " * indent + line)
        final = "".join(result).encode("utf-8")
        assert blob(final) == change["after"], (name, "final blob")
        if name.endswith(".py"):
            ast.parse(final, filename=name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(final)
        subprocess.run(["git", "add", "--", name], check=True)
    return set(edits)

code = b"".join((payload / f"part-{i:02d}").read_bytes() for i in range(17))
paths = apply_delta(code, "fef77b4b6b9d1752d96fda6a31b6ddd5613c825909812eaed0d59f7185bb61c2")
paths.update(apply_delta((payload / "curation.gz").read_bytes(), "a76418132b19cd071396002c35aecab1941bbeb0e23a7c5d350fb708566e2fc7"))
status = b"".join(p.read_bytes() for p in sorted(payload.glob("status-*")))
paths.update(apply_delta(status, sys.argv[2]))
changed = set(subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines())
assert paths == changed and len(paths) == 22, (paths, changed)
print("Validated exact before/after blobs for all 22 changed paths.")
