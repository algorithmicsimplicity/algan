"""One-off validation transport; never included in the implementation tree."""
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys

base = 'a222eca1468990b459528641c3a336a23b888f51'
root = Path.cwd().resolve()
encoded = ''.join((root / '.github' / f'audit_tranche8_code{i}.b64').read_text().strip() for i in range(7))
patch = lzma.decompress(base64.b64decode(encoded, validate=True))
fix = lzma.decompress(base64.b64decode((root / '.github/audit_tranche8_fix.b64').read_text().strip(), validate=True))
digests = [hashlib.sha256(x).hexdigest() for x in (patch, fix)]
assert digests == ['7ddea58fc44ef8d488a43ca2ed470fd3185fd513eed95cb7d33499bfbc57a3ef', '63edc19b4b8280e103ed88145f2ed60e0be97e84c4b15e5e3fdbfb97d3be870c'], digests
subprocess.run(['git', 'fetch', '--depth=1', 'origin', base], check=True)
subprocess.run(['git', 'checkout', '--detach', base], check=True)
for data in (patch, fix):
    subprocess.run(['git', 'apply', '--check', '-'], input=data, check=True)
    subprocess.run(['git', 'apply', '-'], input=data, check=True)
subprocess.run(['git', 'diff', '--check'], check=True)
sys.path.insert(0, str(root))
import importlib.metadata as md
import torch
import algan
import pytest
from algan.rendering.taichi_runtime import init_taichi, _live_arch
from algan.settings._startup import render_device
assert Path(algan.__file__).resolve() == root / 'algan/__init__.py'
requested = os.environ['ALGAN_RENDER_DEVICE']
assert render_device().type == requested, (requested, render_device())
init_taichi()
arch = str(_live_arch())
if requested == 'mps':
    assert 'metal' in arch, arch
print('AUDIT_TRANCHE8_METADATA', json.dumps({'base': base, 'patch_sha256': digests, 'source': algan.__file__, 'torch': torch.__version__, 'compiler_distribution': md.version('algan-quadrants'), 'device': str(render_device()), 'arch': arch}), flush=True)
code = pytest.main(['-q', '-ra', '--tb=short', '--junitxml=tranche8-validation.xml', 'tests/unit_tests/test_array_gather_destinations.py', 'tests/unit_tests/test_sheet_geometry_workspace.py', 'tests/unit_tests/test_sheet_shell_workspace.py', 'tests/unit_tests/test_sheet_reference_workspace.py'])
print('AUDIT_TRANCHE8_RESULT', json.dumps({'pytest_exit': int(code), 'device': str(render_device()), 'arch': arch}), flush=True)
sys.exit(code)
