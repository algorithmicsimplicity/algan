"""Temporary diagnostic, copied outside the helper checkout before execution."""
from pathlib import Path
import os
p = Path(os.environ['RUNNER_TEMP']) / 'audit_tranche8_validate.py'
exec(compile(p.read_text().split('code = pytest.main')[0], str(p), 'exec'))
from tests.unit_tests.test_sheet_reference_workspace import _inputs, _brute_loss
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from algan.utils.memory_utils import ManualMemory
original_sort = sheets._lexsort
original_search = torch.searchsorted

def sort_spy(*keys, **kw):
    got = original_sort(*keys, **kw)
    expected = torch.arange(keys[0].numel())
    for key in reversed(keys):
        expected = expected[torch.argsort(key.cpu()[expected], stable=True)]
    print('SORT_CHECK', json.dumps({'n': got.numel(), 'mismatch': int((got.cpu() != expected).sum()), 'got': got.cpu()[:16].tolist(), 'expected': expected[:16].tolist()}), flush=True)
    return got

def search_spy(key, query, **kw):
    got = original_search(key, query, **kw)
    expected = original_search(key.cpu(), query.cpu())
    print('SEARCH_CHECK', json.dumps({'n': got.numel(), 'mismatch': int((got.cpu() != expected).sum()), 'got': got.cpu()[:16].tolist(), 'expected': expected[:16].tolist()}), flush=True)
    return got
sheets._lexsort = sort_spy
torch.searchsorted = search_spy
for n in (37, 83):
    host = _inputs(torch.device('cpu'), n)
    gpu = _inputs(render_device(), n)
    print('INPUT_CHECK', json.dumps({'n':n, 'equal': [torch.equal(h,g.cpu()) for h,g in zip(host,gpu)], 'pixel':gpu[0].cpu()[:16].tolist(), 'host_mod':(host[0]%5)[:16].tolist(), 'device_mod':(gpu[0]%5).cpu()[:16].tolist(), 'enf_host':int(host[3].sum()), 'enf_device':int(gpu[3].sum())}), flush=True)
    for constructed_on_cpu in (False, True):
        for force_enforcers in (False, True):
            fields = [x.to(render_device()).clone() for x in (host if constructed_on_cpu else gpu)]
            if force_enforcers:
                fields[3].fill_(True)
            expected = _brute_loss(*[x.cpu() for x in fields])
            device_expected = _brute_loss(*fields).cpu()
            memory = ManualMemory(0,device=render_device(),num_bytes=1<<20)
            ws = CompactionWorkspace(memory)
            result = sheets._sample_depth_lose_reference(*fields, workspace=ws).cpu()
            print('LOSS_CHECK',json.dumps({'n':n,'host_inputs':constructed_on_cpu,'force':force_enforcers,'actual_bad':int((result!=expected).sum()),'oracle_bad':int((device_expected!=expected).sum()),'actual':result.tolist(),'expected':expected.tolist(),'device_oracle':device_expected.tolist()}),flush=True)
