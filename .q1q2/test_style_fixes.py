from pathlib import Path
import ast

p = Path('algan/rendering/raytracing/shadow_dispatch.py')
s = p.read_text()
old = '''    inputs = dict(accepted=accepted, source=source, pos=pos, snrm=snrm, fnrm=fnrm,
                  frame=frame, mask=mask, dp=dp, toff=toff)'''
new = '''    inputs = {
        "accepted": accepted, "source": source, "pos": pos, "snrm": snrm,
        "fnrm": fnrm, "frame": frame, "mask": mask, "dp": dp, "toff": toff,
    }'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))

for path in ('tests/unit_tests/test_arena_regions.py', 'tests/unit_tests/test_device_dispatch_taichi.py'):
    p = Path(path)
    s = p.read_text()
    for names in ('dtype,tag', 'footprint,terminator'):
        s = s.replace('"' + names + '"', repr(tuple(names.split(','))))
    lines = []
    for line in s.splitlines(keepends=True):
        if (line.lstrip().startswith('assert ') and ' and ' in line
                and isinstance(ast.parse(line.strip()).body[0].test, ast.BoolOp)
                and isinstance(ast.parse(line.strip()).body[0].test.op, ast.And)):
            prefix, condition = line.split('assert ', 1)
            lines.extend(prefix + 'assert ' + predicate.rstrip() + '\n'
                         for predicate in condition.split(' and '))
        else:
            lines.append(line)
    p.write_text(''.join(lines))
