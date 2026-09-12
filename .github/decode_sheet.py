import base64
import gzip
import hashlib
from pathlib import Path

s = Path('.github/sheet_payload.b64').read_text()
for old, new in [
    ('YMiOUuTKrmLgzI5i5bM6OY', 'YMiOUuTKrmLgzI5i5M6OY'),
    ('BaSljb3d3qJ9m', 'BaSljb3qJ9m'),
    ('IIMayTDDAOmp', 'IIMayDAOmp'),
    ('05lmz9K849802Vrd', '05lmz9K802Vrd'),
]:
    s = s.replace(old, new)
p = gzip.decompress(base64.b64decode(s, validate=True))
assert hashlib.sha256(p).hexdigest() == '2c67f67edb0c69364b0ba7c633121f46598c2bedaf497eb85f9b4e2f10f313eb'
Path('/tmp/sheet.patch').write_bytes(p)
print('Verified sheet patch', len(p), flush=True)
