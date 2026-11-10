import hashlib
import json
import subprocess
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def checked(ref):
    path = Path(ref['path']).resolve()
    if not path.is_file() or sha(path) != ref['sha256']:
        raise ValueError('Asset hash mismatch: ' + str(path))
    return path

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def pinned(root, commit):
    root = Path(root).resolve()
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise ValueError('Dependency commit mismatch: ' + str(root))
    if subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--'], text=True):
        raise ValueError('Tracked dependency modifications: ' + str(root))
    return root
