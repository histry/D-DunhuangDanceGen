"""Small IO contracts; no mutation of any frozen parent output."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path('outputs/quality_r65_01')
PARENT = Path('outputs/timing_r64_01')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fresh(path):
    path = Path(path)
    if path.exists():
        raise ValueError('Fresh R65 output required: '+str(path))
    path.mkdir(parents=True)
    return path


def binding():
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                code={p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.json', '.html')})


CFG = read(HERE/'protocol.json')
