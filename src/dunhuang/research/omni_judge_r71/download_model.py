"""Download official Qwen AWQ files from its author-published ModelScope mirror.

Every file is pinned by the repository's per-file revision and SHA256 before
download. No evaluation media is uploaded. Partial downloads are never admitted.
"""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path('outputs/omni_judge_r71_01')
API = 'https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-Omni-7B-AWQ'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    receipt = ROOT / 'receipts'
    receipt.mkdir(parents=True, exist_ok=True)
    lock = receipt / 'modelscope_lock.json'
    if lock.exists():
        files = json.loads(lock.read_text())
    else:
        response = requests.get(API + '/repo/files', params={'Revision': 'master', 'Recursive': 'true'}, timeout=30)
        response.raise_for_status()
        files = [dict(path=f['Path'], size=f['Size'], revision=f['Revision'], sha256=f['Sha256'])
                 for f in response.json()['Data']['Files'] if f['Type'] == 'blob']
        assert files and all(len(f['sha256']) == 64 and f['revision'] for f in files)
        lock.write_text(json.dumps(files, sort_keys=True, indent=2))
    registration = dict(model_id='Qwen/Qwen2.5-Omni-7B-AWQ',
                        source='author_published_modelscope_mirror',
                        revision='file_manifest_sha256:' + digest(lock), files=files)
    (receipt / 'model_registration.json').write_text(json.dumps(registration, indent=2))
    model = ROOT / 'assets/model'
    model.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        path = model / item['path']
        assert path.resolve().is_relative_to(model.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size == item['size'] and digest(path) == item['sha256']:
            return
        partial = path.with_name(path.name + '.partial')
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                with requests.get(API + '/repo', params={'Revision': item['revision'], 'FilePath': item['path']},
                                  headers={'Range': f'bytes={offset}-'} if offset else {},
                                  stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    append = offset and response.status_code == 206
                    with partial.open('ab' if append else 'wb') as stream:
                        for block in response.iter_content(8 * 1024 * 1024):
                            stream.write(block)
                if partial.stat().st_size != item['size'] or digest(partial) != item['sha256']:
                    partial.unlink()
                    raise ValueError('Official size/SHA256 mismatch: ' + item['path'])
                partial.replace(path)
                print('MODEL_FILE_VERIFIED', item['path'], item['size'], flush=True)
                return
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, files))
    # A second complete hash pass is deliberate admission, after download workers exit.
    assert all((model / f['path']).stat().st_size == f['size'] and digest(model / f['path']) == f['sha256'] for f in files)
    (receipt / 'model_files.json').write_text(json.dumps(files, indent=2))
    print('R71_MODEL_READY', len(files), flush=True)


if __name__ == '__main__':
    main()
