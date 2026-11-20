"""Optional, hash-checked PyTorch wheel prefetch for this isolated AWQ environment."""
import concurrent.futures
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit
import requests

ROOT = Path('outputs/omni_judge_r71_01')
PACKAGES = [('nvidia-cusparse-cu12', '12.3.1.170'), ('nvidia-nccl-cu12', '2.21.5'),
            ('nvidia-nvjitlink-cu12', '12.4.127'), ('triton', '3.2.0')]


def main():
    folder = ROOT / 'assets/wheels'
    folder.mkdir(parents=True, exist_ok=True)
    receipts = []
    for package, version in PACKAGES:
        index = f'https://download.pytorch.org/whl/cu124/{package}/'
        response = requests.get(index, timeout=(15,30))
        response.raise_for_status()
        links = [urljoin(index, html.unescape(u)) for u in re.findall('href="([^"]+)"', response.text)]
        urls = [u for u in links if f'-{version}-' in u and 'x86_64.whl' in u and ('cp310-cp310' in u if package=='triton' else 'py3-none' in u)]
        assert len(urls) == 1, (package, urls)
        url = urls[0]
        expected = urlsplit(url).fragment.removeprefix('sha256=')
        assert urlsplit(url).hostname in ('download.pytorch.org','download-r2.pytorch.org','pypi.nvidia.com') and len(expected)==64
        published_url = url
        if urlsplit(url).hostname == 'pypi.nvidia.com':
            url = 'https://download.pytorch.org/whl/cu124/' + Path(urlsplit(url).path).name
        target = folder / Path(urlsplit(url).path).name
        if not target.exists():
            with requests.get(url, headers={'Range':'bytes=0-0'}, stream=True, timeout=(15,30)) as response:
                response.raise_for_status()
                assert response.status_code == 206
                size = int(response.headers['Content-Range'].split('/')[-1])
            segments = [(a, min(a+16*1024*1024,size)-1) for a in range(0,size,16*1024*1024)]
            def fetch(bounds):
                a,b = bounds
                p = target.with_name(target.name+f'.r71chunk{a}')
                with requests.get(url,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(15,60)) as response:
                    response.raise_for_status()
                    assert response.status_code==206 and response.headers['Content-Range']==f'bytes {a}-{b}/{size}'
                    with p.open('wb') as stream:
                        for block in response.iter_content(1024*1024): stream.write(block)
                assert p.stat().st_size == b-a+1
                return p
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                chunks=list(pool.map(fetch,segments))
            partial=target.with_name(target.name+'.partial')
            with partial.open('wb') as stream:
                for p in chunks:
                    stream.write(p.read_bytes())
            assert hashlib.sha256(partial.read_bytes()).hexdigest()==expected
            partial.replace(target)
            for p in chunks: p.unlink()
        assert hashlib.sha256(target.read_bytes()).hexdigest()==expected
        receipts.append(dict(package=package,version=version,url=url,published_url=published_url,sha256=expected,bytes=target.stat().st_size))
        print('OFFICIAL_WHEEL_VERIFIED',target.name,flush=True)
    (ROOT/'receipts/official_wheels.json').write_text(json.dumps(receipts,indent=2))


if __name__=='__main__': main()
