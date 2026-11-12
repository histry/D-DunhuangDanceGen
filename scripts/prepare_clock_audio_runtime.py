"""Provision hash-pinned CPython3.9/Linux audio wheels into a private import root."""
import argparse,hashlib,json,sys,zipfile,urllib.request
from pathlib import Path

PINS={
 'soxr':('0.5.0.post1','9c8e9c980637e03d3f345a4fd81d56477a58c294fb26205fa121bc4eb23d9d01'),
 'librosa':('0.11.0','0b6415c4fd68bff4c29288abe67c6d80b587e0e1e2cfb0aad23e4559504a7fa1'),
 'lazy_loader':('0.4','342aa8e14d543a154047afb4ba8ef17f5563baad3fc610d7b15b213b0f119efc'),
 'msgpack':('1.1.0','8a706d1e74dd3dea05cb54580d9bd8b2880e9264856ce5068027eed09680aa74')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+chr(10))
def verify(receipt):
 d=json.loads(Path(receipt).read_text());vendor=Path(d['import_root']).resolve()
 for r in d.get('wheels',[d['wheel']])+d['files']:
  if sha(r['path'])!=r['sha256']:raise ValueError('Isolated runtime file identity differs')
 sys.path.insert(0,str(vendor));actual={}
 for name,(version,_) in PINS.items():
  module=__import__(name)
  if module.__version__!=version or not Path(module.__file__).resolve().is_relative_to(vendor):
   raise ValueError('Imported package/version is outside the registered runtime')
  actual[name]=dict(version=version,module_sha256=sha(module.__file__))
 return dict(status='PINNED_AUDIO_RUNTIME_VERIFIED',packages=actual,
     shared_python_modified=False,receipt_sha256=sha(receipt))
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify');p.add_argument('--out');p.add_argument('--base-assets');p.add_argument('--assets-out')
 a=p.parse_args()
 if sys.version_info[:2]!=(3,9) or sys.platform!='linux':raise ValueError('Registered wheel ABI requires CPython3.9 Linux')
 if a.verify:
  print(json.dumps(verify(a.verify)));return
 if not all((a.out,a.base_assets,a.assets_out)):p.error('--out --base-assets --assets-out required')
 root=Path(a.out).resolve();assets_out=Path(a.assets_out)
 if root.exists() or assets_out.exists():raise FileExistsError('Use new runtime/assets paths')
 root.mkdir(parents=True);vendor=root/'vendor';vendor.mkdir();wheels=[]
 for name,(version,expected) in PINS.items():
  metadata=json.load(urllib.request.urlopen('https://pypi.org/pypi/'+name+'/'+version+'/json',timeout=30))
  candidates=[r for r in metadata['urls'] if r['digests']['sha256']==expected]
  if len(candidates)!=1:raise ValueError('Official wheel pin unavailable')
  r=candidates[0];data=urllib.request.urlopen(r['url'],timeout=30).read()
  if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Wheel bytes differ from registered pin')
  wheel=root/r['filename'];wheel.write_bytes(data)
  wheels.append(dict(package=name,version=version,path=str(wheel),sha256=expected,official_url=r['url']))
  with zipfile.ZipFile(wheel) as z:
   for entry in z.namelist():
    target=(vendor/entry).resolve()
    if not target.is_relative_to(vendor):raise ValueError('Wheel path escapes isolated root')
    if not entry.endswith('/'):
     target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(entry))
 files=[dict(path=str(p),sha256=sha(p)) for p in vendor.rglob('*') if p.is_file()]
 receipt=root/'receipt.json'
 write(receipt,dict(schema='isolated_audio_runtime_v1',import_root=str(vendor),files=files,wheel=wheels[0],
     wheels=wheels,version=PINS['soxr'][0],librosa_version=PINS['librosa'][0],shared_python_modified=False))
 actual=verify(receipt);assets=json.loads(Path(a.base_assets).read_text())
 assets['audio_runtime']=dict(path=str(receipt),sha256=sha(receipt));write(assets_out,assets)
 print(json.dumps(actual))
if __name__=='__main__':main()
