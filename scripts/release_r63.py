"""Export small scientific metadata only; viewing media use a separate packet."""
from pathlib import Path
from dunhuang.research.budget_r63.common import ROOT,read,write,sha,binding

def main():
 out=Path('reports/r63_release')
 if out.exists():raise ValueError('Fresh release directory required')
 out.mkdir(parents=True)
 results={mode:read(ROOT/mode/'result.json') for mode in ('main','controls')}
 verification=read(ROOT/'independent/result.json')
 for mode,r in results.items():
  records=[]
  for x in r['records']:
   records.append({k:x[k] for k in ('key','case_key','gap_frames','allocation','status','music_interval','methods')})
  write(out/(mode+'_summary.json'),r['summary'])
  write(out/(mode+'_selections.json'),r['selections'])
  write(out/(mode+'_metrics.json'),records)
 write(out/'protocol.json',results['main']['protocol'])
 write(out/'duration_policy.json',read(ROOT/'main/duration_policy.json'))
 write(out/'evaluation_predictions.json',read(ROOT/'main/evaluation_predictions.json'))
 write(out/'independent.json',verification)
 rows=[]
 for mode in ('main','controls','independent'):
  for p in sorted((ROOT/mode).rglob('*')):
   if p.is_file():rows.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
 write(out/'scientific_files.json',dict(files=rows))
 receipts=[]
 for p in sorted((ROOT/'receipts').glob('*.json')):
  v=read(p);write(out/('receipt_'+p.name),v);receipts.append(p.name)
 write(out/'public_manifest.json',dict(binding=binding(),
  results={mode:dict(sha256=sha(ROOT/mode/'result.json'),commit=r['binding']['commit']) for mode,r in results.items()},
  policy_sha256=sha(ROOT/'main/duration_policy.json'),scientific_files_sha256=sha(out/'scientific_files.json'),
  verification_sha256=sha(ROOT/'independent/result.json'),receipts=receipts,
  coordinates_in_public_release=False,audio_in_public_release=False,weights_in_public_release=False,
  can_route=False,Native24='NOT_EVALUABLE'))
 print('R63_PUBLIC_METADATA_READY',flush=True)

if __name__=='__main__':main()
