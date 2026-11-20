"""Publish only R69 scalar science and receipts; private blind keys stay private."""
import argparse,hashlib,importlib.util,json,subprocess
from pathlib import Path
from dunhuang.research.retiming_baselines_r69.common import read,write,sha

def export(root,out,execution,parent_root='outputs/sdmr_warp_rank_01/main'):
 root=Path(root);out=Path(out);execution=Path(execution)
 if out.exists():raise ValueError('Fresh R69 scalar publication required')
 spec=importlib.util.spec_from_file_location('r68_scalar_projection',Path(__file__).with_name('release_r68.py'))
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 verified=old.hash_manifest(root)
 required={'population.json','records.json','registration.json','summary.json','quality_state.json'}
 if not required.issubset({row['path'] for row in verified['files']}):raise ValueError('Every published science table must be inside the original seal')
 out.mkdir(parents=True)
 files={name:old.project(read(root/name)) for name in ('population.json','records.json','registration.json','summary.json','quality_state.json')}
 replay=root.parent/'replay/independent.json'
 files['independent.json']=old.replay(replay,root/'manifest.json')
 files['scientific_files.json']=dict(verified=verified,private_admin_bytes_copied=False,raw_motion_bytes_copied=False)
 # Three reuse arms share one historical bank; their row-level counts are
 # parent-bank request annotations, never additive per-method paid costs.
 parent=Path(parent_root)
 if (parent/'manifest.json').is_file():
  registered=read(root/'registration.json')['parent_manifests']['r67']
  if sha(parent/'manifest.json')!=registered:raise ValueError('Historical cost source manifest changed')
  historical=old.hash_manifest(parent)
  spec=importlib.util.spec_from_file_location('r69_historical_costs',Path(__file__).with_name('historical_costs_r69.py'))
  helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
  ledger=helper.shared_historical_costs(read(parent/'banks.json'),read(parent/'records.json'),read(parent/'registration.json'))
  ledger.update(source_manifest_sha256=registered,source_files_verified=historical['status'],helper_sha256=sha(Path(__file__).with_name('historical_costs_r69.py')))
  files['shared_historical_bank_costs.json']=ledger
 else:
  files['shared_historical_bank_costs.json']=dict(status='NOT_AVAILABLE_NOT_ZERO_FILLED',scope='CURRENT_PUBLICATION_FIXTURE_OR_MISSING_PARENT')
 files['qualification.json']=dict(independent=files['independent.json']['status'],human_reviewed=False,valid_independent_reviews=0,
  artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE',can_route=False,quality_winner=None,new_model_training_steps=0,new_planner_searches=0)
 if (root/'runtime.json').is_file():
  files['runtime.json']=old.project(read(root/'runtime.json'))
  files['runtime.json']['scope']='WALLTIME_NOT_DETERMINISTIC_NOT_EQUAL_COMPUTE; HISTORICAL_COST_UNKNOWN_NOT_ZERO'
 receipts=[]
 for folder,prefix in ((execution,'execution'),(root.parent/'receipts','delivery')):
  if folder.is_dir():
   for p in sorted(folder.glob('*.json')):receipts.append(dict(path=prefix+'/'+p.name,sha256=sha(p),metadata=old.project(read(p))))
 files['execution_and_delivery_receipts.json']=dict(records=receipts)
 files['publication_inputs.json']=dict(source_manifest_sha256=sha(root/'manifest.json'),publisher_sha256=sha(__file__),
  projection_sha256=sha(Path(__file__).with_name('release_r68.py')),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 for name,value in files.items():write(out/name,value)
 write(out/'public_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()],
  excludes='NO_VIEWER_COORDINATES_AUDIO_ROTATIONS_PRIVATE_ADMIN_MAPPING_SEEDS_OR_REVIEWER_IDENTITIES'))
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='outputs/retiming_baselines_r69_01/main');p.add_argument('--out',default='reports/r69_release');p.add_argument('--execution',default='outputs/retiming_baselines_r69_execution_01');a=p.parse_args();print(export(a.root,a.out,a.execution))
if __name__=='__main__':main()
