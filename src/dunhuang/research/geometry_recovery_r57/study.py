from pathlib import Path
from types import SimpleNamespace
import argparse,subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.bridge_audit_r46.admission import certificate
from dunhuang.research.manifold_bridge_r47.geometry import segmented
from dunhuang.research.bridge_dynamics_r48.study import metrics
from dunhuang.research.tempo_attribution_r57.common import ROOT,PARENT,fresh,save_arrays
from dunhuang.research.tempo_attribution_r57.study import Study

def bridge(left,right,frames,net,reference):
 observed=np.full((frames+6,22,3,3),np.nan);observed[:3]=left[-3:];observed[-3:]=right[:3];gap=(3,3+frames);cert=certificate(observed,gap);options={};error=None
 if cert['chart_eligible']:
  try:options['G0']=observed_bridge(observed,gap)
  except ValueError as exc:
   if str(exc)!='Ambiguous SO3 chart':raise
   error=str(exc)
 # A failure of this G0 chart is not evidence that G1 is impossible.
 q,guard=segmented(observed,gap,(4,8,16),np.pi-.05)
 if q is not None:options['G1']=q
 records={}
 for name,Q in options.items():
  assert np.array_equal(Q[:3],observed[:3]) and np.array_equal(Q[-3:],observed[-3:]);full=np.concatenate((left[-4:-3],Q,right[3:4]));records[name]=metrics(full,net.joints,net.parents,(4,4+frames))
 eligible=[name for name,m in records.items() if m['gap_D2']<=reference['D2'] and m['gap_D3']<=reference['D3'] and m['shoulder_wrist_min_m']>=reference['shoulder_wrist_lower_m']];selected=min(eligible,key=lambda name:(records[name]['gap_D3'],name)) if eligible else None
 return (options[selected][3:-3] if selected else None),dict(selected=selected,methods=records,G0=cert,G0_error=error,G1=guard,frames=frames,physical_qualified=False,can_route=False)

def run(out):
 out=fresh(out);source=ROOT/'controls/result.json';truth=read(source)['G1_diagnostic'];assert truth and all(r['passes_original_screen'] for r in truth);s=Study();old_sha=sha(PARENT/'controls/result.json');arrays={};rows=[];prior=np.load(ROOT/'controls/arrays.npz')
 for k,row in enumerate(truth):
  g=s.parent['graphs'][row['graph_key']];i,j,slot=row['left'],row['right'],row['slot'];cores=[r['interval'] for r in g['spans'] if r['kind']=='core'];L=s.sample(g['pools'][slot][i],cores[slot])['R'];R=s.sample(g['pools'][slot+1][j],cores[slot+1])['R'];Q,e=bridge(L,R,48,s.pop.net,s.pop.reference);assert e['G0_error']=='Ambiguous SO3 chart' and e['selected']=='G1';assert e['methods']['G1']==row['metrics'];assert np.array_equal(Q,prior[f'edge{k}'][3:-3]);arrays[f'edge{k}']=Q;rows.append(dict(graph_key=row['graph_key'],slot=slot,left=i,right=j,evidence=e))
 assert sha(PARENT/'controls/result.json')==old_sha;save_arrays(out/'arrays.npz',arrays);result=dict(records=rows,diagnostic_sha256=sha(source),R56_unchanged_sha256=old_sha,code_sha256=sha(__file__),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),newly_retained_edges=len(rows),original_thresholds_unchanged=True,production_changed=False,physical_qualified=False,can_route=False);write(out/'result.json',result);return result
def verify(out,source):
 out=fresh(out);before=read(Path(source)/'result.json');actual=run(out/'rebuild');assert actual==before;assert sha(Path(source)/'arrays.npz')==sha(out/'rebuild/arrays.npz');write(out/'result.json',dict(result_exact=True,arrays_bytes_exact=True,optimizer_calls=0,source_sha256=sha(Path(source)/'result.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','verify']);p.add_argument('--out',required=True);p.add_argument('--source');a=p.parse_args();verify(a.out,a.source) if a.mode=='verify' else run(a.out)
