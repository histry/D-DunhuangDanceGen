"""EDGE official PFC definition replay on all saved executions, no disk fixtures."""
import argparse,contextlib,importlib.util,io,json,sys,types
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,pinned,sha,write
from dunhuang.contracts import Q
from dunhuang.evaluation_benchmarks.edge_pfc import pfc,SOURCE


def read(path):return json.loads(Path(path).read_text())


def official(module,positions):
    # Execute the unmodified pinned function with an in-memory full_pose fixture.
    z=positions@Q.T
    module.glob=types.SimpleNamespace(glob=lambda pattern:['controlled_fixture.pkl'])
    module.open=lambda *args,**kwargs:io.BytesIO(b'controlled_fixture')
    module.pickle=types.SimpleNamespace(load=lambda stream:dict(full_pose=z.copy()))
    module.tqdm=lambda rows:rows
    output=io.StringIO()
    with contextlib.redirect_stdout(output):module.calc_physical_score('controlled_fixture')
    return float(output.getvalue().strip().rsplit(' ',1)[-1])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--experiment',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    experiment=Path(a.experiment);protocol=read(experiment/'protocol.json')
    assets=read(checked(protocol['generation_assets']));edge=pinned(assets['edge_root'],assets['edge_commit'])
    source=edge/'eval/eval_pfc.py';spec=importlib.util.spec_from_file_location('frozen_official_pfc',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=[]
    for mode,folder in [('legacy_paired300',Path(a.baseline)),('cross',experiment/'cross/mechanism'),('additive',experiment/'additive/mechanism')]:
        for c in read(folder/'generation.json')['cases']:
            cases.append((mode,'aist_factorial',c['case'],c['event_index']==-1,checked(c['fk']),c['motion']['sha256']))
    for mode in ('cross','additive'):
        for case in ('original','replaced'):
            folder=experiment/mode/('dunhuang_'+case)
            cases.append((mode,'dunhuang_unpaired',case,False,folder/'native_fk.npy',sha(folder/'native.npy')))
    records=[];groups=defaultdict(list)
    for mode,domain,label,zero,path,motion_sha in cases:
        fk=np.load(path,allow_pickle=False);result=pfc(fk,up_axis='y')
        score=official(module,fk)
        if result['value'] is None:
            if not np.isnan(score):raise ValueError('Official undefined PFC behavior changed')
        elif result['value']!=score:
            raise ValueError('Official PFC mismatch: '+label)
        row=dict(mode=mode,domain=domain,case=label,zero_adapter=zero,motion_sha256=motion_sha,
                 fk=dict(path=str(path.resolve()),sha256=sha(path)),pfc=result,official_value=None if np.isnan(score) else score,
                 exact_official_replay=True)
        records.append(row)
        if result['value'] is not None:groups[(mode,domain,zero)].append(result['value'])
    summary=[dict(mode=k[0],domain=k[1],zero_adapter=k[2],defined_cases=len(v),mean=float(np.mean(v)),
                  min=min(v),max=max(v)) for k,v in groups.items()]
    write(out,dict(status='ALL_SAVED_PFC_MATCH_OFFICIAL',cases=len(records),records=records,summary=summary,
        official_commit=assets['edge_commit'],official_file_sha256=sha(source),source=SOURCE,
        optimizer_calls=0,new_sampling_calls=0,motion_updates=0,
        scope='official PFC only, small registered execution pool; not Native admission, human quality or paper ranking'))
    print('PFC_OFFICIAL_EXACT',len(records),summary,flush=True)

if __name__=='__main__':main()
