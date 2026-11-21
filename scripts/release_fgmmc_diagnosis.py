"""Scalar-only frozen diagnosis export, including effective graph multiplicity audit."""
import importlib.util, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.research.fgmmc_q01.common import ROOT, CFG, read, write, sha, fresh, array_hash

def build(out):
    source=ROOT/'diagnosis';m=read(source/'manifest.json')
    assert all(sha(source/n)==v for n,v in m['files'].items())
    replay=read(ROOT/'receipts/diagnosis_replay.json');assert replay['status']=='PASS'
    path=Path('scripts/diagnose_fgmmc.py');assert sha(path)==m['script_sha256']
    spec=importlib.util.spec_from_file_location('frozen_fgmmc_diagnosis',path);diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
    centers,bank,masks,order=diag.geometry();classes={}
    for k,mask in enumerate(masks):classes.setdefault(array_hash(mask),[]).append(k)
    assert len(bank)==5 and len(classes)==3
    detail=read(source/'clock_breakdown.json');checked=0
    for row in detail:
        for condition in row['conditions']:
            for ids in classes.values():
                values=[condition['per_clock'][k]['soft_score'] for k in ids]
                assert len(set(values))==1,'Same graph gave different score'
            checked+=1
    from dunhuang.research.fgmmc_q01.network import initial
    template=initial();radius=sum((b.conv.kernel_size[0]-1)//2*b.conv.dilation[0] for b in template.music.blocks)
    assert radius==3
    bounds=np.linspace(0,CFG['frames'],CFG['tokens']+1).round().astype(int);support=[]
    for delta in CFG['clock_offsets']:
        nominal=[];halo=[]
        for a,b in zip(bounds[:-1],bounds[1:]):
            nominal.append(max(0,min(b+delta,b)-max(a+delta,a))/(b-a))
            lo,hi=max(0,a-radius),min(CFG['frames'],b+radius)
            halo.append(max(0,min(hi+delta,hi)-max(lo+delta,lo))/(hi-lo))
        support.append(dict(offset_frames=delta,mean_same_index_nominal_bin_overlap=float(np.mean(nominal)),mean_same_index_TCN_receptive_upper_support_overlap=float(np.mean(halo))))
    tie_summary=[]
    for split in ['train','dev','holdout','fine_diagnostic']:
        for model in CFG['models']:
            rows=[r for r in detail if r['split']==split and r['model']==model]
            for c,delta in enumerate(CFG['clock_offsets']):
                maxima=[]
                for row in rows:
                    values=row['conditions'][c]['per_clock'];best=max(v['soft_score'] for v in values)
                    maxima.append([v['clock_amplitude'] for v in values if abs(v['soft_score']-best)<=1e-7])
                tie_summary.append(dict(split=split,model=model,offset_frames=delta,windows=len(rows),uniform_among_maximal_contributors=sum(0 in v for v in maxima),only_nonuniform_maximal_contributors=sum(0 not in v for v in maxima),multiway_maxima=sum(len(v)>1 for v in maxima)))
    graph=dict(unique_clocks=len(bank),unique_allowed_graphs=len(classes),
               classes=[dict(mask_sha256=h,clock_indices=ids,amplitudes=[order[k]['amplitude'] for k in ids],nominal_aggregation_prior_mass=len(ids)/len(bank)) for h,ids in classes.items()],
               all_equal_graph_scores_exact=True,checked_model_conditions=checked,
               TCN_receptive_radius_frames=radius,identity_path_support_overlap=support,
               support_scope='NOMINAL_BINS_AND_MASK_BLIND_COMPUTATIONAL_SUPPORT_BOUND_NOT_SALIENCY_OR_SEMANTIC_EQUALITY; MODEL_PATH_OVERLAP_METRIC_EXCLUDES_EXTRA_TCN_HALO',
               contribution_max_tie_summary=tie_summary,
               argmax_tie_warning='Original path_summary dominant_nonuniform_fraction uses first argmax; canonical order puts +4 before 0 in an identical graph class. It cannot establish nonuniform preference. Aggregate logmeanexp scores are unchanged.',
               no_reweighted_scores_or_new_rankings=True,
               interpretation='Exact-clock deduplication does not deduplicate effective matching graphs; this is representation degeneracy, not proof of the full failure cause')
    out=fresh(out)
    for name in ['p0_p1','structure','clock_breakdown','path_summary','p3','p4','qualification','manifest']:write(out/('source_'+name+'.json'),read(source/(name+'.json')))
    write(out/'graph_equivalence.json',graph);write(out/'diagnosis_replay.json',replay)
    launch=Path('outputs/fgmmc_launch');exits={n:int((launch/(n+'.exit')).read_text()) for n in ['diagnosis','diagnosis_replay']};assert all(v==0 for v in exits.values())
    write(out/'runtime.json',dict(status='PASS',diagnosis_script_sha256=sha(path),export_script_sha256=sha(Path(__file__)),original_fgmmc_public_manifest_sha256=sha('reports/fgmmc_q01_release/manifest.json'),exit_status=exits,diagnosis_contract_tests=6,model_rows=len(detail),registered_clock_conditions_per_model_row=6,registered_clocks=5,decoded_hard_paths=len(detail)*6*5,new_optimizer_updates=0,new_windows=0,new_clocks=0,new_geometry_calls=0,new_planner_runs=0,qualification_unchanged=True,hard_path_arrays_kept_private_sha256=sha(source/'hard_paths.npz'),model_weight_sha256={n:sha(ROOT/n/'weights.npz') for n in CFG['models']},causal_conclusion='NOT_ESTABLISHED',can_route=False))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}));print('FG_DIAG_PUBLIC',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()))

if __name__=='__main__':build('reports/fgmmc_q01_diagnosis')
