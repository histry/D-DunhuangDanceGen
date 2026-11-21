"""Read-only post-hoc FGMMC diagnosis. Never fit, expand clocks or change qualification."""
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from dunhuang.alignment.partial_path import exact_path, soft_path_cost
from dunhuang.research.fgmmc_q01.common import ROOT, CFG, read, write, sha, fresh, verify, save_arrays, array_hash
from dunhuang.research.fgmmc_q01.data import Store
from dunhuang.research.fgmmc_q01.network import clocks, canonical_bank, tensors
from dunhuang.research.fgmmc_q01.train import load
from dunhuang.research.fgmmc_q01.evaluate import ranking

SCRIPT = Path(__file__).resolve()

def geometry(T=150, N=16, tolerance=5.):
    original, proof = clocks(T)
    bank = canonical_bank(original)
    centers = (np.linspace(0, T, N + 1)[:-1] + np.linspace(0, T, N + 1)[1:]) / 2 - .5
    mapped = np.stack([np.interp(centers, np.arange(T), v) for v in bank])
    allowed = np.abs(centers[None, :, None] - mapped[:, None, :]) <= tolerance
    order = [next(p for v, p in zip(original, proof) if array_hash(v) == array_hash(clock)) for clock in bank]
    return centers, bank, allowed, order

def structural_audit(T=150, N=16, tolerance=5., offsets=(-15, -8, 0, 8, 15)):
    centers, bank, allowed, order = geometry(T, N, tolerance)
    uniform = next(k for k, r in enumerate(order) if r['amplitude'] == 0)
    rows = []
    for delta in offsets:
        # crop(M,start+delta): a music token's ORIGINAL timestamp is center+delta.
        physical = np.abs(centers[:, None] + delta - centers[None, :]) <= tolerance
        count = lambda a: len(exact_path(-np.ones((N, N)), 0., 0., a)['matches'])
        free = count(physical)
        per_clock = [count(physical & mask) for mask in allowed]
        rows.append(dict(offset_frames=delta, unrestricted_reference_matches=free,
                         uniform_clock_matches=per_clock[uniform], per_clock_matches=per_clock,
                         best_finite_clock_matches=max(per_clock),
                         finite_over_unrestricted=None if not free else max(per_clock) / free,
                         scope='LABEL_AWARE_TIME_GEOMETRY_ORACLE_NOT_MUSIC_OR_MODEL_QUALITY'))
    return dict(token_centers=centers.tolist(), clock_order=order, rows=rows,
                actual_pool_bounds=np.linspace(0, T, N + 1).round().astype(int).tolist(),
                network_masks_same_across_offset_labels=True, original_support_sign='MUSIC_ORIGINAL_TIME=RELATIVE_CENTER+DELTA',
                interpretation='Absorption opportunity only; not actual model preference or causal failure proof')

def per_clock(net, batch):
    inputs = tensors(batch)
    with torch.inference_mode():
        frozen = net(*inputs, constrained=True)
        m, d = net.tokens(*inputs)
        centers, bank, allowed, order = geometry(inputs[0].shape[1], CFG['tokens'], CFG['match_tolerance_frames'])
        B, N, K = len(m), CFG['tokens'], len(bank)
        cost = (1 - m @ d.transpose(-1, -2))
        expanded = cost[:, None].expand(B, K, N, N).reshape(B*K, N, N)
        A = torch.as_tensor(allowed, device=m.device).repeat(B, 1, 1)
        valid = torch.ones((B*K, N), device=m.device, dtype=torch.bool)
        drops = torch.full((B*K, N), CFG['drop_cost'], device=m.device)
        total = soft_path_cost(expanded, drops, drops, A, valid, valid, CFG['path_epsilon'])
        prior = soft_path_cost(torch.ones((K, N, N), device=m.device), drops[:K], drops[:K], A[:K], valid[:K], valid[:K], CFG['path_epsilon'])
        scores = (prior.repeat(B) - total).reshape(B, K) / (2*N)
        aggregate = .1 * (torch.logsumexp(scores/.1, 1) - np.log(K))
        assert torch.equal(aggregate, frozen), 'Diagnostic decomposition changes frozen score'
        posterior = (scores/.1).softmax(1)
    return frozen.cpu().numpy(), scores.cpu().numpy(), posterior.cpu().numpy(), cost.cpu().numpy(), centers, allowed, order

def path_statistics(matches, centers, delta, bounds):
    if delta is None or not matches:
        return dict(physical_mean_gap_frames=None, physical_within5_fraction=None, pooled_support_overlap=None)
    gaps = [abs(centers[i] + delta - centers[j]) for i, j in matches]
    overlap = sum(max(0, min(bounds[i+1]+delta, bounds[j+1]) - max(bounds[i]+delta, bounds[j])) for i, j in matches)
    support = sum(min(bounds[i+1]-bounds[i], bounds[j+1]-bounds[j]) for i, j in matches)
    return dict(physical_mean_gap_frames=float(np.mean(gaps)), physical_within5_fraction=float(np.mean(np.asarray(gaps) <= 5)), pooled_support_overlap=float(overlap/support))

def p4_selections():
    from scipy.spatial.transform import Rotation
    parent = Path('outputs/paper1_role_alignment_01/preview')
    registered = read(parent/'registration.json')['cases']
    ledger = read(parent/'candidates.json')
    decisions = read(ROOT/'controls_view/selections.json')
    results = []
    with np.load(parent/'arrays.npz', allow_pickle=False) as z:
        for case in [c for c in registered if c['requested_nodes'] == 3]:
            own = [r for r in ledger if r['case_key'] == case['key'] and r['G1_pass']]
            baseline = next(r for r in own if 'E1W' in r['labels'])
            gtw = min(own, key=lambda r:(r['GTWLocal'], r['clock_RMS'], r['uid']))
            selections = [('E1W', baseline), ('GTWLocal', gtw)]
            selections += [(d['model'], next(r for r in own if r['uid'] == d['selected_uid'])) for d in decisions if d['case_key'] == case['key']]
            for name, row in selections:
                R, B = z[row['uid']+'__R'], z[baseline['uid']+'__R']
                angle = np.linalg.norm(Rotation.from_matrix((B.swapaxes(-1,-2)@R).reshape(-1,3,3)).as_rotvec(), axis=-1)
                delta = z[row['uid']+'__q'] - z[baseline['uid']+'__q']
                metrics = {k:v for k,v in row['dynamics'].items() if isinstance(v,(float,int,str,bool)) or v is None}
                results.append(dict(source_uid=case['source_uid'], method=name, selected_uid=row['uid'],
                                    E1W_uid=baseline['uid'], GTW_uid=gtw['uid'], equals_E1W=row['uid']==baseline['uid'],
                                    equals_GTW=row['uid']==gtw['uid'], candidate_budget=len(own), G1_pass=row['G1_pass'],
                                    rotation_sha256=row['rotation_sha256'], clock_sha256=row['clock_sha256'], bridge_sha256=row['connection']['bridge_sha256'],
                                    mean_angle_from_E1W_rad=float(angle.mean()), max_clock_difference_from_E1W=float(abs(delta).max()),
                                    RMS_clock_difference_from_E1W=float(np.sqrt(np.mean(delta**2))), GTWLocal=row['GTWLocal'], dynamics=metrics,
                                    artistic_quality='NOT_EVALUABLE'))
    return results

def summarize(saved, reg):
    summaries = []
    for s in ['train','dev','holdout','fine_diagnostic']:
        for name in CFG['models']:
            own = [r for r in saved if r['split']==s and r['model']==name]
            groups = sorted({r['music_group'] for r in own})
            avg = lambda field:float(np.mean([np.mean([r[field]['fractional_top1'] for r in own if r['music_group']==g]) for g in groups]))
            summaries.append(dict(split=s, model=name, windows=len(own), music_groups=len(groups), strict_top1=sum(r['clock']['strict_top1'] for r in own),
                                  group_balanced_clock_top1=avg('clock'), group_balanced_zero_top1=avg('zero_clock'), group_balanced_density_top1=avg('density_clock'),
                                  correct_vs_wrong_music=float(np.mean([r['wrong_music_win'] for r in own])), artistic_quality='NOT_EVALUABLE'))
    assert summaries == read(ROOT/'controls/summary.json')
    probes = []
    for s in ['train','dev','holdout','fine_diagnostic']:
        for name in CFG['models']:
            own = [r for r in saved if r['split']==s and r['model']==name];groups=sorted({r['music_group'] for r in own})
            value = lambda key:float(np.mean([np.mean([r[key]['fractional_top1'] for r in own if r['music_group']==g]) for g in groups]))
            probes.append(dict(split=s,model=name,windows=len(own),music_groups=len(groups),actual=value('clock'),music_zero=value('zero_clock'),motion_feature_reverse=value('reverse_clock'),density=value('density_clock'),position=value('position_clock'),
                               actual_minus_reverse=value('clock')-value('reverse_clock'),correct_vs_wrong_music=float(np.mean([r['wrong_music_win'] for r in own])),
                               actual_true_minus_wrong_song_mean=float(np.mean([r['scores'][2]-r['scores'][5] for r in own]))))
    histories=[]
    for name in CFG['models']:
        h=read(ROOT/name/'history.json');t=read(ROOT/name/'training.json');histories.append(dict(model=name,steps=len(h),first25_loss_mean=float(np.mean([v['loss'] for v in h[:25]])),last25_loss_mean=float(np.mean([v['loss'] for v in h[-25:]])),min_loss=min(v['loss'] for v in h),last_loss=h[-1]['loss'],parameters=t['parameters'],initial_digest=t['initial_digest'],final_digest=t['final_digest'],training_seconds=t['elapsed_seconds'],peak_cuda_bytes=t['peak_cuda_bytes']))
    dependence=[]
    for split in ['train','dev','holdout','fine_diagnostic']:
        own=[s for s in reg['sources'] if s['split']==split];known=lambda k:{s[k] for s in own if s.get(k) not in [None,'UNKNOWN','UNKNOWN_UNVERIFIED']}
        dependence.append(dict(split=split,sources=len(own),music_groups=len(known('group')),known_dancers=len(known('dancer_uid')),known_genres=len(known('genre_uid')),known_choreographies=len(known('choreography_uid')),music_bootstrap_does_not_cover_shared_dancer_choreography=True))
    return summaries, probes, histories, dependence

def run(out, replay_from=None):
    def forbidden(*a, **k):raise RuntimeError('Frozen diagnosis forbids optimizers')
    torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden;torch.optim.SGD=forbidden
    out=fresh(out);parents={p:verify(ROOT/p) for p in ['data']+CFG['models']+['controls','preview','controls_view']}
    saved=read(ROOT/'controls/results.json');reg=read(ROOT/'data/registration.json');windows=read(ROOT/'data/windows.json');byid={r['uid']:r for r in windows};donors=read(ROOT/'data/donors.json');normal=dict(np.load(ROOT/'data/normalizer.npz'));store=Store(reg)
    assert len(saved)==len(windows)*len(CFG['models'])
    for r in saved:
        assert ranking(r['scores'][:5])==r['clock'] and ranking(r['zero_scores'])==r['zero_clock'] and ranking(r['reverse_scores'])==r['reverse_clock']
    summaries, probes, histories, dependence=summarize(saved,reg)
    write(out/'p0_p1.json',dict(registered_windows=len(windows),model_output_rows=len(saved),original_summary_exact=True,summaries=summaries,histories=histories,dependence=dependence))
    structure=structural_audit();write(out/'structure.json',structure);details=[];N=CFG['tokens'];K=len(CFG['clock_amplitudes']);pairs=np.full((len(saved),6,K,N,2),-1,np.int16);sample_counter=0
    for name in CFG['models']:
        net=load(name)
        for r in [q for q in saved if q['model']==name]:
            row=byid[r['uid']];donor=byid[donors[row['uid']]['donor_uid']];batch=[store.crop(row,d,normal=normal) for d in CFG['clock_offsets']]+[store.crop(row,donor=donor,normal=normal)]
            frozen,scores,post,cost,centers,allowed,order=per_clock(net,batch)
            assert np.array_equal(frozen.astype(float),np.asarray(r['scores'])),(name,row['uid'],'Frozen inference changed')
            conditions=[];paths=[]
            for c in range(6):
                by_clock=[];cpaths=[];delta=CFG['clock_offsets'][c] if c<5 else None
                for k in range(K):
                    decoded=exact_path(cost[c],CFG['drop_cost'],CFG['drop_cost'],allowed[k]);matches=decoded['matches'];pairs[sample_counter,c,k,:len(matches)]=np.asarray(matches,dtype=np.int16).reshape(-1,2)
                    by_clock.append(dict(clock_amplitude=order[k]['amplitude'],clock_sha256=order[k]['clock_sha256'],soft_score=float(scores[c,k]),contribution_weight=float(post[c,k]),hard_matches=len(matches),hard_music_drops=len(decoded['music_drops']),hard_motion_drops=len(decoded['motion_drops']),hard_cost=decoded['cost'],**path_statistics(matches,centers,delta,structure['actual_pool_bounds'])))
                    cpaths.append(matches)
                paths.append(cpaths);conditions.append(dict(offset_frames=delta,condition='WRONG_SONG' if delta is None else 'RECORDED' if delta==0 else 'NONCIRCULAR_SHIFT',aggregate_score=float(frozen[c]),per_clock=by_clock,dominant_contribution_clock_amplitude=order[int(np.argmax(scores[c]))]['amplitude'],dominant_is_diagnostic_not_model_selected_warp=True))
            for c in range(5):
                delta=CFG['clock_offsets'][c]
                for k in range(K):
                    original={tuple(v) for v in paths[2][k]};converted=set()
                    for i,j in paths[c][k]:
                        v=centers[i]+delta;nearest=int(np.argmin(abs(centers-v)))
                        if abs(centers[nearest]-v)<=CFG['match_tolerance_frames']:converted.add((nearest,j))
                    union=original|converted;conditions[c]['per_clock'][k]['common_record_clock_node_jaccard']=None if not union else len(original&converted)/len(union)
            details.append(dict(uid=row['uid'],source_uid=row['source_uid'],split=row['split'],music_group=row['group'],model=name,conditions=conditions,positive_minus_shift_margins=[float(frozen[2]-frozen[c]) for c in [0,1,3,4]],positive_minus_wrong_song=float(frozen[2]-frozen[5]),best_clock_positive_minus_shift=[float(scores[2].max()-scores[c].max()) for c in [0,1,3,4]],score_reproduction_exact=True))
            sample_counter+=1
        print('FG_DIAG_MODEL',name,sample_counter,flush=True)
    write(out/'clock_breakdown.json',details);save_arrays(out/'hard_paths.npz',dict(matches=pairs));write(out/'p3.json',probes);write(out/'p4.json',p4_selections())
    summary=[]
    for split in ['train','dev','holdout','fine_diagnostic']:
        for name in CFG['models']:
            own=[r for r in details if r['split']==split and r['model']==name]
            for c,delta in enumerate(CFG['clock_offsets']):
                selected=[r['conditions'][c]['per_clock'][int(np.argmax([v['contribution_weight'] for v in r['conditions'][c]['per_clock']]))] for r in own]
                good=[v for v in selected if v['physical_mean_gap_frames'] is not None]
                summary.append(dict(split=split,model=name,offset_frames=delta,windows=len(own),mean_true_minus_condition=float(np.mean([r['conditions'][2]['aggregate_score']-r['conditions'][c]['aggregate_score'] for r in own])),dominant_nonuniform_fraction=float(np.mean([v['clock_amplitude']!=0 for v in selected])),mean_hard_matches=float(np.mean([v['hard_matches'] for v in selected])),mean_physical_gap_frames=None if not good else float(np.mean([v['physical_mean_gap_frames'] for v in good])),mean_pooled_support_overlap=None if not good else float(np.mean([v['pooled_support_overlap'] for v in good])),semantic_quality='NOT_EVALUABLE'))
    write(out/'path_summary.json',summary)
    write(out/'qualification.json',dict(status='FROZEN_POSTHOC_DIAGNOSIS_COMPLETE',original_gate=read(ROOT/'controls/qualification.json'),optimizer_calls=0,new_clocks=0,new_windows=0,new_geometry_calls=0,new_planner_runs=0,causal_failure_identification='NOT_ESTABLISHED_BY_ASSOCIATION',semantic_or_artistic_truth='NOT_EVALUABLE',can_route=False))
    manifest=dict(script_sha256=sha(SCRIPT),parents={k:sha(ROOT/k/'manifest.json') for k in parents},files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()});write(out/'manifest.json',manifest)
    if replay_from:
        previous=read(Path(replay_from)/'manifest.json');assert manifest==previous,'Diagnosis replay differs'
        write(ROOT/'receipts/diagnosis_replay.json',dict(status='PASS',fresh_process=True,all_scalar_files_exact=True,all_hard_paths_exact=True,script_sha256=sha(SCRIPT),optimizer_calls=0,retraining=False,original_qualification_unchanged=True,can_route=False))
    print('FG_DIAG_PASS',len(details),sample_counter*6*K,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'diagnosis'));p.add_argument('--replay-from');a=p.parse_args();run(Path(a.out),a.replay_from)
