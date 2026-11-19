"""Construct pairs before evaluating any warp or bridge outcome."""
from .common import *
from dunhuang.research.paired_event_r41.events import profiles, phases
from dunhuang.research.event_coverage_r47.index import segment_source, merge_protected

def protected_spans(evidence, n, config):
    return [p['evidence_span'] for p in evidence if p['type'] in ('pose_hold','root_turn_span')] + [
        [max(0,p['frame']-config['key_protection_frames']), min(n,p['frame']+config['key_protection_frames']+1)]
        for p in evidence if p['type']=='speed_peak']

def match_pair(unit, speed, protected, config=CFG):
    a,b = proof_interval(unit['source_interval'],unit['support_component']); lo,hi = unit['support_component']
    if crossings([a,b],protected): return dict(status='NOT_MATCHABLE', reason='A4_ALREADY_CUTS_PROTECTED_SUPPORT')
    context=config['context_frames'];gap=config['hidden_gap'];half=gap//2
    if a-lo<context or hi-b<context:
        return dict(status='NOT_MATCHABLE',reason='A4_LACKS_FULL_COMMON_REAL_CONTEXT')
    native = float(np.mean(speed[a:b])); options=[]
    for shift in range(-config['pair_max_shift'],config['pair_max_shift']+1):
        A,B=a+shift,b+shift
        if not shift or A-lo<context or hi-B<context: continue
        overlap=max(0,min(b,B)-max(a,A))/(b-a)
        cross=crossings([A,B],protected)
        if overlap<config['pair_min_overlap'] or not cross: continue
        common=[name for name,x,y in [('left',a,A),('right',b,B)]
                if any(start<y<end for start,end in protected)
                and all(lo<=cut-half-4 and cut+gap-half+4<=hi for cut in (x,y))]
        if not common:continue
        value=float(np.mean(speed[A:B])); difference=abs(native-value)/max(native,value,config['pair_activity_floor_radps'])
        if difference<=config['pair_activity_relative_tolerance']:
            options.append((difference,abs(shift),shift,value,cross,common))
    if not options: return dict(status='NOT_MATCHABLE',reason='NO_ACTIVITY_MATCHED_PROTECTED_CUT_WITH_COMMON_CONTEXT_AND_INTERVENED_HIDDEN_SEAM')
    difference,_,shift,value,cross,common=min(options,key=lambda x:x[:3])
    A,B=a+shift,b+shift
    return dict(status='MATCHED',A4=[a,b],A5=[A,B],length=b-a,shift=shift,
                overlap_fraction=max(0,min(b,B)-max(a,A))/(b-a),
                activity_A4=native,activity_A5=value,relative_activity_difference=difference,
                A4_crossings=[],A5_crossings=cross,common_hidden_seams=common,
                selection='MIN_ACTIVITY_DIFFERENCE_THEN_ABS_SHIFT_THEN_SIGN; NO_OUTCOME_READS',
                context_frames_each_side=config['context_frames'],seam_count_per_arm=len(common),
                evaluated_seams_are_intervened=True,
                intervention_scope='SOURCE_LOCAL_BOUNDARY_SHIFT_NOT_RANDOMIZED_SEMANTIC_CAUSALITY')

class Assets:
    def __init__(self):
        conf=read('src/dunhuang/research/complete_r36/config.json'); assert sha(conf['catalog'])==conf['catalog_sha256']
        self.lib=Library(conf['catalog']); self.old=read('outputs/event_coverage_r47_01/index/result.json')
        self.cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json')
        self.external=read('outputs/paired_benchmark_r59_01/data/manifest.json')
        self.J=np.array(self.external['skeleton']['J']); self.parents=np.array(self.external['skeleton']['parents'])
        self.reference=read('outputs/bridge_dynamics_r48_01/main/result.json')['reference']
        self.R={};self.sources={};self.evidence={};self.units=[]

    def add(self,s,R,components,units=None):
        uid=s['source_uid']; p=profiles(R); evidence=phases(p,self.cfg); protected=protected_spans(evidence,len(R),self.cfg)
        if units is None:
            units=[]
            for comp in components:
                cuts=merge_protected(segment_source(p,protected,self.cfg,comp),protected)
                for a,b in cuts:
                    units.append(dict(unit_uid='r60:'+identity([uid,a,b]),source_uid=uid,
                                      source_interval=[a,b],support_component=comp,phases=phase_slice(evidence,[a,b])))
            # Metadata-only positions; do not choose by phase matchability or outcome.
            indices=np.unique(np.linspace(0,len(units)-1,min(len(units),CFG['external_units_per_source'])).round().astype(int))
            total=len(units);units=[units[i] for i in indices];s['all_segmented_units']=total
        for u in units:
            proof_interval(u['source_interval'],u['support_component'])
            item=dict(u);item['dataset']=s['dataset'];item['pair']=match_pair(u,p['upper'],protected)
            item['protected_crossings']=crossings(u['source_interval'],protected)
            item['phase_evidence']=phase_slice(evidence,u['source_interval'])
            item['semantic_completeness']='UNKNOWN';item['human_reviewed']=False;item['can_route']=False
            item['activity']=float(p['upper'][slice(*u['source_interval'])].mean());self.units.append(item)
        s.update(frames=len(R),accepted_components=components,protected_spans=protected,
                 phase_evidence=evidence,R_sha256=array_hash(R),semantic_truth='UNKNOWN',
                 root_contact='NOT_USED_FOR_BODY22_PROXY_EVALUATION')
        self.sources[uid]=s;self.R[uid]=R;self.evidence[uid]=p

    def build(self):
        for s in self.old['sources']:
            uid=s['source_uid'];units=[u for u in self.old['units'] if u['source_uid']==uid]
            self.add(dict(source_uid=uid,dataset='ChangE',role=s['role'],native=self.lib.rows[uid][0]['reference']),
                     self.lib.rotations[uid],s['accepted_components'],units)
        assert len(self.units)==139
        selected=[]
        for dataset in ('AIST++','FineDance'):
            candidates=sorted((s for s in self.external['sources'] if s['dataset']==dataset and s['role'] in ('train','genre_train_diagnostic')),
                              key=lambda s:(s['group'],s['source_uid']))
            group=set()
            for s in candidates:
                if s['group'] in group: continue
                group.add(s['group']);selected.append(s)
                if len(group)==CFG['external_sources_per_dataset']: break
            assert len(group)==CFG['external_sources_per_dataset']
        for s in selected:
            path=s['arrays'];assert sha(path)==s['arrays_sha256']
            with np.load(path) as z:R=z['R'].copy()
            self.add(dict(source_uid=s['source_uid'],dataset=s['dataset'],role=s['role'],group=s['group'],
                          raw_motion=s['raw_motion'],audio=s['audio'],audio_offset_frames=s['audio_offset_frames'],
                          arrays=dict(path=path,sha256=s['arrays_sha256']),clock_scope=s['clock_scope']),
                     R,[[s['shared_start'],len(R)]])
        return self

def prepare(out):
    out=fresh(out);a=Assets().build();legacy=[]
    for uid,rows in a.lib.rows.items():
        prot=a.sources[uid]['protected_spans']
        for r in rows:legacy.append(dict(event_uid=r['event_uid'],source_uid=uid,source_interval=r['source_interval'],
                                         crossings=crossings(r['source_interval'],prot)))
    bank=read('outputs/validation_r51_01/controls/plans/result.json')['source_bank']
    assert len(bank)==16 and set(bank)<=set(u['unit_uid'] for u in a.old['units'])
    result=dict(binding=binding(),protocol=CFG,sources=list(a.sources.values()),units=a.units,
                population_views=dict(legacy499=legacy,R47_139=[u['unit_uid'] for u in a.old['units']],R58_16=bank),
                legacy_accepted_frames=sum(s['accepted_frames'] for s in a.old['sources']),
                unknown_quality_frames_excluded=sum(s['unknown_quality_frames'] for s in a.old['sources']),
                skeleton=dict(J=a.J.tolist(),parents=a.parents.tolist()),reference=a.reference,
                parent_seal=seal_parents(),source_selection='ALL_R47139; FIRST_12_DISTINCT_AUDIO_GROUP_TRAIN_SOURCES_PER_PAIRED_DATASET; AT_MOST_3_TIME_QUANTILE_UNITS',
                supervision='KINEMATIC_PROXY_AND_ACTUAL_SAME_RECORDING_HIDDEN_MOTION; NOT_HUMAN_PHASE_TRUTH',
                human_reviewed=False,G1_phase_truth='NOT_ESTABLISHED',training_updates=0,can_route=False)
    assert result['legacy_accepted_frames']==41400 and len(legacy)==499
    write(out/'result.json',result);print('R60_POPULATION',len(a.sources),len(a.units),sum(u['pair']['status']=='MATCHED' for u in a.units),flush=True)
    return a,result
