"""Task-separated preference import; reuse R69 administrative eligibility checks."""
from collections import defaultdict,Counter
from dunhuang.research.retiming_baselines_r69.reviews import qualify,stamp,text

def validate(doc,admin):
    if doc.get('schema')!='RETIME_PREF_V1' or doc.get('pack_id')!=admin['pack_id'] or doc.get('task') not in ['naturalness','music_coordination']:raise ValueError('Unknown task or pack')
    for k in ['review_id','reviewer_id','reviewer_roster_ref']:text(doc.get(k),k)
    stamp(doc.get('reviewed_at'),'reviewed_at')
    if type(doc.get('independence',{}).get('own_judgment')) is not bool:raise ValueError('Own judgment declaration required')
    items={i['id']:i for i in admin['items']};seen=set()
    for row in doc.get('items',[]):
        if row.get('id') not in items or row['id'] in seen or row.get('choice') not in ['A','B','TIE','UNSURE','MISSING']:raise ValueError('Invalid or duplicate preference')
        seen.add(row['id'])
        if not items[row['id']]['available'] and row['choice'] in ['A','B','TIE']:raise ValueError('Unavailable execution cannot be rated')
        if row.get('different') not in ['YES','NO','UNSURE','MISSING']:raise ValueError('Explicit observation status required')
    return doc

def summarize(documents,admin,roster=None):
    docs=[validate(d,admin) for d in documents]
    if len({d['review_id'] for d in docs})!=len(docs):raise ValueError('Duplicate review ID')
    grouped=defaultdict(lambda:defaultdict(list));states=Counter();ids={i['id']:i for i in admin['items']}
    for d in docs:
        q=qualify(d,admin,roster)
        provided={r['id']:r for r in d['items']}
        controls=[i for i in admin['items'] if i['control']]
        flow_ok=all(provided.get(i['id'],{}).get('different')==i['different_expected'] for i in controls)
        for r in d['items']:
            states[r['choice']]+=1;i=ids[r['id']]
            if q['eligible'] and flow_ok and not i['control'] and r['choice'] in ['A','B','TIE']:
                value=0 if r['choice']=='TIE' else 1 if r['choice']==i['positive_side'] else -1
                grouped[(d['task'],i['contrast'],i['source_uid'])][(i['id'],q['person_uid'])].append(value)
    output=[]
    for (task,contrast,source),cells in grouped.items():
        byitem=defaultdict(list)
        for (item,person),values in cells.items():byitem[item].append(sum(values)/len(values))
        valid=[sum(v)/len(v) for v in byitem.values() if len(v)>=2]
        output.append(dict(task=task,contrast=contrast,source_uid=source,qualified_items=len(valid),source_mean=sum(valid)/len(valid) if valid else None))
    import numpy as np
    comparisons={};rng=np.random.default_rng(73001)
    for task,contrast in {(r['task'],r['contrast']) for r in output}:
        values=np.array([r['source_mean'] for r in output if r['task']==task and r['contrast']==contrast and r['source_mean'] is not None])
        interval=None if len(values)<2 else np.quantile(rng.choice(values,(2000,len(values)),replace=True).mean(axis=1),[.025,.975]).tolist()
        comparisons[task+'__'+contrast]=dict(recordings=len(values),mean=None if not len(values) else float(values.mean()),source_bootstrap_95=interval)
    return dict(imported_documents=len(docs),states=dict(states),source_groups=output,comparisons=comparisons,quality='NOT_EVALUABLE' if not any(r['qualified_items'] for r in output) else 'DESCRIPTIVE_REAL_RECORDS_ONLY',
        group_unit='RECORDING_UID_NOT_WARP_OR_MUSIC_POSITION',phase_prediction='NOT_TRAINED',R74='NOT_STARTED')

def validate_phases(doc,sources):
    if doc.get('schema')!='SOURCE_PHASE_V1':raise ValueError('Unknown phase schema')
    source=sources[doc['source_uid']]
    if doc['source_sha256']!=source['native_sha256']:raise ValueError('Source identity changed')
    text(doc.get('annotator_id'),'annotator_id')
    for r in doc['intervals']:
        if not (type(r['start']) is int and type(r['end']) is int and 0<=r['start']<r['end']<=source['frames']):raise ValueError('Phase span outside actual source')
        if r['role'] not in ['PREPARATION','MAIN','HOLD','ENDING','UNKNOWN']:raise ValueError('Unregistered role')
    return doc

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--admin',required=True);parser.add_argument('--roster');parser.add_argument('--reviews',nargs='*',default=[]);parser.add_argument('--out',required=True);args=parser.parse_args()
    load=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    result=summarize([load(p) for p in args.reviews],load(args.admin),load(args.roster) if args.roster else None)
    p=Path(args.out)
    if p.exists():raise ValueError('Fresh import summary required')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
