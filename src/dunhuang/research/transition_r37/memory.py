"""Raw source interval reuse, not UID diversity or cultural semantics."""
def union_length(intervals):
    out=[]
    for a,b in sorted(intervals):
        if not a<b:raise ValueError('Empty interval')
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return sum(b-a for a,b in out)

def reuse(events):
    sources={}
    for e in events:sources.setdefault(e['source_uid'],[]).append(e['source_interval'])
    total=sum(b-a for values in sources.values() for a,b in values)
    unique=sum(union_length(values) for values in sources.values())
    return dict(total_source_frames=total,unique_source_frames=unique,reuse_ratio=1-unique/total if total else 0.,
                per_source={s:dict(used_frames=sum(b-a for a,b in v),unique_frames=union_length(v),intervals=v) for s,v in sorted(sources.items())})

def incremental(path,event,music):
    a,b=event['source_interval'];same=[p for p in path if p['source_uid']==event['source_uid']]
    intervals=[p['source_interval'] for p in same]
    new=union_length(intervals+[[a,b]])-union_length(intervals)
    overlap=1-new/(b-a)
    # Repetition evidence can discount distant returns, never immediate replay.
    theme=False
    for p in same:
        if event['music_interval'][0]-p['music_interval'][1]<480:continue
        for r in music.get('repetitions',[]):
            if r['first'][0]<=p['music_interval'][0]<r['first'][1] and r['second'][0]<=event['music_interval'][0]<r['second'][1]:theme=True
    backward=bool(same and a<same[-1]['source_interval'][1]-12)
    return dict(overlap_fraction=overlap,new_source_frames=new,backtrack=backward,theme_return_evidence=theme,
                theme_discount=.5 if theme else 1.)
