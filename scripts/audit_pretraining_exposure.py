"""Published recipe membership audit. Cannot infer actual checkpoint lineage."""
import argparse,json,sys
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,pinned,sha,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    assets=json.loads(Path(a.assets).read_text())
    edge=pinned(assets['edge_root'],assets['edge_commit']);d=edge/'data/splits'
    sets={name:set((d/name).read_text().split()) for name in ('crossmodal_train.txt','crossmodal_test.txt','ignore_list.txt')}
    train=sets['crossmodal_train.txt']-sets['ignore_list.txt']
    test=sets['crossmodal_test.txt']-sets['ignore_list.txt']
    manifest=json.loads(checked(assets['paired_generation_manifest']).read_text())
    seen=set();counts=defaultdict(Counter)
    for r in manifest['rows']:
        name=r['recording_uid'].split(':')[-1];key=(r['role'],name)
        if key in seen:continue
        seen.add(key)
        state=('official_train_retained' if name in train else 'official_test_retained' if name in test else
               'official_ignore' if name in sets['ignore_list.txt'] else 'not_in_pinned_edge_split')
        counts[r['role']][state]+=1
    write(out,dict(pinned_edge_commit=assets['edge_commit'],checkpoint=assets['checkpoint'],
        split_files={name:dict(sha256=sha(d/name),count=len(v)) for name,v in sets.items()},
        admitted_recordings_by_project_role=dict(counts),
        checkpoint_training_membership='NOT_AUDITED_FROM_CHECKPOINT_OPTIMIZER_LINEAGE',
        scope='Published recipe/list membership, not proof of exact checkpoint exposure or unseen-data generalization.'))
    print('PRETRAINING_RECIPE_MEMBERSHIP',dict(counts))

if __name__=='__main__':main()
