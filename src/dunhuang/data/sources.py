"""Hash-bound local research assets; raw restricted data is never Git content."""
import json,shutil
from pathlib import Path
from dunhuang.io import sha,checked,write

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def relative(asset):
    text=asset['path'].replace(chr(92),'/')
    suffix=text.split('/assets/datasets/')[-1]
    path=Path(suffix)
    if path.is_absolute() or '..' in path.parts or suffix==text:
        raise ValueError('Asset is outside registered dataset root')
    return path
def raw_path(root,asset,required=True):
    path=Path(root)/relative(asset)
    if required:checked(dict(path=str(path),sha256=asset['sha256']))
    return path
def materialize(source,output,population):
    source=Path(source);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    payload=read(population);files={}
    for row in payload['records']:
        asset=row['motion_asset'];rel=relative(asset)
        original=raw_path(source/'assets/datasets',asset)
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target)
        if sha(target)!=asset['sha256']:raise ValueError('Copied source differs')
        files[rel.as_posix()]=dict(sha256=sha(target),bytes=target.stat().st_size)
    for rel in ('README.md','aistpp/README.md','finedance/README.md',
                'aistpp/annotations/ignore_list.txt','aistpp/metadata/motion_file_manifest.json',
                'aistpp/metadata/music_file_manifest.json','finedance/metadata/official_LICENSE',
                'finedance/metadata/official_data_code_pre_motion.py'):
        original=source/'assets/datasets'/rel
        if not original.exists():continue
        target=output/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
        files[rel]=dict(sha256=sha(target),bytes=target.stat().st_size)
    shutil.copyfile(population,output/'population.json')
    write(output/'inventory.json',dict(schema='paired_raw_copy_v1',population_sha256=sha(population),
        files=files,audio_copied=False,source_root=str(source.resolve()),dataset_admission=False))
    return output

def validate_links(records):
    identities={};roles={};members={}
    for row in records:
        role=row['purpose_role']
        if role=='excluded_official_ignore' and row.get('official_ignore'):continue
        group=row['group_uid']
        if role not in ('train','val','test'):raise ValueError('Explicit historical role required')
        if group in roles and roles[group]!=role:raise ValueError('Component role leakage')
        roles[group]=role;members.setdefault(group,[]).append(row['pair_uid'])
        keys=[('motion',row['motion_asset']['sha256']),('audio',row['audio_asset']['sha256'])]
        for field in ('music_id','normalized_song_name','recording_uid','choreography_uid','conservative_family_key'):
            if row.get(field):keys.append((row['dataset_id']+':'+field,row[field]))
        for key in keys:
            if key in identities and identities[key]!=role:raise ValueError('Known identity crosses roles: '+str(key))
            identities[key]=role
    return dict(known_links_isolated=True,groups=len(roles),roles=roles,
        grouping_scope='historical known links and conservative families; recording independence unverified',
        blind_test_claimed=False,recording_independence_verified=False)
