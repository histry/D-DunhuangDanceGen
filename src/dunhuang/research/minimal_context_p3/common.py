from pathlib import Path
from dunhuang.research.whole_song_witness_p2.common import inputs as p2_inputs,reconstruct,read,write,sha,array_hash,save_arrays,fresh,identity,check_seal,boundary_key,relative_fk

ROOT=Path('outputs/paper1_minimal_context_edit_p3_01')
P2=Path('outputs/paper1_whole_song_witness_p2_01')
CFG=read(Path(__file__).with_name('protocol.json'))

def binding():return {p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json','.html')}

def seal(folder):
    write(folder/'manifest.json',dict(science=binding(),files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}))

def inputs():
    for stage,digest in CFG['P2_stage_manifests'].items():
        assert sha(P2/stage/'manifest.json')==digest;check_seal(P2/stage)
        receipt=read(P2/'receipts'/('independent_'+stage+'.json'));assert receipt['status']=='PASS' and receipt['source_manifest_sha256']==digest
    assert sha(P2/'main/viewer/data.js')==CFG['P2_viewer_data_sha256']
    p,z,rows,slots,bodies,graph,paths=p2_inputs()
    return p,z,rows,slots,graph,read(P2/'main/result.json')

def eligible_edges(graph,song,layers):
    where={u:k for k,layer in enumerate(layers) for u in layer};edges={}
    for e in graph['graph']:
        u,v=e['left'],e['right']
        if e['song']!=song or u not in where or v not in where:continue
        if e['measured'] and e['eligible'] and e['policy_admissible']:
            assert e['numeric']['passes_original_screen'] and where[v]==where[u]+1
            edges[u,v]=e
    return edges
