"""Read-only independent comparison against saved generation receipts."""
import argparse,hashlib,json
from pathlib import Path

def read(path):return json.loads(path.read_text(encoding='utf8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report',default='reports/event_plan_workflow.json')
    ap.add_argument('--root',default='outputs/event_plan_01')
    args=ap.parse_args();report=read(Path(args.report));root=Path(args.root)
    for row in report['experiments']:
        folder=root/(row['profile']+'_'+str(row['frames']))
        physical=read(folder/'physical.json');generation=read(folder/'generation.json')
        assert digest(folder/'native.npy')==row['motion_sha256'],str(folder)
        assert physical['absolute']['ok']==row['native_pass']
        assert physical['fk_max_abs_m']==row['fk_error_m']
        assert generation['weights_before']==generation['weights_after']
        assert read(folder/'request.json')['code_commit']==row['generation_commit']
        for metric,value in row['native_metrics'].items():
            assert physical['audit'][metric]==value,(folder,metric)
        if row['profile']=='dunhuang':
            plan=read(folder/'event_plan.json')
            assert len(plan['events'])==row['summary']['events']
            assert plan['controlled_frames']/row['frames']==row['summary']['controlled_time_fraction']
            assert [(e['event_uid'],e['time_warp']) for e in plan['events']]==[(e['event_uid'],e['time_warp']) for e in row['selected_events']]
    assert read(root.parent/'sync_frozen_01/result.json')==report['sync']
    assert read(root.parent/'adapter_asset_verified/asset_audit.json')==report['motion_adapter_audit']
    assert read(root.parent/'local_time_retrieval.json')==report['local_time_retrieval']
    assert read(root.parent/'paired_adapter_preflight/status.json')==report['paired_adapter_preflight']
    print('Verified six motions, metrics, model immutability, plans, and training/evaluation receipts; no optimization executed.')
if __name__=='__main__':main()
