"""Audit actual channel consumption; contact proxies are never measured-contact truth."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.bridge_r38.study import runtime
from .train import assemble


def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh observation audit')
    forbid_optimizers();rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));z=np.load(root/'data_v2/arrays.npz');rows=read(root/'data_v2/result.json')['records'];checks=[]
    for arm in ('main','real','aist'):
        assert read(root/arm/'protocol.json')['data_sha256']==sha(root/'data_v2/arrays.npz')
        assert read(root/arm/'training.json')['contact_loss'] is False
    for start in range(0,len(rows),4):
        part=rows[start:start+4];a,m,ref,cp=assemble(rt,z,part);assert not cp.valid[:,:,:4].any();assert not ref.values['valid'][:,104:152].any();cp.validate()
        for j,row in enumerate(part):
            if not row['root_position_observed']:assert not cp.valid[j,:,4:7].any()
            checks.append(dict(key=row['key'],domain=row['domain'],role=row['role'],archive_contact_observed_flag=row.get('contact_observed'),contact_truth_observed=False,contact_proxy_available=row['domain']=='AIST++',actual_context_C4_valid=False,actual_source_feature_channels='ROW6D_AND_ANGULAR_VELOCITY_ONLY_NO_C4',actual_hidden_pose_valid=False,root_position_supported_in_model_gauge=row['root_position_observed'],absolute_world_root_truth=False))
    rt.c.clear_all();rt.check();rt.close();write(out,dict(records=checks,data_arrays_sha256=sha(root/'data_v2/arrays.npz'),archive_metadata_contact_flag_mislabelled_proxy_presence=True,archive_metadata_preserved=True,numeric_inputs_or_weights_changed=False,contact_supervision_used=False,all_C4_condition_masks_false=True,all_hidden_reference_masks_false=True,optimizer_steps=0,native24='NOT_EVALUABLE_22_OF24'));print('R42_OBSERVATION_AUDIT',len(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
