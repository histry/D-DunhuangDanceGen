import argparse
import numpy as np
from dunhuang.research.complete_r36.core import read,write,Library

def run(out):
    lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);rows=[]
    for uid in sorted(lib.rows):
        x=lib.native[uid];entry=lib.rows[uid][0]
        rows.append(dict(source_uid=uid,source_sha256=entry['source_sha256'],native_sha256=entry['reference']['sha256'],shape=list(x.shape),finite=bool(np.isfinite(x).all()),root_range_m=np.ptp(x[:,4:7],axis=0).tolist(),contact_channel_unique=[np.unique(x[:,i]).tolist() for i in range(4)],body22_observed=True,world_root_truth='UNKNOWN_UNVERIFIED_COORDINATE_AND_OBSERVATION',contact_truth='UNKNOWN_NO_INDEPENDENT_MEASUREMENT',foot_slide_truth='NOT_EVALUABLE',native24_qualification=False))
    source=read('outputs/paired_event_r41_01/data/manifest.json');counts={r:sum(s['role']==r for s in source['sources']) for r in ('train','val')}
    write(out,dict(change=rows,aist_source_counts=counts,aist_root_scope='CLOCK_VERIFIED_FITTED_TRANSLATION_NOT_CONTACT_TRUTH',finedance_admitted=False,contact_truth_sources=0,whole_body_physics_qualified=False,can_route=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
