"""Predeclared synthetic kinematic errors; never real human quality labels."""
from pathlib import Path
import numpy as np
from dunhuang.io import write,sha
from .smpl_mesh import SMPLMesh
from .physdiff_metrics import population_metrics as metrics
from .contact_evidence import conditioned_metrics
from .calibration import jerk_report


def controlled(backend,mesh_assets,config,out):
    out=Path(out);out.mkdir()
    mesh=SMPLMesh(mesh_assets['mesh_model'],mesh_assets.get('mesh_device','cpu'))
    n=config['controlled']['frames'];base=np.zeros((n,151),np.float32)
    base[:,7:]=np.tile([1,0,0,0,1,0],24);base[:,5]=1.
    baseline_minimum,_=mesh.sequence(base);floor=float(np.percentile(baseline_minimum,5))
    np.save(out/'baseline.npy',base)
    design=dict(schema='synthetic_native_validity_design_v1',baseline_sha256=sha(out/'baseline.npy'),
        ground_y_m=floor,ground_source='specified_synthetic_world_not_real_scene_calibration',
        source_quality='synthetic neutral pose, not independently reviewed real dance',
        injected_motion_is_copy=True,config=config['controlled'])
    write(out/'design.json',design)
    cases=[]
    def evaluate(name,amplitude,x,labels,meaning):
        folder=out/name;folder.mkdir();np.save(folder/'native.npy',x);np.save(folder/'labels.npy',labels)
        fk=backend.fk(x);minimum,joints=mesh.sequence(x)
        value=backend.audit(x,backend.cfg,precomputed_joints=fk);gate=backend.gate(value)
        row=dict(case=name,amplitude=float(amplitude),design=meaning,motion_sha256=sha(folder/'native.npy'),
            label_evidence='SYNTHETIC_DESIGN_ONLY_NOT_REAL_CONTACT_ANNOTATION',
            legacy=gate,legacy_floor_y_m=value['floor_y_m'],legacy_foot_penetration_min_m=value['foot_penetration_min_m'],
            fixed_floor_y_m=floor,mesh=metrics(minimum,joints,floor,contact_height=.05),
            independent_synthetic_contact=conditioned_metrics(fk,floor,labels=labels),
            jerk=jerk_report(fk,backend.limits['joint_jerk_mps3_max']))
        write(folder/'report.json',row);cases.append(row)
    static=np.ones((n,4),np.int8)
    for speed in config['controlled']['static_slide_mps']:
        x=base.copy();x[:,4]+=speed*np.arange(n)/30
        evaluate('static_slide_'+str(speed),speed,x,static,'Known static-contact design with imposed horizontal Root translation')
        evaluate('declared_slide_'+str(speed),speed,x,np.full((n,4),2,np.int8),'Same kinematics, explicitly declared synthetic slide task; not automatic legitimate real sliding')
    for key,sign in [('penetration_offsets_m',-1),('float_offsets_m',1)]:
        for a in config['controlled'][key]:
            x=base.copy();x[:,5]+=sign*a
            evaluate(key+'_'+str(a),a,x,static,'Global vertical offset against a frozen synthetic plane')
    for a in config['controlled']['jitter_amplitudes_m']:
        x=base.copy();x[:,5]+=a*np.sin(2*np.pi*config['controlled']['jitter_hz']*np.arange(n)/30)
        evaluate('root_jitter_'+str(a),a,x,static,'Fixed-frequency RootY sinusoid on static synthetic design')
    x=base.copy();x[:,5]+=.3
    evaluate('synthetic_flight_no_static',.3,x,np.zeros((n,4),np.int8),'No independently labelled static pairs')
    x=base.copy();x[:,4]+=.3*np.arange(n)/30
    evaluate('unknown_high_speed',.3,x,np.full((n,4),-1,np.int8),'High speed alone must not become intentional slide')
    def curve(prefix,get):
        rows=sorted([r for r in cases if r['case'].startswith(prefix)],key=lambda r:r['amplitude'])
        values=[get(r) for r in rows]
        return dict(amplitudes=[r['amplitude'] for r in rows],values=values,
                    nondecreasing=bool(np.all(np.diff(values)>=-1e-5)))
    curves=dict(slide=curve('static_slide_',lambda r:r['independent_synthetic_contact']['independent_static_speed']['p95']),
        penetration=curve('penetration_offsets_m_',lambda r:r['mesh']['penetrate_mm']),
        floating=curve('float_offsets_m_',lambda r:r['mesh']['float_mm']),
        jitter=curve('root_jitter_',lambda r:r['jerk']['raw']['max']))
    if not all(c['nondecreasing'] for c in curves.values()):raise ValueError('Predeclared synthetic operator sensitivity not monotonic')
    write(out/'report.json',dict(status='COMPLETED_CONTROLLED_KINEMATIC_PERTURBATIONS',cases=cases,curves=curves,
        genuine_quality_false_reject_rate=None,real_world_detection_rate=None,
        used_to_tune_thresholds=False,operational_qualified=False,optimizer_calls=0))
