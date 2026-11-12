"""Equal-capacity closed-form probes on a shared real 30Hz grid, not semantic labels."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from dunhuang.io import checked,write
from .clock_data import read
from .model import RetrievalModel
from .clock_metrics import grouped

GRID=np.arange(2,117)
TYPES=("start","speed_peak","deceleration","turn","hold")

def reconstruct(samples,source_grid,target_grid=GRID):
    source_grid=np.asarray(source_grid)
    if target_grid.min()<source_grid.min() or target_grid.max()>source_grid.max():raise ValueError("No phase extrapolation")
    return np.stack([np.column_stack([np.interp(target_grid,source_grid,row[:,c]) for c in range(row.shape[1])]) for row in samples]).astype(np.float32)

def stages(raw,tcn,transformer):
    if raw.shape[1]!=120 or tcn.shape[1]!=120 or transformer.shape[1]!=60:raise ValueError("Registered 120/60 source grids required")
    raw=np.pad(raw,((0,0),(0,0),(0,128-raw.shape[-1])))
    antialias=(tcn[:,1:117:2]+2*tcn[:,2:118:2]+tcn[:,3:119:2])/4
    return dict(A0_raw=raw[:,GRID],A1_tcn=tcn[:,GRID],
        A2_even=reconstruct(tcn[:,::2],np.arange(0,120,2)),
        A3_antialias=reconstruct(antialias,np.arange(2,118,2)),
        A4_transformer=reconstruct(transformer,np.arange(0,120,2)))

def binary_auc(y,s):
    y=np.asarray(y,bool).reshape(-1);s=np.asarray(s,float).reshape(-1)
    p=int(y.sum());n=len(y)-p
    if not p or not n:return None
    ranks=rankdata(s,method="average")
    return float((ranks[y].sum()-p*(p+1)/2)/(p*n))

def binary_metrics(y,p):
    yp=p>=.5;y=y.astype(bool);f=[];au=[]
    for k in range(y.shape[-1]):
        a=y[...,k];b=yp[...,k];tp=int((a&b).sum());den=int(a.sum()+b.sum())
        f.append(2*tp/den if den else None);au.append(binary_auc(a,p[...,k]))
    return dict(macro_auc=float(np.mean([v for v in au if v is not None])) if any(v is not None for v in au) else None,
        per_class_auc=au,per_class_f1_at_half=f,brier=float(np.mean((p-y)**2)),score_is_calibrated_probability=False)

def ridge_fit(x,y,train,regularization):
    # Only Train enters feature statistics, response means and the normal equations.
    a=x[train].reshape(-1,x.shape[-1]).astype(np.float64);b=y[train].reshape(-1,y.shape[-1]).astype(np.float64)
    mean=a.mean(0);std=a.std(0);std[std<1e-6]=1;a=(a-mean)/std;offset=b.mean(0)
    beta=np.linalg.solve(a.T@a/len(a)+regularization*np.eye(a.shape[1]),a.T@(b-offset)/len(a))
    return dict(mean=mean,std=std,beta=beta,offset=offset)

def predict(model,x):
    return ((x-model["mean"])/model["std"])@model["beta"]+model["offset"]

def peak_neighborhood(values):
    center=(values[1:-1]>values[:-2])&(values[1:-1]>=values[2:])
    peaks=np.pad(center,((1,1),(0,0)))
    return np.maximum.reduce([peaks,np.pad(peaks[:-1],((1,0),(0,0))),np.pad(peaks[1:],((0,1),(0,0)))]).astype(np.float32)

def targets(source,clock,rows):
    # FK speeds already came from whole-record derivatives in the verified source cache.
    speeds=np.load(Path(source)/"data/probes.npz")["target"]
    motion_type=np.zeros((len(rows),120,5),np.float32);music_type=np.zeros_like(motion_type)
    music_values=np.empty((len(rows),120,3),np.float32);music_peak=np.empty_like(music_values)
    motion_peak=np.stack([peak_neighborhood(x) for x in speeds]);audio={};events={}
    for i,r in enumerate(rows):
        ak=r["audio"]["sha256"];rk=r["raw_motion"]["sha256"]
        if ak not in audio:
            values=np.load(Path(clock)/"data/audio"/(ak+".npz"))["values"]
            audio[ak]=(values[:,[0,1,11]],peak_neighborhood(values[:,[1,11,2]]),
                np.maximum(np.diff(values[:,0],prepend=values[0,0]),0))
        if rk not in events:events[rk]=read(Path(source)/"data/sidecars"/(rk+".json"))["events"]
        if ak not in events:events[ak]=read(Path(source)/"data/sidecars"/(ak+".json"))["events"]
        offset=r["audio_offset_seconds"]*30
        if abs(offset-round(offset))>1e-6:raise ValueError("Sub-frame annotation grid unavailable")
        a=r["start_frame"]+15;ma=a+round(offset)
        values,peaks,growth=audio[ak]
        music_values[i]=values[ma:ma+120];music_peak[i]=peaks[ma:ma+120]
        for side,key,origin,names in ((motion_type,rk,a,TYPES),(music_type,ak,ma,("onset","accent","beat","silence"))):
            for e in events[key]:
                if e["event_type"] in names:
                    index=e["frame"]-origin
                    for j in range(max(0,index-1),min(120,index+2)):side[i,j,names.index(e["event_type"])]=1
        g=peak_neighborhood(growth[:,None])[:,0];music_type[i,:,4]=g[ma:ma+120]
    return dict(music=np.concatenate((music_values,music_peak,music_type),-1)[:,GRID],
        motion=np.concatenate((speeds,motion_peak,motion_type),-1)[:,GRID])

def run(source,clock,rows,config,out):
    out=Path(out);out.mkdir()
    obs=dict(np.load(Path(clock)/"data/observations.npz"));enc=np.load(Path(clock)/"encodings.npz")
    manifest=read(Path(source)/"data/manifest.json")
    model=RetrievalModel().cuda().eval()
    model.load_state_dict(torch.load(checked(manifest["global_checkpoint"]),map_location="cuda",weights_only=True)["state"])
    extracted={k:[] for k in ("music_tcn","motion_tcn","music_transformer","motion_transformer")}
    with torch.no_grad():
        for start in range(0,len(rows),32):
            m=torch.as_tensor(obs["music"][start:start+32],device="cuda");mm=torch.as_tensor(obs["mm"][start:start+32],device="cuda")
            d=torch.as_tensor(obs["motion"][start:start+32,15:135],device="cuda");dm=torch.as_tensor(obs["dm"][start:start+32,15:135],device="cuda")
            for name,x,mask in (("music",m,mm),("motion",d,dm)):
                t,_=getattr(model,name).temporal_features(x,mask);_,z,_=getattr(model,name)(x,mask)
                extracted[name+"_tcn"].append(t.cpu().numpy());extracted[name+"_transformer"].append(z.cpu().numpy())
    extracted={k:np.concatenate(v) for k,v in extracted.items()}
    for name,saved in (("music_transformer",enc["mt"]),("motion_transformer",enc["dt"][:,2]),("motion_tcn",enc["tcn"])):
        if not np.array_equal(extracted[name],saved):raise ValueError("Frozen encoder feature rebuild differs: "+name)
    y=targets(source,clock,rows);train=np.array([i for i,r in enumerate(rows) if r["role"]=="train"]);val=np.array([i for i,r in enumerate(rows) if r["role"]=="val"])
    results={};predictions={};models={}
    for modality in ("music","motion"):
        raw=obs["music"] if modality=="music" else obs["motion"][:,15:135]
        representations=stages(raw,extracted[modality+"_tcn"],extracted[modality+"_transformer"])
        results[modality]={}
        for name,x in representations.items():
            fit=ridge_fit(x,y[modality],train,config["ridge_regularization"]);prediction=predict(fit,x[val]);true=y[modality][val]
            reg=(prediction[...,:3]-true[...,:3])**2
            report=dict(parameters=128*11+11,regression_mse=float(reg.mean()),per_channel_mse=reg.mean((0,1)).tolist(),
                peak_neighborhood=binary_metrics(true[...,3:6],prediction[...,3:6]),event_type_proxy=binary_metrics(true[...,6:],prediction[...,6:]),
                peak_phase_mae_seconds=(np.abs(prediction[...,:3].argmax(1)-true[...,:3].argmax(1))/30).mean(0).tolist(),
                mse_grouped=grouped(reg.mean((1,2)),[rows[i] for i in val]),shift_response={})
            for shift in (-3,-2,-1,0,1,2,3):
                # Intersect observed support, never wrap or pad. Target frames5..113.
                shifted=predict(fit,x[val,3+shift:112+shift]);target=y[modality][val,3:112]
                report["shift_response"][str(shift)]=dict(seconds=shift/30,regression_mse=float(np.mean((shifted[...,:3]-target[...,:3])**2)),
                    peak_neighborhood=binary_metrics(target[...,3:6],shifted[...,3:6]))
            results[modality][name]=report;predictions[modality+"_"+name]=prediction
            for k,v in fit.items():models[modality+"_"+name+"_"+k]=v
            print("LOCAL_INFO_PROBE",modality,name,report["regression_mse"],flush=True)
    np.savez_compressed(out/"predictions.npz",**predictions);np.savez_compressed(out/"ridge_models.npz",**models)
    report=dict(schema="local_information_audit_v1",results=results,grid_frames=GRID.tolist(),grid_fps=30,
        fitting="Train-only closed-form ridge, no neural optimizer",probe_parameters_per_stage=1419,encoder_changed=False,
        targets="numerical acoustic reconstruction / fixed-offset FK speed / detector proxies; no human event correspondence",
        music_type_proxy=["onset","accent","beat","low_activity","energy_growth"],motion_type_proxy=list(TYPES),
        causal_scope="linear accessibility on common real grid; not proof of semantic information destruction",
        optimizer_calls=0,fresh_blind=False)
    write(out/"result.json",report);return report
