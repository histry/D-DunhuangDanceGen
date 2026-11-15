"""Matched full-Train budget; replay reads actual updates without an optimizer."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.common import capture,restore,parameters,vector
from dunhuang.research.temporal_r20.common import selected_digest,frozen_except_null
from dunhuang.research.temporal_r20.training import backward,observe_before,parameter_stats
from dunhuang.research.masked_denoiser.common import tensor_hash
from .engine import load,cache_data,state,schedule,weights,probes,model_meta
from .objective import build,branch_scores

def after(cfg,arm,record,gu,delta,p,s,w,h,cache,ow,tw,forward,back,clipping):
    with torch.no_grad():
        _,terms,_,_,_=build(cfg,p,cache,s,ow,tw,arm)
        branches=branch_scores(p,s[0],s[4])
    return dict(record,unknown_after=float(terms["unknown"]),guided_unknown_after=float(terms["guided_unknown"]),
        unknown_gradient_dot_update=float(gu@delta),actual_update_norm=float(delta.norm()),
        parameter_stats=parameter_stats(w,h,delta),layers_forward=forward,layers_backward=back,clipping=clipping,
        branches_after=branches)

def run(cfg,root,out,arm,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    w,d,data,p,h=load(cfg,root,arm);cache=cache_data(cfg,root,d,data);ow,tw=weights(cfg)
    initial=model_meta(w,h);rows=schedule(cfg,cache["population"]);curves=[];history=[];updates=[];curve_refs={}
    if replay:
        previous=read(Path(replay)/"result.json");initial_ref=previous["initial_checkpoint"]
        assert initial==previous["initial"];restore(cfg,w,h,initial_ref)
    else:
        (out/"snapshots").mkdir();initial_ref=capture(cfg,w,h,out/"snapshots/0_pre.pt",arm,0)
    def curve(update,weight):
        scores,arrays=probes(cfg,cache,p);path=out/f"curve_{update}.npz";np.savez_compressed(path,**arrays)
        if replay:
            old=next(c for c in previous["curves"] if c["update"]==update)
            assert scores==old["scores"];same_arrays(path,old["predictions"]["path"])
        curves.append(dict(update=update,weight=weight,scores=scores,predictions=ref(path)))
    curve(0,initial_ref);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if not replay:
        optimizer=torch.optim.AdamW(parameters(w,h),lr=cfg["learning_rate"])
        for row in rows:
            step=row["step"];s,_,_=state(cfg,cache,row["case"],row["t"],row["variant"],row["pattern"],row["seed"])
            row.update(x_t_sha256=tensor_hash(s[-1]),known_sha256=tensor_hash(s[4]))
            watch=step in cfg["snapshot_steps"];p.telemetry.enabled=watch;p.telemetry.grad_enabled=watch
            pred,terms,counts,active,obj=build(cfg,p,cache,s,ow,tw,arm);assert torch.isfinite(obj)
            if watch:
                pre=initial_ref if step==0 else capture(cfg,w,h,out/f"snapshots/{step}_pre.pt",arm,step)
                before=vector(w,h);record,gu=observe_before(w,h,pred,terms,counts,active,cache)
                record["branches_before"]=branch_scores(p,s[0],s[4])
                forward=copy.deepcopy(p.telemetry.forward);p.telemetry.backward={}
            clipping=backward(w,h,obj)
            if watch:back=copy.deepcopy(p.telemetry.backward)
            optimizer.step()
            if watch:
                actual=after(cfg,arm,record,gu,vector(w,h)-before,p,s,w,h,cache,ow,tw,forward,back,clipping)
                post=capture(cfg,w,h,out/f"snapshots/{step}_post.pt",arm,step+1)
                updates.append(dict(step=step,pre=pre,post=post,readout=actual))
            p.telemetry.enabled=False;p.telemetry.grad_enabled=False
            history.append(dict(step=step,unknown=float(terms["unknown"].detach()),guided_unknown=float(terms["guided_unknown"].detach()),loss=float(obj.detach())))
            if step+1 in cfg["curve_updates"]:
                post=updates[-1]["post"] if watch else capture(cfg,w,h,out/f"snapshots/{step}_post.pt",arm,step+1)
                curve_refs[step+1]=post;curve(step+1,post)
            if (step+1)%100==0:print("R22_TRAIN",arm,step+1,history[-1],flush=True)
        final=curve_refs[len(rows)];optimizer_steps=len(rows)
    else:
        for item in previous["update_observations"]:
            row=rows[item["step"]];s,_,_=state(cfg,cache,row["case"],row["t"],row["variant"],row["pattern"],row["seed"])
            restore(cfg,w,h,item["pre"]);p.telemetry.enabled=True;p.telemetry.grad_enabled=True
            pred,terms,counts,active,obj=build(cfg,p,cache,s,ow,tw,arm)
            before=vector(w,h);record,gu=observe_before(w,h,pred,terms,counts,active,cache)
            record["branches_before"]=branch_scores(p,s[0],s[4])
            forward=copy.deepcopy(p.telemetry.forward);p.telemetry.backward={}
            clipping=backward(w,h,obj);back=copy.deepcopy(p.telemetry.backward)
            restore(cfg,w,h,item["post"])
            actual=after(cfg,arm,record,gu,vector(w,h)-before,p,s,w,h,cache,ow,tw,forward,back,clipping)
            assert actual==item["readout"],("actual-update replay",arm,item["step"])
            p.telemetry.enabled=False;p.telemetry.grad_enabled=False
        for row in rows:
            s,_,_=state(cfg,cache,row["case"],row["t"],row["variant"],row["pattern"],row["seed"])
            row.update(x_t_sha256=tensor_hash(s[-1]),known_sha256=tensor_hash(s[4]))
        assert rows==previous["ledger"]
        for c in previous["curves"][1:]:restore(cfg,w,h,c["weight"]);curve(c["update"],c["weight"])
        final=previous["checkpoint"];restore(cfg,w,h,final);optimizer_steps=0;updates=previous["update_observations"]
    scores,arrays=probes(cfg,cache,p);np.savez_compressed(out/"final_q.npz",**arrays)
    assert w.digest(True)==initial["frozen"] and frozen_except_null(w)==initial["frozen_except_null"]
    result=dict(arm=arm,layout="joint",initial=initial,initial_checkpoint=initial_ref,checkpoint=final,curves=curves,ledger=rows,history=history,
        update_observations=updates,final_scores=scores,final_q=ref(out/"final_q.npz"),
        final_selected=selected_digest(w),final_head=model_meta(w,h)["head"],optimizer_steps=optimizer_steps,
        trainable_motion=sum(x.numel() for x in w.selected()),trainable_head=sum(x.numel() for x in h.parameters()),
        wall_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),forward_calls=p.calls,
        whole_song_generations=0,operational_qualified=False)
    if replay:
        assert scores==previous["final_scores"];same_arrays(out/"final_q.npz",previous["final_q"]["path"])
        assert result["final_selected"]==previous["final_selected"] and result["final_head"]==previous["final_head"]
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),data=ref(root/"data/result.json"),
        warm=ref(Path(cfg["parent"])/"models/B3/result.json"),dev_selection=False))
    p.close()
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--config",required=True);a.add_argument("--root",required=True);a.add_argument("--out",required=True)
    a.add_argument("--arm",required=True);a.add_argument("--replay-from");x=a.parse_args();run(read(x.config),x.root,x.out,x.arm,x.replay_from)
