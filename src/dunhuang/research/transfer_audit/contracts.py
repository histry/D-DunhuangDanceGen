"""Explicit new task closure and Train-only statistics. Historical roles survive."""
import numpy as np

def validate_song_task(rows):
    seen={}
    for r in rows:
        if r["role"] not in ("train","val"):raise ValueError("Test tensors forbidden")
        for kind,value in (("music",r["music_uid"]),("recording",r["recording_uid"]),
            ("group",r["source_group"]),("raw",r["raw_motion"]["sha256"]),("audio",r["audio"]["sha256"])):
            if not value:raise ValueError("Required task identity unknown")
            key=(kind,value)
            if key in seen and seen[key]!=r["role"]:raise ValueError("Song-task identity crosses roles")
            seen[key]=r["role"]
    # Shared choreography is legal for the unseen-song task, not for every task.

class Moments:
    def __init__(self,width):
        self.count=np.zeros(width,np.int64);self.total=np.zeros(width);self.squares=np.zeros(width)
    def add(self,values,observed,role):
        if role!="train":return
        x=np.asarray(values,np.float64);mask=np.asarray(observed)
        if x.ndim!=2 or mask.shape!=x.shape or mask.dtype!=bool or x.shape[1]!=len(self.count):raise ValueError("Statistics support differs")
        if not np.isfinite(x[mask]).all():raise ValueError("Nonfinite observed Train data")
        x=np.where(mask,x,0);self.count+=mask.sum(0);self.total+=x.sum(0);self.squares+=(x*x).sum(0)
    def result(self):
        mean=np.divide(self.total,self.count,out=np.zeros_like(self.total),where=self.count>0)
        variance=np.maximum(np.divide(self.squares,self.count,out=np.zeros_like(self.total),where=self.count>0)-mean**2,0)
        std=np.maximum(np.sqrt(variance),1e-6);std[self.count==0]=1
        return dict(mean=mean.tolist(),std=std.tolist(),observed_count=self.count.tolist(),scope="Train-only full recordings/audio once per asset")

def training_gate(populations,protocol,checkpoint_binding):
    checks=dict(population_sufficient=all(p["population_sufficient"] for p in populations.values()),
        work_aliases_certified=protocol.get("composition_aliases_outside_publisher_ids")=="VERIFIED",
        pretrained_exposure_certified=protocol.get("pretrained_backbone_exposure")=="VERIFIED",
        task_training_authorized=protocol.get("ready_for_new_training") is True,
        checkpoint_bound=bool(checkpoint_binding))
    return dict(ready=all(checks.values()),checks=checks,optimizer_updates=0,
        reason="Candidate population alone does not authorize new encoder/head training")
