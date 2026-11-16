"""One shared prior/IO, two small reference modules, explicit diffusion-time binding."""
from pathlib import Path
import torch
from .controller import Controller
from .context import ContextBranch
from dunhuang.research.time_r29.model import TimeReference
from dunhuang.research.content_r32.common import runtime as old_runtime
from dunhuang.research.state_r27.common import read,sha,digest

def explicit_forward(io,original):
    def forward(x,cond,genre,t,cond_drop_prob=0):
        if t.ndim!=1 or len(t)!=len(x) or t.dtype!=torch.long:raise ValueError("Explicit batch diffusion t required")
        return io.decode(original(io.encode(x,t),cond,genre,t,cond_drop_prob=cond_drop_prob))
    return forward

class Runtime:
    def __init__(self,cfg,checkpoint=None):
        root=Path(cfg["r32"]);assert sha(root/"artifacts_manifest.json")==cfg["r32_manifest_sha256"]
        path=root/"models/ChangE/adapter.pt";assert sha(path)==cfg["change_checkpoint_sha256"]
        self.d,self.n,self.s,self.f,self.e,self.info,self.old=old_runtime(read("src/dunhuang/research/content_r32/config.json"))
        self.io=self.old.io;general=self.old.reference;general.close();general.requires_grad_(False)
        change=TimeReference(4.).cuda();state=torch.load(path,map_location="cpu",weights_only=True)["state"]
        change.load_state_dict({k[len("reference."):]:v for k,v in state.items() if k.startswith("reference.")},strict=True)
        for k,v in self.io.state_dict().items():assert torch.equal(v.cpu(),state["io."+k])
        context=ContextBranch().cuda();context.requires_grad_(False)
        self.c=Controller(general,change,context);self.c.attach(self.d.model)
        self.d.model.forward=explicit_forward(self.io,self.old.original_forward)
        if checkpoint:
            value=torch.load(checkpoint,map_location="cpu",weights_only=True);change.load_state_dict(value["state"],strict=True)
        assert sum(p.numel() for p in change.parameters())==265328
        self.frozen=self.fingerprint()
    def fingerprint(self):
        return dict(backbone=digest(self.d.model),io=digest(self.io),general=digest(self.c.general),context=digest(self.c.context_branch))
    def check(self):
        assert self.fingerprint()==self.frozen
        assert all(p.grad is None for module in (self.d.model,self.io,self.c.general,self.c.context_branch) for p in module.parameters())
    def close(self):
        self.check();self.c.close();self.old.close()

def save_change(rt,path,cfg,arm,step):
    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in rt.c.change.state_dict().items()},config=cfg,arm=arm,step=step,frozen=rt.frozen),path)

def restore_change(rt,path):
    value=torch.load(path,map_location="cpu",weights_only=True);assert value["frozen"]==rt.frozen;rt.c.change.load_state_dict(value["state"],strict=True)
