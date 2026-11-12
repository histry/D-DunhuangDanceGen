"""Observe actual forward calls, including guided_forward -> self.forward."""
class ForwardObserver:
    def __init__(self,module,callback):
        self.module=module;self.callback=callback;self.original=module.forward
        self.had_override="forward" in module.__dict__;self.active=True
        def observed(*args,**kwargs):
            callback(module,args,kwargs)
            return self.original(*args,**kwargs)
        module.forward=observed
    def close(self):
        if not self.active:return
        if self.had_override:self.module.forward=self.original
        else:delattr(self.module,"forward")
        self.active=False
