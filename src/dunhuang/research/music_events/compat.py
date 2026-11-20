"""Process-local legacy API aliases; official weights and model graph unchanged."""
def install():
    import natten.functional as f
    if not hasattr(f,'natten1dqkrpb'):
        # Same semantics as the deprecated wrappers in official NATTEN v0.15.1.
        f.natten1dqkrpb=lambda q,k,rpb,kernel_size,dilation:f.na1d_qk(q,k,kernel_size=kernel_size,dilation=dilation,rpb=rpb)
        f.natten2dqkrpb=lambda q,k,rpb,kernel_size,dilation:f.na2d_qk(q,k,kernel_size=kernel_size,dilation=dilation,rpb=rpb)
        f.natten1dav=f.na1d_av
        f.natten2dav=f.na2d_av
