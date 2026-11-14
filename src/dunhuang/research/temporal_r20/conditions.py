"""Endpoint controls preserve a common middle evaluation mask, never score pasted ends."""
from dunhuang.research.global_clock_r17.engine import observed

CONTROLS=("correct","left_wrong","right_wrong","left_absent","right_absent","both_absent")
def endpoint(y,donor,control,hard=True):
    if control not in CONTROLS:raise ValueError("Endpoint control")
    value,known,fixed=observed(y,"inbetween",hard=hard);original=known.clone()
    for side,span in (("left",slice(0,8)),("right",slice(217,225))):
        if control==side+"_wrong":value[:,span,4:]=donor[:,span,4:]
        if control in (side+"_absent","both_absent"):
            value[:,span,4:]=0;known[:,span,4:]=False;fixed[:,span,4:]=False
    return value,known,fixed,original
