"""Canonical audible time, independent of motion-window frame origins."""
from fractions import Fraction
import math

SAMPLE_RATE=44100

def music_seconds(row):
    start=row["start_frame"];offset=row["audio_offset_seconds"]
    if type(start) is not int or start<0 or row["fps"]!=30 or not math.isfinite(offset):raise ValueError("Verified finite physical clock required")
    result=Fraction(start,30)+Fraction(str(offset))
    if result<0:raise ValueError("Negative audible source clock")
    return result

def music_sample(row):
    x=music_seconds(row)*SAMPLE_RATE
    return round(x)

def same_music_time(a,b):
    return a["music_uid"]==b["music_uid"] and a["audio"]["sha256"]==b["audio"]["sha256"] and abs(music_seconds(a)-music_seconds(b))<=Fraction(1,2*SAMPLE_RATE)
