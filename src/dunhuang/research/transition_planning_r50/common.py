from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import Library, read
from dunhuang.research.event_alignment_r49.common import Frozen, audio_signals, sealed
from dunhuang.research.event_alignment_r49.rank import CATALOG, SONGS

ROOT = Path("outputs/transition_planning_r50_01")


def assets():
    net = Frozen()
    from dunhuang.research.song_transfer.music import runtime
    runtime(read("inputs/clock_assets_v2.json"))
    from evaluation.paired_alignment_observations_v3 import full_audio_observations
    lib = Library(read(CATALOG)["catalog"])
    units = read(sealed(Path("outputs/event_coverage_r47_01"),"index/result.json"))["units"]
    songs = []
    for i in range(3):
        path = sealed(SONGS,f"viewer/audio/dunhuangwu{i+2}.wav")
        values,mask,_ = full_audio_observations(path)
        structure = read(sealed(SONGS,f"planning/song{i}_music.json"))
        songs.append(dict(path=path,values=values,mask=mask,signal=audio_signals(path),structure=structure))
    return net,lib,units,songs


def layout(start, durations, transitions):
    if len(transitions)!=len(durations)-1 or min(durations)<2 or any(t<1 for t in transitions):
        raise ValueError("Positive actual Core and Transition durations required")
    regions = []
    cursor = start
    for i,d in enumerate(durations):
        regions.append(dict(kind="core",interval=[cursor,cursor+d]))
        cursor += d
        if i<len(transitions):
            t = transitions[i]
            regions.append(dict(kind="transition",interval=[cursor,cursor+t]))
            cursor += t
    assert cursor-start == sum(durations)+sum(transitions)
    return regions
