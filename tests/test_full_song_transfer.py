import numpy as np
import pytest
from dunhuang.research.song_transfer.structure import timeline
from dunhuang.music import clock_plan
CFG=dict(min_frames=50,max_frames=150,target_frames=90,grid_frames=30,novelty_weight=.5,phrase_weight=.75,beat_weight=.25,phrase_distance=180)
def test_structure_covers_every_full_song_frame_with_legal_slot_lengths():
    frames=3601
    legacy=dict(frames=frames,fps=30,beats=list(range(0,frames,15)),slots=[dict(start=0,end=frames,density=.6)])
    music=np.zeros((frames,50));music[:,18:]=np.sin(np.arange(frames)[:,None]/21);music[500:900,18:]+=3
    value=timeline(legacy,music,CFG);slots=value["slots"]
    assert slots[0]["start"]==0 and slots[-1]["end"]==frames
    assert all(a["end"]==b["start"] for a,b in zip(slots,slots[1:]))
    assert all(50<=s["end"]-s["start"]<=150 for s in slots)
    assert value["phrase_ground_truth"] is False if "phrase_ground_truth" in value else "not independent" in value["phrase_method"]
    assert len(slots)!=clock_plan(1920523,16000)["windows"]
def test_musical_slots_are_not_independent_edge_generation_windows():
    clock=clock_plan(1920523,16000)
    assert clock["frames"]==3601 and clock["windows"]==48 and clock["hop"]==75 and clock["horizon"]==150
    assert clock["covered_frames"]>=clock["frames"] and clock["starts"][-1]+150==clock["covered_frames"]
def test_structure_rejects_truncated_audio_observations():
    with pytest.raises(ValueError):timeline(dict(frames=3601,beats=[],slots=[]),np.zeros((150,50)),CFG)
