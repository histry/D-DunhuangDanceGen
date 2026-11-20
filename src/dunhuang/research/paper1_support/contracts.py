"""Small, explicit Paper 1 exchange objects; no formal-generator dependency."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class MusicEvidence:
    audio_sha256: str
    interval: tuple
    fps: int = 30
    annotation_status: str = 'AUTOMATIC_NOT_GROUND_TRUTH'

@dataclass(frozen=True)
class SourcePrimitive:
    event_uid: str
    source_uid: str
    source_interval: tuple
    source_sha256: str
    semantic_completeness: str = 'UNKNOWN'
    world_root_contact: str = 'UNKNOWN'

@dataclass(frozen=True)
class ExecutionCandidate:
    primitive: SourcePrimitive
    music: MusicEvidence
    target_frames: int
    time_warp: tuple
    def validate(self):
        from dunhuang.research.warp_coupling_r53.warp_lattice import validate
        validate(self.time_warp, self.primitive.source_interval, [0,self.target_frames])
        if self.music.fps != 30 or self.music.interval[1]-self.music.interval[0] != self.target_frames:
            raise ValueError('Music and Core must use the same actual 30 FPS support')
        return self

@dataclass(frozen=True)
class ExecutionAudit:
    candidate_uid: str
    rotation_sha256: str
    bridge_screen: bool
    quality: str = 'NOT_EVALUABLE'
    native24: str = 'NOT_EVALUABLE'

@dataclass(frozen=True)
class SelectionDecision:
    candidate_uid: str
    rule: str
    scope: str = 'MECHANISM_DIAGNOSTIC_ONLY'
    can_route: bool = False

def payload(value):
    return asdict(value)

def boundary_unchanged(q, baseline, R, old):
    ids=np.r_[0:4,len(q)-4:len(q)]
    return bool(np.array_equal(q[ids],baseline[ids]) and np.array_equal(R[ids],old[ids]))
