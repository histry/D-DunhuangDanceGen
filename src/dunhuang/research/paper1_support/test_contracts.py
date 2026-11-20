import unittest
from .contracts import MusicEvidence,SourcePrimitive,ExecutionCandidate
from .study import clocks,duplicate_clock
from dunhuang.research.warp_coupling_r53.warp_lattice import validate
class Contracts(unittest.TestCase):
    def test_roundoff_is_not_a_new_execution(self):
        import numpy as np
        q=np.linspace(100,382,240)
        self.assertTrue(duplicate_clock(q+1e-12,{'base':(q,None)}))
        self.assertFalse(duplicate_clock(q+1e-5,{'base':(q,None)}))
    def test_reject_clock_and_music_mismatch(self):
        p=SourcePrimitive('e','s',(0,240),'x')
        with self.assertRaises(ValueError):ExecutionCandidate(p,MusicEvidence('x',(0,239)),240,((0,0),(239,239))).validate()
        with self.assertRaises(ValueError):ExecutionCandidate(p,MusicEvidence('x',(0,240)),240,((0,0),(120,180),(239,239))).validate()
    def test_finite_pool_preserves_edges_and_changes_internal_hold(self):
        import numpy as np
        ws=clocks({'source_interval':[100,383]},{'cells_half_open':[200,260]},240);holds=[];base=np.linspace(100,382,240)
        for w in ws:
            validate(w,[100,383],[0,240]);q=np.interp(np.arange(240),w[:,1],w[:,0]);self.assertTrue(np.allclose(q[:4],base[:4],atol=1e-10));self.assertTrue(np.allclose(q[-4:],base[-4:],atol=1e-10));holds.append(np.diff(np.interp([200,260],w[:,0],w[:,1]))[0])
        self.assertGreater(max(holds)-min(holds),1)
if __name__=='__main__':unittest.main()
