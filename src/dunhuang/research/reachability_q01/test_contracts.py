import unittest
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from dunhuang.planning.anchor_reachability import single,multiple

class Reachability(unittest.TestCase):
    def test_declared_failed_case_exact_bound(self):
        r=single([4153,4446],4311,240,64.2,3);self.assertEqual(r['continuous_reachable_interval'],[118.5,138.5]);self.assertAlmostEqual(r['minimum_target_error_lower_bound'],54.3);self.assertEqual(r['status'],'EXCLUDED_BY_RATE')
    def test_additional_constraints_never_promoted(self):
        r=single([0,101],50,101,50,3,additional_constraints=['G1']);self.assertEqual(r['rate_status'],'REACHABLE_BY_RATE');self.assertEqual(r['status'],'UNKNOWN_WITH_ADDITIONAL_CONSTRAINTS')
    def test_endpoint_and_one_frame_errors(self):
        self.assertEqual(single([0,101],100,101,100)['continuous_reachable_interval'],[100.,100.]);self.assertEqual(single([0,101],50,76,37.5)['continuous_reachable_interval'],[37.5,37.5])
        with self.assertRaises(ValueError):single([0,1],0,5,0)
    def test_invalid_inputs_fail_closed(self):
        for a in [float('nan'),float('inf'),True,-1,101]:
            with self.assertRaises(ValueError):single([0,101],a,101,50)
        with self.assertRaises(ValueError):single([0,101],50,101,50,rates=[1,.5])
    def test_duration_infeasibility_has_no_fake_interval(self):
        r=single([0,101],50,10,5);self.assertIsNone(r['continuous_reachable_interval']);self.assertEqual(r['rate_status'],'EXCLUDED_BY_RATE')
    def test_marginal_reachability_is_not_joint(self):
        self.assertEqual(single([0,101],40,101,40)['rate_status'],'REACHABLE_BY_RATE');self.assertEqual(single([0,101],60,101,70)['rate_status'],'REACHABLE_BY_RATE');r=multiple([0,101],101,[dict(source_frame=40,requested=40),dict(source_frame=60,requested=70)]);self.assertEqual(r['rate_status'],'EXCLUDED_BY_RATE')
    def test_multi_witness_and_duplicate_constraints(self):
        r=multiple([0,101],101,[dict(source_frame=40,requested=40,tolerance=3),dict(source_frame=60,requested=60,tolerance=3)]);self.assertEqual(r['rate_status'],'REACHABLE_BY_RATE');w=np.asarray(r['witness_clock']);self.assertTrue(np.all(np.diff(w[:,1])/np.diff(w[:,0])>=.75))
        r=multiple([0,101],101,[dict(source_frame=50,requested=40),dict(source_frame=50,requested=60)]);self.assertEqual(r['rate_status'],'EXCLUDED_BY_RATE')
    def test_fixed_guard_can_shrink_domain(self):
        r=multiple([0,101],101,[dict(source_frame=50,requested=45)],fixed_knots=[(49,49)]);self.assertEqual(r['rate_status'],'EXCLUDED_BY_RATE')
    def test_single_interval_matches_independent_linear_program(self):
        rng=np.random.default_rng(78001)
        for _ in range(80):
            n=int(rng.integers(20,500));s=int(rng.integers(1,n));D=int(rng.integers(max(2,int(.75*n)),int(1.33*n)+2));r=single([0,n+1],s,D+1,D/2)
            A=[[-1],[1],[1],[-1]];b=[-.75*s,1.33*s,D-.75*(n-s),1.33*(n-s)-D]
            lo=linprog([1],A_ub=A,b_ub=b,bounds=[(None,None)],method='highs');hi=linprog([-1],A_ub=A,b_ub=b,bounds=[(None,None)],method='highs')
            self.assertEqual(lo.success,r['continuous_reachable_interval'] is not None)
            if lo.success:self.assertTrue(np.allclose([lo.x[0],hi.x[0]],r['continuous_reachable_interval'],atol=1e-7,rtol=0))
    def test_joint_chain_matches_independent_linear_program(self):
        rng=np.random.default_rng(78002)
        for _ in range(80):
            xs=np.array([0,20,55,80,100]);target=rng.uniform(5,95,3);tol=3;anchors=[dict(source_frame=int(x),requested=float(t),tolerance=tol) for x,t in zip(xs[1:-1],target)];r=multiple([0,101],101,anchors);A=[];b=[]
            for i,dx in enumerate(np.diff(xs)):
                row=np.zeros(5);row[i+1]=1;row[i]=-1;A.extend([row,-row]);b.extend([1.33*dx,-.75*dx])
            res=linprog(np.zeros(5),A_ub=A,b_ub=b,bounds=[(0,0)]+[(max(0,t-tol),min(100,t+tol)) for t in target]+[(100,100)],method='highs');self.assertEqual(res.success,r['rate_status']=='REACHABLE_BY_RATE')
if __name__=='__main__':unittest.main()
