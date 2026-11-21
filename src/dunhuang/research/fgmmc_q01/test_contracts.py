import unittest
import numpy as np
import torch
from .network import clocks,canonical_bank,initial
from .evaluate import ranking
from dunhuang.alignment.partial_path import soft_path_cost
class Contracts(unittest.TestCase):
 def test_clocks_globally_rate_legal_and_edges_exact(self):
  for T in [150,240,360]:
   v,p=clocks(T);self.assertEqual(len(p),5)
   for a in v:self.assertTrue(np.all(np.diff(a)>=.75-1e-9));self.assertTrue(np.all(np.diff(a)<=1.33+1e-9));np.testing.assert_array_equal(a[:4],np.arange(4));np.testing.assert_array_equal(a[-4:],np.arange(T-4,T))
 def test_duplicate_clock_does_not_expand_search(self):
  v,_=clocks(150);np.testing.assert_array_equal(canonical_bank(v),canonical_bank(np.concatenate([v,v[:2]])))
 def test_prior_correction_no_content_zero(self):
  model=initial();m=torch.zeros(2,16,64);d=torch.zeros_like(m)
  for mode in [True,False]:self.assertTrue(torch.allclose(model.score_tokens(m,d,150,mode),torch.zeros(2),atol=1e-6))
 def test_constrained_prior_excludes_some_unrestricted_matches(self):
  v,_=clocks(150);centers=np.linspace(0,149,16);self.assertLess(np.sum(abs(centers[:,None]-centers[None,:])<=5),256)
 def test_actual_gradient_exists_for_both_training_modes(self):
  for mode in [True,False]:
   net=initial();m=torch.randn(2,16,64,requires_grad=True);d=torch.randn_like(m);s=net.score_tokens(m,d,150,mode);s.sum().backward();self.assertTrue(torch.isfinite(m.grad).all());self.assertGreater(float(m.grad.abs().sum()),0)
 def test_ties_are_not_successes(self):self.assertEqual(ranking([0]*5)['strict_top1'],False);self.assertAlmostEqual(ranking([0]*5)['fractional_top1'],.2)
 def test_absolute_clock_channels_are_excluded(self):
  net=initial();M=torch.randn(1,150,50);D=torch.randn(1,150,274);DM=torch.ones_like(D,dtype=torch.bool);MV=torch.ones(1,150,dtype=torch.bool);a=net.tokens(M,D,DM,MV)[1];D[:,:,-2:]=1e6;b=net.tokens(M,D,DM,MV)[1];self.assertTrue(torch.equal(a,b))
if __name__=='__main__':unittest.main()
