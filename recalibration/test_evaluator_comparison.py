"""Focused boundary-atom/CDF tests for the new evaluator diagnostics."""
import numpy as np
from scipy.stats import ks_2samp
from evaluator_comparison import ks,tail,weighted_quantile

def test_atomic_tails():
    x=np.array([0.,10.,30.]);p=np.array([.02,.96,.02])
    assert weighted_quantile(x,p,.05)==10
    assert abs(tail(x,p)-6)<1e-12  # only 0.03 of the 0.96 boundary atom
    assert abs(tail(x[::-1],p[::-1])-18)<1e-12

def test_cdf_left_right_and_ties():
    x=np.array([0.,1.,3.]);p=np.array([.2,.5,.3])
    equivalent=np.repeat(x,[2,5,3]);other=np.array([0.,0.,0.,1.,2.,2.,3.,3.,4.,4.])
    assert ks(x,p,equivalent)<1e-14
    assert abs(ks(x,p,other)-ks_2samp(equivalent,other).statistic)<1e-14
    assert ks(np.array([0.]),np.array([1.]),np.array([0.,0.]))==0
if __name__=='__main__':
    test_atomic_tails();test_cdf_left_right_and_ties();print('CDF left/right limits and fractional-tail tests passed')
