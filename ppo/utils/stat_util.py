import torch as tc
import numpy as np


def standardize(arr):
    eps = 1e-8
    mu = arr.mean()
    sigma = arr.std()
    standardized = (arr - mu) / (eps + sigma)
    return standardized


def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted a constant
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting a constant
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = y.var()
    return 1 - (y-ypred).var()/(1e-8 + vary)