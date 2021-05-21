import torch as tc


def standardize(arr):
    eps = 1e-8
    mu = arr.mean(keepdim=True)
    sigma = arr.stddev(keepdim=True)
    standardized = (arr - mu) / (eps + sigma)
    return standardized
