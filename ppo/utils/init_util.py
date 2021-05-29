import torch as tc
import numpy as np


def normc_initializer(weight_tensor, gain=1.0):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L97

    Note that in tensorflow the weight tensor in a linear layer is stored with the
    input dim first and the output dim second. See
    https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/layers/core.py#L1193

    In contrast, in pytorch the output dim is first. See
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

    This means if we want a normc init in pytorch,
    we have to change which dim(s) we normalize on.

    We currently only support normc init for linear layers.
    Performance not guaranteed with other layer weight types.
    """
    with tc.no_grad():
        out = np.random.normal(loc=0.0, scale=1.0, size=weight_tensor.size())
        out = gain * out / np.sqrt(np.sum(np.square(out), axis=1, keepdims=True))
        weight_tensor.copy_(tc.tensor(out))
