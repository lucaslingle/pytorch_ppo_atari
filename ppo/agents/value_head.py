import torch as tc
from ppo.utils.init_util import normc_initializer


class ValueHead(tc.nn.Module):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py#L38
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.value_head = tc.nn.Sequential(
            tc.nn.Linear(self.feature_dim, 1)
        )

        for m in self.value_head.modules():
            if isinstance(m, tc.nn.Linear):
                normc_initializer(m.weight, gain=1.0)
                tc.nn.init.zeros_(m.bias)

    def forward(self, features):
        return self.value_head(features).squeeze(-1)