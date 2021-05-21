import torch as tc
from utils.init_util import normc_initializer


class PolicyHead(tc.nn.Module):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py#L36
    """
    def __init__(self, feature_dim, num_actions):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.policy_head = tc.nn.Sequential(
            tc.nn.Linear(self.feature_dim, self.num_actions)
        )
        for m in self.policy_head.modules():
            if isinstance(m, tc.nn.Linear):
                normc_initializer(m.weight, gain=0.01)
                tc.nn.init.zeros_(m.bias)

    def forward(self, features):
        logits = self.policy_head(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist
