import torch as tc
from agents.policy_head import PolicyHead
from agents.value_head import ValueHead
from agents.preprocess import ConvPreprocess
from utils.init_util import normc_initializer


class NatureCNN(tc.nn.Module):
    """
    Implements the convolutional torso of the agent from Mnih et al., 2015
    - 'Human Level Control through Deep Reinforcement Learning'.
    """
    def __init__(self, hparams):
        super().__init__()
        self._feature_dim = 512
        self.conv_stack = tc.nn.Sequential(
            tc.nn.Conv2d(hparams.img_chan, 32, kernel_size=(8,8), stride=(4,4)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1)),
            tc.nn.ReLU(),
            tc.nn.Flatten(),
            tc.nn.Linear(3136, self.feature_dim),
            tc.nn.ReLU()
        )
        self.initialize_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def initialize_weights(self):
        for m in self.conv_stack.modules():
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.xavier_uniform_(m.weight)
                tc.nn.init.zeros_(m.bias)
            elif isinstance(m, tc.nn.Linear):
                normc_initializer(m.weight, gain=1.0)
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_stack(x)


class AsyncCNN(tc.nn.Module):
    """
    Implements the convolutional torso of the agent from Mnih et al., 2016
    - 'Asynchronous Methods for Deep Reinforcement Learning'.
    """
    def __init__(self, hparams):
        super().__init__()
        self._feature_dim = 256
        self.conv_stack = tc.nn.Sequential(
            tc.nn.Conv2d(hparams.img_chan, 16, kernel_size=(8,8), stride=(4,4)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2)),
            tc.nn.ReLU(),
            tc.nn.Flatten(),
            tc.nn.Linear(2592, self._feature_dim),
            tc.nn.ReLU()
        )
        self.initialize_weights()

    @property
    def feature_dim(self):
        return self._feature_dim

    def initialize_weights(self):
        for m in self.conv_stack.modules():
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.xavier_uniform_(m.weight)
                tc.nn.init.zeros_(m.bias)
            elif isinstance(m, tc.nn.Linear):
                normc_initializer(m.weight, gain=1.0)
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_stack(x)


class CnnPolicy(tc.nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.preprocessor = ConvPreprocess()
        if hparams.model_size == 'small':
            self.conv_stack = AsyncCNN(hparams)
        elif hparams.model_size == 'large':
            self.conv_stack = NatureCNN(hparams)

        self.feature_dim = self.conv_stack.feature_dim
        self.num_actions = hparams.num_actions

        self.policy_head = PolicyHead(self.feature_dim, self.num_actions)
        self.value_head = ValueHead(self.feature_dim)

    def forward(self, x):
        x = self.preprocessor(x)
        features = self.conv_stack(x)
        dist_pi = self.policy_head(features)
        vpred = self.value_head(features)
        return dist_pi, vpred