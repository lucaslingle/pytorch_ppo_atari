import typing
import numpy as np


class MutableExperienceTrajectory:
    def __init__(self, horizon, dummy_obs):
        self._horizon = horizon

        self._observations = np.array([dummy_obs for _ in range(horizon)])
        self._actions = np.zeros([horizon], 'int64')
        self._rewards = np.zeros([horizon], 'float32')
        self._dones = np.zeros([horizon], 'float32')
        self._logprobs = np.zeros([horizon], 'float32')
        self._value_estimates = np.zeros([horizon+1], 'float32')

        self._advantage_estimates = np.zeros(horizon, 'float32')
        self._td_lambda_returns = np.zeros(horizon, 'float32')

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    @property
    def dones(self):
        return self._dones

    @property
    def logprobs(self):
        return self._logprobs

    @property
    def value_estimates(self):
        return self._value_estimates

    @property
    def advantage_estimates(self):
        return self._advantage_estimates

    @property
    def td_lambda_returns(self):
        return self._td_lambda_returns


class TrajectoryMetrics:
    def __init__(self):
        self.current_episode_length = 0
        self.current_episode_return = 0.0
        self.current_episode_return_unclipped = 0.0

        self.episode_lengths = []
        self.episode_returns = []
        self.episode_returns_unclipped = []
