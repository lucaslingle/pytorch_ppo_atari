import typing
import numpy as np


class MutableExperienceTrajectory:
    def __init__(self, horizon, obs_shape):
        self._horizon = horizon
        self._obs_shape = obs_shape

        self._observations = np.zeros([horizon, *obs_shape], 'float32')
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
        return self._observations

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
        self.episode_return = []
        self.episode_return_unclipped = []