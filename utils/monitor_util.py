"""Implements a monitoring function that wraps the environment.
   similar to baselines we will put this monitoring wrapper
   between the make_atari env wrapper and the wrap_deepmind env wrapper.

   Ideally, this would also append to the info dict an 'unclipped_reward' kv pair,
   so that the unclipped returns could be printed out by the logger later.
"""

from gym.core import Wrapper
import os
import time
from mpi4py import MPI


class ResultWriter:
    def __init__(self, monitoring_file, column_names):
        """
        A helper class that writes rows to a CSV.

        :param monitoring_file: filepath to the monitoring CSV.
        :param column_names: a list of column names.
        """
        self.monitoring_file = monitoring_file
        self.column_names = sorted(column_names)
        self.write_column_names()

    def write_column_names(self):
        with open(self.monitoring_file, 'w') as f:
            line = ", ".join(self.column_names)
            f.write(line)

    def write_row(self, row_dict):
        """
        :param row: dict of column names and values to write.
        :return:
        """
        if len(row_dict) != len(self.column_names):
            raise RuntimeError("Wrong number of columns!")

        with open(self.monitoring_file, 'w') as f:
            line = ", ".join([row_dict[col] for col in self.column_names])
            f.write(line)


class Monitor(Wrapper):
    def __init__(self, env, monitoring_dir, model_name, comm):
        """
        Initialize a monitor for an environment. Throughout training,
        a Monitor instance will write out important quantities,
        like the unclipped reward per episode, to a log file.

        :param env: a gym.core.Env object or a gym.core.Wrapper object.
        :param monitoring_dir: directory to store monitoring logs.
        :param model_name: model name used to associate monitoring logs with correct model
        """

        super().__init__(env)
        self.monitoring_dir = os.path.join(monitoring_dir, model_name)
        self.monitoring_file = os.path.join(
            self.monitoring_dir, f"monitoring_rank{comm.Get_rank()}.csv")
        os.makedirs(self.monitoring_dir, exist_ok=True)

        self.metric_names = [
            "env_steps",
            "episode_len",
            "episode_rew",
            "episode_sec"
        ]
        self.writer = ResultWriter(
            self.monitoring_file, column_names=self.metric_names)

        self.comm = comm
        self.local_steps = 0
        self.env_steps = 0
        self.stats = None

    def reset(self):
        ob = self.env.reset()
        self.comm.allreduce(self.local_steps, self.env_steps, op=MPI.SUM)
        self.stats = {k: 0.0 for k in self.metric_names}
        self.stats['env_steps'] = self.env_steps  # env steps at start of episode
        self.stats['episode_starttime'] = time.perf_counter()
        return ob

    def step(self, a):
        ob, rew, done, info = self.env.step(a)
        info['monitored_reward'] = rew

        self.local_steps += 1
        self.stats['episode_len'] += 1
        self.stats['episode_rew'] += rew
        if done:
            now = time.perf_counter()
            delta = now - self.stats['episode_starttime']
            self.stats['episode_sec'] = delta
            del self.stats['episode_starttime']
            self.writer.write_row(self.stats)

        return ob, rew, done, info
