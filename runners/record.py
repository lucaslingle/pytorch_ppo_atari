import torch as tc
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt
import uuid
import moviepy.editor as mpy
from runners.runner import Runner
from runners.constants import ROOT_RANK
from utils.checkpoint_util import maybe_load_checkpoint


class Recorder(Runner):
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.args = args

        self.base_path = os.path.join(args.asset_dir, args.model_name)
        os.makedirs(self.base_path, exist_ok=True)

        self.fps = 64
        self.max_frames = 2048
        self.queue = deque(maxlen=self.max_frames)

    def _make_video(self):
        def make_frame(t):
            # t will range from 0 to (self.max_frames / self.fps).
            frac_done = t / (self.max_frames // self.fps)
            max_idx = len(self.queue) - 1
            idx = int(max_idx * frac_done)
            arr_fp = self.queue[idx]
            x = plt.imread(arr_fp)
            return (255 * x).astype(np.int32).astype(np.uint8)

        filename = f"{uuid.uuid4()}.gif"
        fp = os.path.join(self.base_path, filename)

        clip = mpy.VideoClip(make_frame, duration=(self.max_frames // self.fps))
        clip.write_gif(fp, fps=self.fps)

        print(f"Saving video to {fp}")

    def _collect_footage(self, env, agent):
        t = 0
        total_reward = 0.0
        o_t = env.reset()
        while t < self.queue.maxlen:
            pi_dist, vpred = agent(tc.tensor(o_t).float().unsqueeze(0))
            a_t = pi_dist.sample()
            o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy())

            arr = env.render(mode='rgb_array')
            arr_fp = f"/tmp/{str(uuid.uuid4())}.png"
            plt.imsave(arr=arr, fname=arr_fp)
            self.queue.append(arr_fp)

            total_reward += r_t
            t += 1
            if done_t:
                break
            o_t = o_tp1

    def run(self):
        if self.agent.comm.Get_rank() == ROOT_RANK:
            maybe_load_checkpoint(
                checkpoint_dir=self.args.checkpoint_dir,
                model_name=self.args.model_name,
                agent=self.agent
            )

            self._collect_footage(
                env=self.env,
                agent=self.agent
            )

            self._make_video()

