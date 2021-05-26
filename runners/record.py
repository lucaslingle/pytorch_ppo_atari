from collections import deque
import os
import uuid
import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from utils.constants import ROOT_RANK
from utils.checkpoint_util import maybe_load_checkpoint


def _collect_footage(env, model, max_frames):
    queue = deque(maxlen=max_frames)
    t = 0
    total_reward = 0.0
    o_t = env.reset()
    while t < queue.maxlen:  # in general we could also make this larger than the queue
        pi_dist, vpred = model(tc.tensor(o_t).float().unsqueeze(0))
        a_t = pi_dist.sample()
        o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy())

        arr = env.render(mode='rgb_array')
        arr_fp = f"/tmp/{str(uuid.uuid4())}.png"
        plt.imsave(arr=arr, fname=arr_fp)
        queue.append(arr_fp)

        total_reward += r_t
        t += 1
        if done_t:
            break
        o_t = o_tp1

    return queue


def _make_video(queue, base_path, fps, max_frames):
    def make_frame(t):
        # t will range from 0 to (self.max_frames / self.fps).
        frac_done = t / (max_frames // fps)
        max_idx = len(queue) - 1
        idx = int(max_idx * frac_done)
        arr_fp = queue[idx]
        x = plt.imread(arr_fp)
        return (255 * x).astype(np.int32).astype(np.uint8)

    filename = f"{uuid.uuid4()}.gif"
    fp = os.path.join(base_path, filename)

    clip = mpy.VideoClip(make_frame, duration=(max_frames // fps))
    clip.write_gif(fp, fps=fps)
    return fp


def run(env, agent, args):
    if agent.comm.Get_rank() == ROOT_RANK:
        maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            agent=agent)

        base_path = os.path.join(args.asset_dir, args.model_name)
        os.makedirs(base_path, exist_ok=True)

        max_frames = 2048
        fps = 64
        queue = _collect_footage(env=env, model=agent.model, max_frames=max_frames)
        fp = _make_video(queue=queue, base_path=base_path, fps=fps, max_frames=max_frames)

        print(f"Saved video to {fp}")
