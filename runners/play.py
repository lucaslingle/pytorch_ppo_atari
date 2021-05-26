import torch as tc
from utils.checkpoint_util import maybe_load_checkpoint
from utils.constants import ROOT_RANK


@tc.no_grad()
def _play(env, agent, env_steps):
    t = 0
    total_reward = 0.0
    o_t = env.reset()
    while t < env_steps:
        _ = env.render()
        pi_dist, vpred = agent.model(tc.FloatTensor(o_t).unsqueeze(0))
        a_t = pi_dist.sample()
        o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy())
        total_reward += r_t
        t += 1
        if done_t:
            print(f"Episode finished after {t} timesteps.")
            print(f"Total reward was {total_reward}.")
            break
        o_t = o_tp1


def run(env, agent, args):
    if agent.comm.Get_rank() == ROOT_RANK:
        maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            agent=agent)

    _play(env=env, agent=agent, env_steps=args.env_steps)
