import torch as tc
from runners.runner import Runner
from utils.checkpoint_util import maybe_load_checkpoint


class Player(Runner):
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.args = args

    @staticmethod
    @tc.no_grad()
    def play(env, agent, env_steps):
        if agent.comm.Get_rank() == 0:
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
                    break
                o_t = o_tp1

            print(f"Episode finished after {t} timesteps.")
            print(f"Total reward was {total_reward}.")

    def run(self):
        maybe_load_checkpoint(
            checkpoint_dir=self.args.checkpoint_dir,
            model_name=self.args.model_name,
            agent=self.agent
        )

        self.play(
            env=self.env,
            agent=self.agent,
            env_steps=self.args.env_steps
        )
