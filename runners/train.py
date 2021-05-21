import torch as tc
from runners.runner import Runner
from utils.checkpoint_util import save_checkpoint, maybe_load_checkpoint
from utils.comm_util import sync_params, sync_grads
from utils.stat_util import standardize
from utils.dataset_util import Dataset



class Trainer(Runner):
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.args = args

    @staticmethod
    @tc.no_grad()
    def trajectory_segment_generator(env, model, timesteps_per_actorbatch):
        t = 0
        o_t = env.reset()

        episode_returns = []
        episode_lengths = []
        current_episode_return = 0.0
        current_episode_length = 0

        observations = []
        logprobs = []
        actions = []
        rewards = []
        dones = []
        value_estimates = []

        model.eval()
        while True:
            pi_dist_t, vpred_t = model(tc.FloatTensor(o_t).unsqueeze(0))
            a_t = pi_dist_t.sample()
            logprob_a_t = pi_dist_t.log_prob(a_t)

            a_t = a_t.squeeze(0).detach()
            logprob_a_t = logprob_a_t.squeeze(0).detach()
            vpred_t = vpred_t.squeeze(0).detach()

            if t > 0 and t % timesteps_per_actorbatch == 0:
                value_estimates.append(vpred_t)
                yield {
                    "observations": observations,
                    "logprobs": logprobs,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                    "value_estimates": value_estimates  # length is timesteps_per_actorbatch+1 for GAE.
                }
                observations = []
                logprobs = []
                actions = []
                rewards = []
                dones = []
                value_estimates = []

            o_tp1, r_t, done_t, _ = env.step(a_t.numpy())

            observations.append(o_t)
            logprobs.append(logprob_a_t)
            actions.append(a_t)
            rewards.append(r_t)
            dones.append(done_t)
            value_estimates.append(vpred_t)

            if done_t:
                episode_returns.append(current_episode_return)
                episode_lengths.append(current_episode_length)
                current_episode_return = 0.0
                current_episode_length = 0
                o_tp1 = env.reset()

            t += 1
            o_t = o_tp1

    @staticmethod
    @tc.no_grad()
    def add_vtarg_and_adv(seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        T = len(seg['actions'])
        advantages = tc.zeros(size=(T+1,))
        for t in reversed(range(1, T+1)):  # T, ..., 1.
            done_t = seg['dones'][t-1]
            r_t = seg['rewards'][t-1]
            V_t = seg['value_estimates'][t-1]
            V_tp1 = seg['value_estimates'][t]

            delta_t = -V_t + r_t + gamma * (1.-float(done_t)) * V_tp1
            advantages[t-1] = delta_t + gamma * lam * (1.-float(done_t)) * advantages[t]

        seg["advantage_estimates"] = advantages[0:-1]
        seg["value_estimates"] = seg["value_estimates"][0:-1]
        seg["td_lambda_returns"] = seg["advantage_estimates"] + seg["value_estimates"]
        return seg

    @staticmethod
    def compute_losses(model, batch):
        raise NotImplementedError

    @staticmethod
    def train(env, agent, args):
        sync_params(model=agent.model, comm=agent.comm)
        seg_generator = Trainer.trajectory_segment_generator(
            env=env, model=agent.model, timesteps_per_actorbatch=args.timesteps_per_actorbatch)

        env_steps_so_far = 0
        while env_steps_so_far < args.env_steps:
            seg = next(seg_generator)
            seg = Trainer.add_vtarg_and_adv(seg, gamma=args.discount_gamma, lam=args.gae_lambda)
            seg['advantage_estimates'] = standardize(seg['advantage_estimates'])
            dataset = Dataset(data_map={
                'obs': seg['observations'],
                'acs': seg['actions'],
                'logprobs': seg['logprobs'],
                'vtargs': seg['td_lambda_returns'],
                'adv': seg['advantage_estimates']
            })
            for _ in range(args.optim_epochs):
                for batch in dataset.iterate_once(batch_size=args.optim_batchsize):
                    agent.optimizer.zero_grad()
                    losses = Trainer.compute_losses(model=agent.model, batch=batch)
                    losses['total_loss'].backward()
                    sync_grads(model=agent.model, comm=agent.comm)
                    agent.optimizer.step()
                    agent.scheduler.step()

            env_steps_so_far += args.timesteps_per_actorbatch * agent.comm.Get_size()

    def run(self):
        maybe_load_checkpoint(
            checkpoint_dir=self.args.checkpoint_dir,
            model_name=self.args.model_name,
            agent=self.agent
        )

        self.train(
            env=self.env,
            agent=self.agent,
            args=self.args
        )
