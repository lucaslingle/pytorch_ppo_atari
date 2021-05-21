import torch as tc
from runners.runner import Runner
from runners.constants import ROOT_RANK
from utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
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
    def __trajectory_segment_generator(env, model, timesteps_per_actorbatch):
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
    def __add_vtarg_and_adv(seg, gamma, lam):
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
    def __compute_losses(model, batch, clip_param, entcoeff):
        # get relevant info from minibatch dict
        mb_obs = batch["obs"]
        mb_acs = batch["acs"]
        mb_logpi_old = batch["logprobs"]
        mb_advs = batch["advs"]
        mb_vtargs = batch["vtargs"]

        # cast to correct type
        mb_obs = tc.FloatTensor(mb_obs).detach()
        mb_acs = tc.LongTensor(mb_acs).detach()
        mb_logpi_old = tc.FloatTensor(mb_logpi_old).detach()
        mb_advs = tc.FloatTensor(mb_advs).detach()
        mb_vtargs = tc.FloatTensor(mb_vtargs).detach()

        # evaluate observations using agent
        mb_pi_dist, mb_vpred_new = model(mb_obs)
        mb_logpi_new = mb_pi_dist.log_prob(mb_acs)

        # entropy
        ent = mb_pi_dist.entropy()
        meanent = tc.mean(ent)
        pol_entpen = (-entcoeff) * meanent

        # ppo policy loss
        policy_ratio = tc.exp(mb_logpi_new - mb_logpi_old)
        clipped_policy_ratio = tc.clip(policy_ratio, 1.0 - clip_param, 1.0 + clip_param)
        surr1 = mb_advs * policy_ratio
        surr2 = mb_advs * clipped_policy_ratio
        pol_surr = -tc.mean(tc.min(surr1, surr2))

        # ppo value loss
        vf_loss = tc.mean(tc.square(mb_vtargs - mb_vpred_new))

        # total loss
        total_loss = pol_surr + pol_entpen + vf_loss

        return {
            "pol_surr": pol_surr,
            "pol_entpen": pol_entpen,
            "vf_loss": vf_loss,
            "meanent": meanent,
            "total_loss": total_loss
        }

    @staticmethod
    def __train(env, agent, args):
        sync_params(model=agent.model, comm=agent.comm, root=ROOT_RANK)
        seg_generator = Trainer.__trajectory_segment_generator(
            env=env, model=agent.model, timesteps_per_actorbatch=args.timesteps_per_actorbatch)

        env_steps_so_far = 0
        while env_steps_so_far < args.env_steps:
            seg = next(seg_generator)
            seg = Trainer.__add_vtarg_and_adv(seg, gamma=args.discount_gamma, lam=args.gae_lambda)
            seg['advantage_estimates'] = standardize(seg['advantage_estimates'])
            dataset = Dataset(data_map={
                'obs': seg['observations'],
                'acs': seg['actions'],
                'logprobs': seg['logprobs'],
                'vtargs': seg['td_lambda_returns'],
                'advs': seg['advantage_estimates']
            })
            for _ in range(args.optim_epochs):
                for batch in dataset.iterate_once(batch_size=args.optim_batchsize):
                    agent.optimizer.zero_grad()
                    losses = Trainer.__compute_losses(
                        model=agent.model, batch=batch, clip_param=args.ppo_epsilon,
                        entcoeff=args.entropy_coef)
                    losses['total_loss'].backward()
                    sync_grads(model=agent.model, comm=agent.comm)
                    agent.optimizer.step()
                    agent.scheduler.step()

            env_steps_so_far += args.timesteps_per_actorbatch * agent.comm.Get_size()

    def run(self):
        if self.agent.comm.Get_rank() == ROOT_RANK:
            maybe_load_checkpoint(
                checkpoint_dir=self.args.checkpoint_dir,
                model_name=self.args.model_name,
                agent=self.agent
            )

        self.__train(
            env=self.env,
            agent=self.agent,
            args=self.args
        )
