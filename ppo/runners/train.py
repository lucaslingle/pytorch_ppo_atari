"""
Started with https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
and ported it to Pytorch, removing all dependencies on baselines modules along the way.
"""

import torch as tc
import numpy as np
from mpi4py import MPI
from collections import deque
from ppo.utils.constants import ROOT_RANK
from ppo.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from ppo.utils.comm_util import sync_state, sync_grads
from ppo.utils.stat_util import standardize, explained_variance
from ppo.utils.dataset_util import Dataset
from ppo.utils.print_util import print_metrics


@tc.no_grad()
def _trajectory_segment_generator(env, model, timesteps_per_actorbatch):
    """
    Generates trajectory segments, maintaining environment state across segments,
    and resetting the environment when episodes end.

    :param env: openai gym environment or wrapper thereof.
    :param model: agent taking observations and returning distribution and value prediction.
    :param timesteps_per_actorbatch: integer num timesteps per actor.
    :return: generator of batches of experience.
    """
    t = 0
    o_t = env.reset()

    ep_lengths = []
    ep_returns = []
    ep_returns_unclipped = []

    curr_ep_length = 0
    curr_ep_return = 0.0
    curr_ep_return_unclipped = 0.0

    observations = np.array([o_t for _ in range(timesteps_per_actorbatch)])
    logprobs = np.zeros(timesteps_per_actorbatch, 'float32')
    actions = np.zeros(timesteps_per_actorbatch, 'int64')
    rewards = np.zeros(timesteps_per_actorbatch, 'float32')
    dones = np.zeros(timesteps_per_actorbatch, 'float32')
    value_estimates = np.zeros(timesteps_per_actorbatch+1, 'float32')

    model.eval()
    while True:
        pi_dist_t, vpred_t = model(tc.FloatTensor(o_t).unsqueeze(0))
        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t)

        a_t = a_t.squeeze(0).detach().numpy()
        logprob_a_t = logprob_a_t.squeeze(0).detach().numpy()
        vpred_t = vpred_t.squeeze(0).detach().numpy()

        if t > 0 and t % timesteps_per_actorbatch == 0:
            value_estimates[-1] = vpred_t   # length is timesteps_per_actorbatch+1 for GAE.
            yield {
                "observations": observations,
                "logprobs": logprobs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "value_estimates": value_estimates,
                "ep_lengths": ep_lengths,
                "ep_returns": ep_returns,
                "ep_returns_unclipped": ep_returns_unclipped
            }
            ep_lengths = []
            ep_returns = []
            ep_returns_unclipped = []

        o_tp1, r_t, done_t, info_t = env.step(a_t)

        i = t % timesteps_per_actorbatch
        observations[i] = o_t
        logprobs[i] = logprob_a_t
        actions[i] = a_t
        rewards[i] = r_t
        dones[i] = done_t
        value_estimates[i] = vpred_t

        curr_ep_length += 1
        curr_ep_return += r_t
        curr_ep_return_unclipped += info_t['monitored_reward']

        if done_t:
            ep_lengths.append(curr_ep_length)
            ep_returns.append(curr_ep_return)
            curr_ep_length = 0
            curr_ep_return = 0.0
            if info_t['ale.lives'] == 0:
                ep_returns_unclipped.append(curr_ep_return_unclipped)
                curr_ep_return_unclipped = 0.0
            o_tp1 = env.reset()

        t += 1
        o_t = o_tp1


@tc.no_grad()
def _add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: dictionary containing segments returned by _trajectory_segment_generator.
    :param gamma: float discount factor gamma.
    :param lam: float GAE decay parameter lambda.
    :return: dictionary seg with extra keys and values for advantages and td lambda returns.
    """
    T = len(seg['actions'])
    advantages = np.zeros(T+1, 'float32')
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


def _clip_anneal(clip_param, env_steps_so_far, max_env_steps):
    """
    Anneals PPO clip param to zero over the course of training.

    :param clip_param: float ppo clip param
    :param env_steps_so_far: int environment steps so far
    :param max_env_steps: int max environment steps
    :return: float annealed ppo clip param
    """
    frac_done = (env_steps_so_far / max_env_steps)
    clip_param_annealed = (1.0 - frac_done) * clip_param
    return clip_param_annealed


def _compute_losses(model, batch, clip_param, entcoeff):
    """
    Compute losses for Proximal Policy Optimization (Schulman et al., 2017).

    :param model: agent taking observations and returning distribution and value prediction.
    :param batch: batch from a Dataset with fields obs, acs, logprobs, advs, vtargs.
    :param clip_param: float clip parameter for PPO.
    :param entcoeff: float entropy coefficient for PPO.
    :return: dictionary of losses.
    """
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


@tc.no_grad()
def _metric_update_closure():
    """
    A closure encapsulating some queues and a metric update op.
    The queues are used to maintain sliding window estimates of certain metrics.
    The metric update op updates the queues and computes metric averages across processes.

    :return: function metric_update_op.
    """

    # we use queues for rolling mean of the following metrics:
    metric_names = [
        'ep_lengths',
        'ep_returns',
        'ep_returns_unclipped'
    ]
    buffers = {
        name: deque(maxlen=100) for name in metric_names
    }

    def metric_update_op(
        seg,
        dataset,
        args,
        agent,
        iterations_thus_far,
        env_steps_so_far
    ):

        metrics = dict()
        metrics['iteration'] = iterations_thus_far
        metrics['env_steps'] = env_steps_so_far

        # metrics with heterogeneous counts per process need to be updated using an allgather.
        for name in metric_names:
            metric_locals = seg[name]
            metric_globals = agent.comm.allgather(metric_locals)
            metric_globals_flat = [x for loc in metric_globals for x in loc]
            buffers[name].extend(metric_globals_flat)
            metric_global_mean = np.mean(buffers[name])
            metrics['mean_'+name] = metric_global_mean

        metrics['ev_tdlam_before'] = explained_variance(
            ypred=seg['value_estimates'], y=seg['td_lambda_returns'])

        # metrics with homogenous counts per process can be updated using an allreduce.
        losses = dict()
        n_batches = 0
        for batch in dataset.iterate_once(batch_size=args.optim_batchsize):
            batch_losses = _compute_losses(model=agent.model,
                                           batch=batch,
                                           clip_param=args.ppo_epsilon,
                                           entcoeff=args.entropy_coef)
            n_batches += 1
            batch_losses = dict(
                list(map(lambda kv: (kv[0], kv[1].detach().numpy()), batch_losses.items()))
            )
            for name in batch_losses:
                if name not in losses:
                    losses[name] = 0.0
                losses[name] += batch_losses[name]

        for name in losses:
            loss_local_mean = losses[name] / n_batches
            loss_global_sum = agent.comm.allreduce(loss_local_mean, op=MPI.SUM)
            loss_global_mean = loss_global_sum / agent.comm.Get_size()
            metrics['loss_'+name] = loss_global_mean

        return metrics

    return metric_update_op


def _train(env, agent, args, env_steps_so_far=0):
    """
    Train a reinforcement learning agent by Proximal Policy Optimization.

    :param env: openai gym environment or wrapper thereof.
    :param agent: agent_util.Agent encapsulating model, optimizer, scheduler, comm.
    :param args: argparsed args.
    :return:
    """
    seg_generator = _trajectory_segment_generator(
        env=env, model=agent.model, timesteps_per_actorbatch=args.timesteps_per_actorbatch)
    metric_update_op = _metric_update_closure()

    iterations_thus_far = 0
    while env_steps_so_far < args.env_steps:
        seg = next(seg_generator)
        seg = _add_vtarg_and_adv(seg, gamma=args.discount_gamma, lam=args.gae_lambda)
        seg['advantage_estimates'] = standardize(seg['advantage_estimates'])
        dataset = Dataset(data_map={
            'obs': seg['observations'],
            'acs': seg['actions'],
            'logprobs': seg['logprobs'],
            'vtargs': seg['td_lambda_returns'],
            'advs': seg['advantage_estimates']
        })
        clip_param_annealed = _clip_anneal(
            args.ppo_epsilon, env_steps_so_far, args.env_steps)
        for _ in range(args.optim_epochs):
            for batch in dataset.iterate_once(batch_size=args.optim_batchsize):
                agent.optimizer.zero_grad()
                losses = _compute_losses(model=agent.model,
                                         batch=batch,
                                         clip_param=clip_param_annealed,
                                         entcoeff=args.entropy_coef)
                losses['total_loss'].backward()
                sync_grads(model=agent.model, comm=agent.comm)
                agent.optimizer.step()
                agent.scheduler.step()

        env_steps_so_far += args.timesteps_per_actorbatch * agent.comm.Get_size()
        iterations_thus_far += 1

        metrics = metric_update_op(
            seg, dataset, args, agent, iterations_thus_far, env_steps_so_far)

        if agent.comm.Get_rank() == ROOT_RANK:
            print_metrics(metrics)
            if iterations_thus_far % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir=args.checkpoint_dir,
                                model_name=args.model_name,
                                agent=agent,
                                steps=env_steps_so_far)


def run(env, agent, args):
    """
    Train a reinforcement learning agent by Proximal Policy Optimization.
    On process with MPI rank zero, tries to restore from latest checkpoint.
    Synchronization of model/opt/sched state follows.
    Then runs _train.

    :param env: openai gym environment or wrapper thereof.
    :param agent: agent_util.Agent encapsulating model, optimizer, scheduler, comm.
    :param args: argparsed args.
    :return:
    """
    env_steps_so_far = 0
    if agent.comm.Get_rank() == ROOT_RANK:
        env_steps_so_far = maybe_load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            agent=agent)
    
    env_steps_so_far = agent.comm.bcast(env_steps_so_far, root=ROOT_RANK)
    sync_state(agent=agent, comm=agent.comm, root=ROOT_RANK)
    _train(env=env, agent=agent, args=args, env_steps_so_far=env_steps_so_far)
    
