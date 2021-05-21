from collections import namedtuple
from utils.hparam_util import get_hparams
from agents.cnn_policy import CnnPolicy
import torch as tc

Agent = namedtuple('Agent', field_names=['model', 'optimizer', 'scheduler'])


def get_agent(args):
    hparams = get_hparams(args)
    agent = CnnPolicy(hparams)
    optimizer = tc.optim.Adam(agent.parameters(), lr=args.optim_stepsize, eps=1e-5)

    max_grad_steps = args.optim_epochs * args.env_steps // args.optim_batchsize
    scheduler = tc.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=max_grad_steps,
        pct_start=0.0,
        anneal_strategy='linear',
        cycle_momentum=False,
        div_factor=1.0
    )
    agent = Agent(agent=agent, optimizer=optimizer, scheduler=scheduler)
    return agent
