from collections import namedtuple
from utils.hparam_util import get_hparams
from agents.cnn_policy import CnnPolicy
import torch as tc

Agent = namedtuple('Agent', field_names=['model', 'optimizer', 'scheduler', 'comm'])


def get_agent(args, comm, env):
    hparams = get_hparams(args, env)
    model = CnnPolicy(hparams)
    optimizer = tc.optim.Adam(model.parameters(), lr=args.optim_stepsize, eps=1e-5)

    max_grad_steps = int(args.optim_epochs * args.env_steps // args.optim_batchsize)
    scheduler = tc.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.optim_stepsize,
        total_steps=max_grad_steps,
        pct_start=0.0,
        anneal_strategy='linear',
        cycle_momentum=False,
        div_factor=1.0
    )
    agent = Agent(model=model, optimizer=optimizer, scheduler=scheduler, comm=comm)
    return agent
