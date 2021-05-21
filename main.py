import torch as tc
from utils.cmd_util import parse_args
from utils.comm_util import get_comm
from utils.rand_util import set_seed
from utils.env_util import get_env
from utils.hparam_util import get_hparams
from agents.cnn_policy import CnnPolicy
from runners.train import Trainer
from runners.play import Player


def main(args):
    comm = get_comm()

    worker_seed = set_seed(args, comm)
    env = get_env(args, worker_seed)
    hparams = get_hparams(args)
    agent = CnnPolicy(hparams)

    runners = {
        'train': Trainer(env, agent, args),
        'player': Player(env, agent, args)
    }
    runner = runners[args.mode]
    runner.run()


if __name__ == '__main__':
    args = parse_args()
    main(args)


