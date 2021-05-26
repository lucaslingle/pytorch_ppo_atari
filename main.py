from ppo.utils.cmd_util import parse_args
from ppo.utils.comm_util import get_comm
from ppo.utils.rand_util import set_seed
from ppo.utils.env_util import get_env
from ppo.utils.agent_util import get_agent
from ppo.runners.train import run as train_run
from ppo.runners.play import run as play_run
from ppo.runners.record import run as record_run


def main(args):
    comm = get_comm()

    worker_seed = set_seed(args, comm)
    env = get_env(args, comm, worker_seed)
    agent = get_agent(args, comm, env)

    runners = {
        'train': train_run,
        'play': play_run,
        'record': record_run
    }
    run = runners[args.mode]
    run(env, agent, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
