from ppo.utils.atari_wrappers import make_atari, wrap_deepmind
from ppo.utils.monitor_util import Monitor


def get_env(args, comm, seed):
    env = make_atari(args.env_name)
    env.seed(seed)

    if args.mode == 'train':
        env = Monitor(env, monitoring_dir=args.monitoring_dir, model_name=args.model_name, comm=comm)

    # Mnih et al., 2015 -> Methods -> Training Details.
    env = wrap_deepmind(env, frame_stack=args.frame_stacking,
                        clip_rewards=(args.mode == 'train'),
                        episode_life=(args.mode == 'train'))
    env.seed(seed)

    return env
