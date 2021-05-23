from utils.atari_wrappers import make_atari, wrap_deepmind
from utils.monitor_util import Monitor


def get_env(args, seed, comm):
    env = make_atari(args.env_name)
    env.seed(seed)
    env = Monitor(env, monitoring_dir=args.monitoring_dir, model_name=args.model_name, comm=comm)
    env = wrap_deepmind(env, frame_stack=args.frame_stacking, clip_rewards=(args.mode == 'train'))
    env.seed(seed)
    return env
