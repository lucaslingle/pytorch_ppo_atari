from utils.atari_wrappers import make_atari, wrap_deepmind


def get_env(args, seed):
    env = make_atari(args.env_name)
    env.seed(seed)
    env = wrap_deepmind(env, frame_stack=args.frame_stacking, clip_rewards=(args.mode == 'train'))
    env.seed(seed)
    return env
