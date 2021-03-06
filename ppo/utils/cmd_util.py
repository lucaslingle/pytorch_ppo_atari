import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Implementation of Proximal Policy Optimization for Atari 2600 games.""")

    parser.add_argument("--env_name", type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument("--env_steps", type=int, default=11e6)
    parser.add_argument("--mode", choices=['train', 'play', 'record'], default='train')
    parser.add_argument("--model_name", type=str, default='defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--timesteps_per_actorbatch", type=int, default=256)
    parser.add_argument("--optim_batchsize", type=int, default=64)
    parser.add_argument("--optim_epochs", type=int, default=3)
    parser.add_argument("--optim_stepsize", type=float, default=1.0e-3)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--model_size", choices=['small', 'large'], default='large')
    parser.add_argument("--no_frame_stacking", dest='frame_stacking', action='store_false')
    parser.add_argument("--monitoring_dir", type=str, default='monitoring')
    parser.add_argument("--asset_dir", type=str, default='assets')
    parser.add_argument("--rng_seed", type=int, default=0)
    args = parser.parse_args()

    return args
