import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Implementation of Proximal Policy Optimization for Atari 2600 games.""")

    parser.add_argument("--env_name", type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument("--env_steps", type=int, default=40e6)
    parser.add_argument("--model_name", type=str, default='model-paper-defaults')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    parser.add_argument("--timesteps_per_actorbatch", type=int, default=128)
    parser.add_argument("--optim_batchsize", type=int, default=32)
    parser.add_argument("--optim_epochs", type=int, default=3)
    parser.add_argument("--optim_stepsize", type=float, default=2.5e-4)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--model_size", type=str, default='small')
    args = parser.parse_args()

    return args
