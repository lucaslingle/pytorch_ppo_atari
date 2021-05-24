from collections import namedtuple

HParams = namedtuple('HParams', field_names=['img_chan', 'model_size', 'num_actions'])


def get_hparams(args, env):
    hps = HParams(
        img_chan=env.observation_space.shape[-1],
        model_size=args.model_size,
        num_actions=env.action_space.n)

    return hps
