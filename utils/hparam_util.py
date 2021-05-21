from collections import namedtuple

HParams = namedtuple('HParams', field_names=['img_chan', 'frame_stack', 'model_size'])


def get_hparams(args):
    img_chan = 1
    frame_stack = 4 if args.frame_stacking else 1
    hps = HParams(img_chan=img_chan, frame_stack=frame_stack, model_size=args.model_size)
    return hps
