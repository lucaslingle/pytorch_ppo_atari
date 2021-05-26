import random
import numpy as np
import torch as tc


def set_seed(args, comm):
    worker_seed = args.rng_seed + 10000 * comm.Get_rank() if args.rng_seed is not None else None
    if worker_seed is not None:
        tc.manual_seed(worker_seed)
        np.random.seed(worker_seed % 2 ** 32)
        random.seed(worker_seed % 2 ** 32)

    return worker_seed
