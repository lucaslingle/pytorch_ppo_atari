import torch as tc


def get_device():
    if tc.cuda.is_available():
        return "cuda"
    return "cpu"
