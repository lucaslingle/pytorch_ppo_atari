import torch as tc


class ConvPreprocess(tc.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[1] == x.shape[2] == 84
        x = x / 255.
        x = x.permute(0, 3, 1, 2)
        x = x.detach()
        return x