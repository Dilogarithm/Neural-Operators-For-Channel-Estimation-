from functools import partialmethod
from typing import Tuple, List, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)


from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class GaborConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes=None,
        n_fft=64,
        hop_length=32,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

        # simple learnable multiplier
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, n_fft // 2 + 1)
        )

    def forward(self, x):
         # x shape: (batch, channels, T)

        B, C, T = x.shape
        device = x.device
        window = self.window.to(device)

        outputs = torch.zeros(B, self.out_channels, T, device=device)

        for i in range(self.in_channels):
            U = torch.stft(
                x[:, i, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True
            )

           #  U: (batch, freq, time)

            for o in range(self.out_channels):
                Y = self.weight[o, i].unsqueeze(-1) * U

                y = torch.istft(
                    Y,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=window,
                    length=T
                )

                outputs[:, o, :] += y

        return outputs

class GNO(BaseModel):

    def __init__(self, in_channels, out_channels, hidden_channels, n_layers=4):
        super().__init__()

        self.lifting = nn.Conv1d(in_channels, hidden_channels, 1)

        self.layers = nn.ModuleList([
            GaborConv1D(hidden_channels, hidden_channels)
            for _ in range(n_layers)
        ])

        self.projection = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, **kwargs):
        x = self.lifting(x)

        for layer in self.layers:
            x = x + layer(x)   # residual like FNO

        x = self.projection(x)
        return x

def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    See the Spherical FNO class in neuralop/models/sfno.py for an example.

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )


