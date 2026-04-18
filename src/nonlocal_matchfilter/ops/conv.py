import torch.nn as nn
from einops import rearrange

from nonlocal_matchfilter.ops.ffn import GLUFeedForwardNetwork


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels,
        inverse_bottleneck_ratio=4,
        norm=nn.LayerNorm,
        ffn=GLUFeedForwardNetwork,
        activation=nn.GELU,
        selection=nn.Identity,
        scaling=nn.Identity,
        bias=True,
        dropout=0.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=bias,
            groups=channels,
        )
        self.norm = norm(channels)
        self.ffn = ffn(
            channels,
            inverse_bottleneck_ratio * channels,
            activation=activation,
            selection=selection,
            bias=bias,
            dropout=dropout,
        )
        self.scaling = scaling(channels)

    def forward(self, x):
        _batch_size, _channels, height, width = x.shape
        y = self.dwconv(x)
        y = rearrange(y, "b c h w -> b (h w) c")
        y = self.norm(y)
        y = self.ffn(y)
        y = self.scaling(y)
        y = rearrange(y, "b (h w) c -> b c h w", h=height, w=width)
        return x + y
