import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        out_channels: int = None,
        activation=nn.GELU,
        selection=nn.Identity,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, inner_channels, bias=bias),
            activation(),
            selection(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels, out_channels, bias=bias),
        )

    def forward(self, x):
        return self.ffn(x)


class GLUFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        out_channels: int = None,
        activation=nn.GELU,
        selection=nn.Identity,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.linear11 = nn.Linear(in_channels, inner_channels, bias=bias)
        self.linear12 = nn.Linear(in_channels, inner_channels, bias=bias)
        self.activation = activation()
        self.selection = selection(inner_channels)
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_channels, out_channels)

    def forward(self, x):
        x = self.activation(self.linear11(x)) * self.linear12(x)
        x = self.selection(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x
