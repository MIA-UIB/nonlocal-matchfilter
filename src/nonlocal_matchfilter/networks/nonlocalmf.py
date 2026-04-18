import torch
import torch.nn as nn
from deform_neighbourhood_sampling.ops import deform_neighbourhood
from einops import rearrange
from patchmatch import patch_match, stack_matches

from nonlocal_matchfilter.ops.conv import ConvNeXtBlock
from nonlocal_matchfilter.ops.ffn import FeedForwardNetwork
from nonlocal_matchfilter.ops.padding import InputPadder


class OffsetCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_channels: int,
        offset_groups: int,
        num_neighbours: int,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, embedding_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(embedding_channels, embedding_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(embedding_channels, embedding_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(embedding_channels, embedding_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(embedding_channels, embedding_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                embedding_channels,
                2 * offset_groups * num_neighbours,
                kernel_size=1,
                padding=0,
            ),
        )
        nn.init.zeros_(self.cnn[-1].weight.data)
        nn.init.zeros_(self.cnn[-1].bias.data)

    def forward(self, x):
        return self.cnn(x)


class NonlocalBlock(nn.Module):
    def __init__(
        self,
        n_features,
        neighbours,
        max_search_dist,
        n_features_offsets=None,
        bias=False,
        groups=1,
    ):
        super().__init__()

        self.num_features = n_features
        self.neighbours = neighbours
        self.max_search_dist = max_search_dist
        self.groups = groups
        if n_features_offsets is None:
            n_features_offsets = n_features

        self.encoder = nn.Sequential(
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            )
        )

        self.decoder = nn.Sequential(
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            )
        )

        num_neighbours = neighbours[0] * neighbours[1]
        self.offset_cnn = OffsetCNN(
            n_features,
            n_features_offsets,
            offset_groups=1,
            num_neighbours=num_neighbours,
        )

        self.dct_sub = nn.Conv2d(
            n_features * num_neighbours,
            n_features * num_neighbours,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            groups=n_features // groups,
        )
        self.wiener_sub = nn.Sequential(
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                groups=n_features * num_neighbours,
            ),
            nn.ReLU(),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                groups=n_features * num_neighbours,
            ),
            nn.ReLU(),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                groups=n_features * num_neighbours,
            ),
            nn.Sigmoid(),
        )

        self.inv_dct_sub = nn.Sequential(
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                groups=n_features // groups,
            ),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
        )

    def forward(self, x):
        _, _, height, width = x.shape
        num_neighbours = self.neighbours[0] * self.neighbours[1]
        x = self.encoder(x)

        offsets = self.offset_cnn(x)
        offsets = self.max_search_dist * torch.tanh(offsets)

        x_blocks = deform_neighbourhood(
            x,
            offsets,
            neighbourhood_size=self.neighbours,
            stride=1,
            padding=(self.neighbours[0] // 2, self.neighbours[1] // 2),
            dilation=1,
            offset_groups=1,
        )
        x_blocks = rearrange(
            x_blocks,
            "b (g f K) h w -> b (f K g) h w",
            g=self.groups,
            f=self.num_features // self.groups,
            K=num_neighbours,
        )
        dct_coeffs = self.dct_sub(x_blocks)
        wiener_coeffs = self.wiener_sub(x_blocks)
        x = self.inv_dct_sub(wiener_coeffs * dct_coeffs)

        x = self.decoder(x)
        return x


class NonlocalBlockCherel(nn.Module):
    def __init__(
        self,
        n_features,
        neighbours,
        bias=False,
        groups=1,
    ):
        super().__init__()

        self.num_features = n_features
        self.neighbours = neighbours
        self.groups = groups

        self.neighbours = neighbours
        num_neighbours = neighbours

        self.dct_sub = nn.Conv2d(
            n_features * num_neighbours,
            n_features * num_neighbours,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            groups=n_features // groups,
        )
        self.wiener_sub = nn.Sequential(
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                groups=n_features * num_neighbours,
            ),
            nn.ReLU(),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                groups=n_features * num_neighbours,
            ),
            nn.ReLU(),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                n_features * num_neighbours,
                n_features * num_neighbours,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                groups=n_features * num_neighbours,
            ),
            nn.Sigmoid(),
        )

        self.inv_dct_sub = nn.Conv2d(
            n_features * num_neighbours,
            n_features * num_neighbours,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            groups=n_features // groups,
        )

        self.aggregation_sub = nn.Conv2d(
            n_features * num_neighbours,
            n_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        _, _, height, width = x.shape
        num_neighbours = self.neighbours

        # Hardcoded to 15 neighbors. Other values require recompilation of the patch_match CUDA extension.
        shift_map, cost_map = patch_match(x, x, 3, 5)
        x_blocks = stack_matches(x, shift_map)
        x_blocks = rearrange(
            x_blocks,
            "b (g f K) h w -> b (f K g) h w",
            g=self.groups,
            f=self.num_features // self.groups,
            K=num_neighbours,
        )
        dct_coeffs = self.dct_sub(x_blocks)
        wiener_coeffs = self.wiener_sub(x_blocks)
        x_blocks_filtered = self.inv_dct_sub(wiener_coeffs * dct_coeffs)
        x = self.aggregation_sub(x_blocks_filtered)
        return x


class SimpleBlockMatchingNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        n_features=64,
        neighbours=(5, 5),
        max_search_dist: float = 9.0,
    ):
        super().__init__()

        self.in_channels = input_channels
        self.out_channels = output_channels

        self.head = nn.Conv2d(
            input_channels, n_features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.filter = NonlocalBlock(
            n_features=n_features,
            neighbours=neighbours,
            max_search_dist=max_search_dist,
            bias=False,
        )

        self.tail = nn.Conv2d(
            n_features, output_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        x = self.head(x)
        x = self.filter(x) + x
        return self.tail(x)


class CherelBlockMatchingNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        n_features=64,
        neighbours=15,
    ):
        super().__init__()

        self.in_channels = input_channels
        self.out_channels = output_channels

        self.head = nn.Conv2d(
            input_channels, n_features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.filter = nn.Sequential(
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            NonlocalBlockCherel(
                n_features=n_features,
                neighbours=neighbours,
                bias=False,
            ),
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
        )

        self.tail = nn.Conv2d(
            n_features, output_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        x = self.head(x)
        x = self.filter(x) + x
        return self.tail(x)


class SimpleBlockMatchingUNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        n_features=64,
        neighbours=None,
        max_search_dist: float = 9.0,
        bias: bool = False,
    ):
        super().__init__()

        if neighbours is None:
            neighbours = {"scale1": [5, 5], "scale2": [3, 5], "scale3": [3, 3]}

        self.in_channels = input_channels
        self.out_channels = output_channels

        self.max_search_dist = max_search_dist
        self.neighbours = neighbours

        self.head = nn.Conv2d(
            input_channels,
            n_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        self.down1 = nn.Sequential(
            NonlocalBlock(
                n_features=n_features,
                neighbours=neighbours["scale1"],
                max_search_dist=max_search_dist,
                bias=bias,
            ),
            nn.Conv2d(
                n_features,
                n_features * 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
        )

        self.down2 = nn.Sequential(
            NonlocalBlock(
                n_features=n_features * 2,
                neighbours=neighbours["scale2"],
                max_search_dist=max_search_dist,
                bias=bias,
            ),
            nn.Conv2d(
                n_features * 2,
                n_features * 4,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
        )

        self.body = nn.Sequential(
            NonlocalBlock(
                n_features=n_features * 4,
                neighbours=neighbours["scale3"],
                max_search_dist=max_search_dist,
                bias=bias,
            )
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                n_features * 4,
                n_features * 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
            NonlocalBlock(
                n_features=n_features * 2,
                neighbours=neighbours["scale2"],
                max_search_dist=max_search_dist,
                bias=bias,
            ),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                n_features * 2,
                n_features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
            NonlocalBlock(
                n_features=n_features,
                neighbours=neighbours["scale1"],
                max_search_dist=max_search_dist,
                bias=bias,
            ),
        )

        self.tail = nn.Conv2d(
            n_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        _b, _c, height, width = x.shape
        input_padder = InputPadder((height, width), to_multiple=4, mode="reflect")
        x = input_padder.pad(x)

        feat0 = self.head(x)
        feat1 = self.down1(feat0)
        feat2 = self.down2(feat1)
        feat2 = self.body(feat2) + feat2
        feat1 = self.up2(feat2) + feat1
        feat0 = self.up1(feat1) + feat0
        feat0 = self.tail(feat0)
        feat0 = input_padder.unpad(feat0)
        return feat0


class CherelBlockMatchingUNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        n_features=64,
        bias: bool = False,
    ):
        super().__init__()

        self.in_channels = input_channels
        self.out_channels = output_channels

        self.head = nn.Conv2d(
            input_channels,
            n_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        self.down1 = nn.Sequential(
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            NonlocalBlockCherel(
                n_features=n_features,
                neighbours=15,
                bias=False,
            ),
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            nn.Conv2d(
                n_features,
                n_features * 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
        )

        self.down2 = nn.Sequential(
            ConvNeXtBlock(
                n_features * 2,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            NonlocalBlockCherel(
                n_features=n_features * 2,
                neighbours=15,
                bias=False,
            ),
            ConvNeXtBlock(
                n_features * 2,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            nn.Conv2d(
                n_features * 2,
                n_features * 4,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
        )

        self.body = nn.Sequential(
            ConvNeXtBlock(
                n_features * 4,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            NonlocalBlockCherel(
                n_features=n_features * 4,
                neighbours=15,
                bias=False,
            ),
            ConvNeXtBlock(
                n_features * 4,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                n_features * 4,
                n_features * 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
            ConvNeXtBlock(
                n_features * 2,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            NonlocalBlockCherel(
                n_features=n_features * 2,
                neighbours=15,
                bias=False,
            ),
            ConvNeXtBlock(
                n_features * 2,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                n_features * 2,
                n_features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            ),
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
            NonlocalBlockCherel(
                n_features=n_features,
                neighbours=15,
                bias=False,
            ),
            ConvNeXtBlock(
                n_features,
                inverse_bottleneck_ratio=2,
                norm=nn.Identity,
                activation=nn.ReLU,
                ffn=FeedForwardNetwork,
                selection=nn.Identity,
                scaling=nn.Identity,
                bias=False,
            ),
        )

        self.tail = nn.Conv2d(
            n_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        feat0 = self.head(x)
        feat1 = self.down1(feat0)
        feat2 = self.down2(feat1)
        feat2 = self.body(feat2) + feat2
        feat1 = self.up2(feat2) + feat1
        feat0 = self.up1(feat1) + feat0
        return self.tail(feat0)
